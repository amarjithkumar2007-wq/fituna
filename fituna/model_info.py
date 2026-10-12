# SPDX-License-Identifier: MIT
# Copyright (c) 2026 FiTuna contributors
"""fituna.model_info
====================

Ensures a base F16/F32 GGUF exists for the input model (converting from a HF
directory via llama.cpp's convert script if needed) and reads back model
metadata (architecture, layer count, param count) plus a cheap fingerprint
used as a cache key.

GGUF metadata is parsed directly from the file with ``struct`` (no llama.cpp
binary exposes a "dump metadata as JSON" contract we can rely on across
versions), following the format at
https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
"""

from __future__ import annotations

import hashlib
import os
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any, BinaryIO

from fituna.config import BinaryPaths, FiTunaError, ModelConversionError, ModelInfo

# --- GGUF value type ids (see ggml/docs/gguf.md) ---------------------------
_GGUF_MAGIC = b"GGUF"
_T_UINT8 = 0
_T_INT8 = 1
_T_UINT16 = 2
_T_INT16 = 3
_T_UINT32 = 4
_T_INT32 = 5
_T_FLOAT32 = 6
_T_BOOL = 7
_T_STRING = 8
_T_ARRAY = 9
_T_UINT64 = 10
_T_INT64 = 11
_T_FLOAT64 = 12

# type id -> (struct format, byte size) for fixed-width scalars
_SCALAR_FORMATS: dict[int, tuple[str, int]] = {
    _T_UINT8: ("<B", 1),
    _T_INT8: ("<b", 1),
    _T_UINT16: ("<H", 2),
    _T_INT16: ("<h", 2),
    _T_UINT32: ("<I", 4),
    _T_INT32: ("<i", 4),
    _T_FLOAT32: ("<f", 4),
    _T_BOOL: ("<B", 1),
    _T_UINT64: ("<Q", 8),
    _T_INT64: ("<q", 8),
    _T_FLOAT64: ("<d", 8),
}


def _read_exact(f: BinaryIO, n: int, file_size: int) -> bytes:
    # GGUF string/array length fields are attacker-controllable 64-bit values
    # (this file is untrusted input -- it's whatever the user pointed --model
    # at, e.g. a downloaded HF/GGUF file). Bound every read against the
    # actual remaining file size *before* calling f.read(n), so a fabricated
    # length can fail immediately instead of attempting a multi-GB
    # allocation: reproduced pre-fix, a single crafted length field drove
    # peak RSS to ~10GB before Python's own MemoryError kicked in.
    if n < 0 or f.tell() + n > file_size:
        raise FiTunaError(
            "GGUF field length exceeds remaining file size -- file is "
            "truncated or its metadata is corrupt/malicious"
        )
    data = f.read(n)
    if len(data) != n:
        raise FiTunaError("unexpected end of file while parsing GGUF header")
    return data


def _read_u32(f: BinaryIO, file_size: int) -> int:
    return struct.unpack("<I", _read_exact(f, 4, file_size))[0]


def _read_u64(f: BinaryIO, file_size: int) -> int:
    return struct.unpack("<Q", _read_exact(f, 8, file_size))[0]


def _read_length(f: BinaryIO, version: int, file_size: int) -> int:
    # GGUF v1 used uint32 for string/array lengths; v2+ uses uint64.
    return _read_u64(f, file_size) if version >= 2 else _read_u32(f, file_size)


def _read_string(f: BinaryIO, version: int, file_size: int) -> str:
    length = _read_length(f, version, file_size)
    return _read_exact(f, length, file_size).decode("utf-8", errors="replace")


def _read_value(f: BinaryIO, value_type: int, version: int, file_size: int) -> Any:
    if value_type == _T_STRING:
        return _read_string(f, version, file_size)
    if value_type == _T_ARRAY:
        elem_type = _read_u32(f, file_size)
        count = _read_length(f, version, file_size)
        # Same untrusted-length concern as strings: even with the read-level
        # bound above, a huge count of cheap elements (e.g. count=10**9
        # empty strings) would still balloon into a giant Python list before
        # any individual read fails. Each array element needs at least 1
        # byte in the file (a scalar) or 8 bytes (a nested string's own
        # length prefix), so count can never legitimately exceed file_size --
        # reject it up front instead of discovering that partway through.
        if count > file_size:
            raise FiTunaError(
                "GGUF array element count exceeds what the file could "
                "possibly contain -- file is truncated or its metadata is "
                "corrupt/malicious"
            )
        return [_read_value(f, elem_type, version, file_size) for _ in range(count)]
    fmt_size = _SCALAR_FORMATS.get(value_type)
    if fmt_size is None:
        raise FiTunaError(f"unknown GGUF value type id {value_type}")
    fmt, size = fmt_size
    return struct.unpack(fmt, _read_exact(f, size, file_size))[0]


def ensure_base_gguf(model_path: Path, work_dir: Path, binaries: BinaryPaths) -> Path:
    """If model_path is already a .gguf file, return it unchanged. If it's an
    HF-format directory, invoke binaries.convert_script via subprocess to
    produce work_dir/base-f16.gguf and return that path.

    Raises ModelConversionError on subprocess failure or missing convert_script.
    """
    model_path = Path(model_path)

    if model_path.is_file() and model_path.suffix.lower() == ".gguf":
        return model_path

    if not model_path.is_dir():
        raise ModelConversionError(
            f"{model_path} is neither a .gguf file nor an HF-format model "
            "directory -- nothing to convert."
        )

    if binaries.convert_script is None:
        raise ModelConversionError(
            f"{model_path} looks like an HF-format directory but no "
            "convert_script is configured. Point --llama-bin-dir at a "
            "llama.cpp checkout containing convert_hf_to_gguf.py, or pass a "
            ".gguf file directly."
        )

    convert_script = Path(binaries.convert_script)
    if not convert_script.is_file():
        raise ModelConversionError(f"convert_script not found: {convert_script}")

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    out_path = work_dir / "base-f16.gguf"

    # Idempotent, like quantize.py: skip the (slow) conversion subprocess if
    # a previous run already produced it. This also matters for --resume --
    # without it, base-f16.gguf's mtime changes on every run, so
    # model_fingerprint() never repeats and the cache can never hit for an
    # HF-directory input.
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path

    cmd = [
        sys.executable,
        str(convert_script),
        str(model_path),
        "--outfile",
        str(out_path),
        "--outtype",
        "f16",
    ]
    try:
        # encoding/errors explicit: see hardware.py's _run for why.
        proc = subprocess.run(
            cmd, capture_output=True, text=True, check=False,
            encoding="utf-8", errors="replace",
        )
    except OSError as exc:
        raise ModelConversionError(
            f"failed to launch convert script {convert_script}: {exc}"
        ) from exc

    if proc.returncode != 0:
        raise ModelConversionError(
            f"convert script exited with code {proc.returncode} for {model_path}\n"
            f"command: {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )

    if not out_path.is_file():
        raise ModelConversionError(
            f"convert script exited 0 but did not produce {out_path}"
        )

    return out_path


def read_model_info(gguf_path: Path, binaries: BinaryPaths) -> ModelInfo:
    """Read architecture / n_layers / n_params from the GGUF header.

    n_params is computed by summing tensor element counts from the tensor
    info section (always present and accurate), rather than trusting an
    optional 'general.parameter_count' metadata key that not all writers set.
    """
    gguf_path = Path(gguf_path)
    with open(gguf_path, "rb") as f:
        file_size = os.fstat(f.fileno()).st_size

        magic = _read_exact(f, 4, file_size)
        if magic != _GGUF_MAGIC:
            raise FiTunaError(f"{gguf_path} is not a valid GGUF file (bad magic)")

        version = _read_u32(f, file_size)
        if version >= 2:
            tensor_count = _read_u64(f, file_size)
            kv_count = _read_u64(f, file_size)
        else:
            tensor_count = _read_u32(f, file_size)
            kv_count = _read_u32(f, file_size)

        metadata: dict[str, Any] = {}
        for _ in range(kv_count):
            key = _read_string(f, version, file_size)
            value_type = _read_u32(f, file_size)
            metadata[key] = _read_value(f, value_type, version, file_size)

        n_params = 0
        for _ in range(tensor_count):
            _read_string(f, version, file_size)  # tensor name, unused
            n_dims = _read_u32(f, file_size)
            nelements = 1
            for _ in range(n_dims):
                dim = _read_u64(f, file_size) if version >= 2 else _read_u32(f, file_size)
                nelements *= dim
            _read_u32(f, file_size)  # ggml tensor dtype, unused for param count
            _read_u64(f, file_size)  # data offset, unused
            n_params += nelements

    architecture = metadata.get("general.architecture")
    if not isinstance(architecture, str) or not architecture:
        raise FiTunaError(
            f"{gguf_path}: missing or invalid 'general.architecture' metadata key"
        )

    n_layers = metadata.get(f"{architecture}.block_count")
    if n_layers is None:
        raise FiTunaError(
            f"{gguf_path}: missing '{architecture}.block_count' metadata key"
        )

    file_type = metadata.get("general.file_type")

    return ModelInfo(
        architecture=architecture,
        n_layers=int(n_layers),
        n_params=int(n_params),
        base_gguf_path=gguf_path,
        file_type=int(file_type) if isinstance(file_type, int) else None,
    )


# LLAMA_FTYPE values that are NOT quantized (llama.cpp llama.h): F32, F16, BF16.
_UNQUANTIZED_FILE_TYPES = frozenset({0, 1, 32})


def is_already_quantized(info: ModelInfo) -> bool:
    """True when the base GGUF's 'general.file_type' says it is already a
    quantized file. Re-quantizing a quantized GGUF silently degrades quality
    twice AND makes the perplexity 'baseline' a quantized model, so callers
    should warn loudly. None (key absent) is treated as not-quantized."""
    return info.file_type is not None and info.file_type not in _UNQUANTIZED_FILE_TYPES


def model_fingerprint(path: Path) -> str:
    """sha256(f'{path.name}:{size}:{mtime}') -- a low-cost identity fingerprint
    used as a cache key, deliberately NOT a full-file hash (large GGUFs can be
    tens of GB).
    """
    path = Path(path)
    st = path.stat()
    raw = f"{path.name}:{st.st_size}:{st.st_mtime}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# --- self-check -------------------------------------------------------------
# no test framework needed for a single module's worth of parsing
# logic -- an assert-based demo() that builds a minimal synthetic GGUF is the
# smallest thing that fails if the binary parser or fingerprint break.
def _write_gguf_string(buf: bytearray, s: str) -> None:
    encoded = s.encode("utf-8")
    buf += struct.pack("<Q", len(encoded))
    buf += encoded


def _build_synthetic_gguf() -> bytes:
    """A hand-built, minimal but spec-valid GGUF v3 file: one string kv
    (general.architecture='testarch'), one uint32 kv (testarch.block_count=32),
    and one tensor with dims [4, 8] (-> 32 elements)."""
    buf = bytearray()
    buf += _GGUF_MAGIC
    buf += struct.pack("<I", 3)  # version
    buf += struct.pack("<Q", 1)  # tensor_count
    buf += struct.pack("<Q", 2)  # kv_count

    # kv 1: general.architecture (string)
    _write_gguf_string(buf, "general.architecture")
    buf += struct.pack("<I", _T_STRING)
    _write_gguf_string(buf, "testarch")

    # kv 2: testarch.block_count (uint32)
    _write_gguf_string(buf, "testarch.block_count")
    buf += struct.pack("<I", _T_UINT32)
    buf += struct.pack("<I", 32)

    # tensor 0: name "test.weight", dims [4, 8], type 0 (F32), offset 0
    _write_gguf_string(buf, "test.weight")
    buf += struct.pack("<I", 2)  # n_dims
    buf += struct.pack("<Q", 4)
    buf += struct.pack("<Q", 8)
    buf += struct.pack("<I", 0)  # ggml tensor type
    buf += struct.pack("<Q", 0)  # offset

    return bytes(buf)


def demo() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # --- read_model_info against a synthetic GGUF ---
        gguf_path = tmp_path / "synthetic.gguf"
        gguf_path.write_bytes(_build_synthetic_gguf())

        info = read_model_info(gguf_path, binaries=None)  # type: ignore[arg-type]
        assert info.architecture == "testarch", info.architecture
        assert info.n_layers == 32, info.n_layers
        assert info.n_params == 32, info.n_params  # 4*8
        assert info.base_gguf_path == gguf_path

        # bad magic -> FiTunaError
        bad_path = tmp_path / "bad.gguf"
        bad_path.write_bytes(b"NOPE" + b"\x00" * 16)
        try:
            read_model_info(bad_path, binaries=None)  # type: ignore[arg-type]
            raise AssertionError("expected FiTunaError for bad magic")
        except FiTunaError:
            pass

        # --- model_fingerprint: deterministic, differs by name ---
        fp1 = model_fingerprint(gguf_path)
        fp2 = model_fingerprint(gguf_path)
        assert fp1 == fp2
        assert len(fp1) == 64  # hex sha256

        other_path = tmp_path / "other.gguf"
        other_path.write_bytes(_build_synthetic_gguf())
        assert model_fingerprint(other_path) != fp1  # different name

        # --- ensure_base_gguf: passthrough for existing .gguf ---
        binaries = BinaryPaths(
            llama_quantize=tmp_path / "llama-quantize",
            llama_bench=tmp_path / "llama-bench",
            llama_perplexity=tmp_path / "llama-perplexity",
        )
        result = ensure_base_gguf(gguf_path, tmp_path / "work", binaries)
        assert result == gguf_path

        # --- ensure_base_gguf: neither file nor dir -> ModelConversionError ---
        missing = tmp_path / "does-not-exist"
        try:
            ensure_base_gguf(missing, tmp_path / "work", binaries)
            raise AssertionError("expected ModelConversionError for missing path")
        except ModelConversionError:
            pass

        # --- ensure_base_gguf: HF dir but no convert_script -> ModelConversionError ---
        hf_dir = tmp_path / "hf_model"
        hf_dir.mkdir()
        (hf_dir / "config.json").write_text("{}")
        try:
            ensure_base_gguf(hf_dir, tmp_path / "work", binaries)
            raise AssertionError("expected ModelConversionError for missing convert_script")
        except ModelConversionError:
            pass

    print("model_info self-check: OK")


if __name__ == "__main__":
    demo()
