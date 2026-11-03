# SPDX-License-Identifier: MIT
# Copyright (c) 2026 FiTuna contributors
"""GGUF header parser tests for fituna.model_info.

read_model_info() parses untrusted input (whatever the user pointed --model
at), so these tests build small synthetic GGUF byte strings -- most of them
slightly broken -- and check both the happy paths (v3 and v1 headers) and
every guard: bad magic, truncation, attacker-controlled length fields,
unknown value types, and missing required metadata. All fixtures live under
tmp_path; no network and no llama.cpp binaries are involved.
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from fituna.config import FiTunaError, ModelInfo
from fituna.model_info import (
    is_already_quantized,
    model_fingerprint,
    read_model_info,
)

_T_UINT8 = 0
_T_UINT32 = 4
_T_STRING = 8
_T_ARRAY = 9


def _pack_len(version: int, n: int) -> bytes:
    # GGUF v1 used uint32 for string/array lengths and counts; v2+ uses uint64.
    return struct.pack("<Q", n) if version >= 2 else struct.pack("<I", n)


def _write_string(buf: bytearray, version: int, s: str) -> None:
    encoded = s.encode("utf-8")
    buf += _pack_len(version, len(encoded))
    buf += encoded


def _string_value(version: int, s: str) -> bytes:
    buf = bytearray()
    _write_string(buf, version, s)
    return bytes(buf)


def build_gguf(
    version: int = 3,
    kvs: list[tuple[str, int, bytes]] | None = None,
    tensors: list[tuple[str, list[int], int, int]] | None = None,
) -> bytes:
    """Minimal spec-valid GGUF builder. kvs are (key, value_type, raw_value)
    triples; tensors are (name, dims, ggml_dtype, offset)."""
    kvs = kvs if kvs is not None else []
    tensors = tensors if tensors is not None else []
    buf = bytearray()
    buf += b"GGUF"
    buf += struct.pack("<I", version)
    buf += _pack_len(version, len(tensors))
    buf += _pack_len(version, len(kvs))
    for key, value_type, raw in kvs:
        _write_string(buf, version, key)
        buf += struct.pack("<I", value_type)
        buf += raw
    for name, dims, dtype, offset in tensors:
        _write_string(buf, version, name)
        buf += struct.pack("<I", len(dims))
        for dim in dims:
            buf += _pack_len(version, dim)
        buf += struct.pack("<I", dtype)
        buf += struct.pack("<Q", offset)
    return bytes(buf)


def valid_kvs(version: int) -> list[tuple[str, int, bytes]]:
    return [
        ("general.architecture", _T_STRING, _string_value(version, "testarch")),
        ("testarch.block_count", _T_UINT32, struct.pack("<I", 32)),
        ("general.file_type", _T_UINT32, struct.pack("<I", 1)),
    ]


def write_gguf(tmp_path: Path, data: bytes, name: str = "model.gguf") -> Path:
    path = tmp_path / name
    path.write_bytes(data)
    return path


def test_valid_gguf_v3_parses_metadata_and_param_count(tmp_path):
    path = write_gguf(
        tmp_path,
        build_gguf(
            version=3,
            kvs=valid_kvs(3),
            tensors=[("test.weight", [4, 8], 0, 0)],
        ),
    )
    info = read_model_info(path, binaries=None)
    assert info.architecture == "testarch"
    assert info.n_layers == 32
    assert info.n_params == 32  # 4 * 8
    assert info.file_type == 1
    assert info.base_gguf_path == path


def test_valid_gguf_v1_parses_the_same_way(tmp_path):
    # v1 headers use uint32 for counts, string lengths, and tensor dims.
    path = write_gguf(
        tmp_path,
        build_gguf(
            version=1,
            kvs=valid_kvs(1),
            tensors=[("test.weight", [4, 8], 0, 0)],
        ),
    )
    info = read_model_info(path, binaries=None)
    assert info.architecture == "testarch"
    assert info.n_layers == 32
    assert info.n_params == 32
    assert info.file_type == 1


def test_bad_magic_raises(tmp_path):
    path = write_gguf(tmp_path, b"NOPE" + b"\x00" * 32)
    with pytest.raises(FiTunaError, match="bad magic"):
        read_model_info(path, binaries=None)


def test_header_truncated_midway_raises(tmp_path):
    data = build_gguf(version=3, kvs=valid_kvs(3), tensors=[("t", [2], 0, 0)])
    path = write_gguf(tmp_path, data[:20])
    with pytest.raises(FiTunaError):
        read_model_info(path, binaries=None)


def test_oversized_string_length_is_rejected_before_reading(tmp_path):
    # A key whose declared length (10**12) dwarfs the file must fail on the
    # bounds check, not on a multi-GB read attempt.
    buf = bytearray()
    buf += b"GGUF"
    buf += struct.pack("<I", 3)
    buf += struct.pack("<Q", 0)  # tensor_count
    buf += struct.pack("<Q", 1)  # kv_count
    buf += struct.pack("<Q", 10**12)  # declared key length
    buf += b"tiny"
    path = write_gguf(tmp_path, bytes(buf))
    with pytest.raises(FiTunaError, match="length exceeds remaining file size"):
        read_model_info(path, binaries=None)


def test_oversized_array_count_is_rejected(tmp_path):
    raw_array = struct.pack("<I", _T_UINT8) + _pack_len(3, 10**9)
    data = build_gguf(
        version=3, kvs=[("evil.array", _T_ARRAY, raw_array)], tensors=[]
    )
    path = write_gguf(tmp_path, data)
    with pytest.raises(FiTunaError, match="array element count exceeds"):
        read_model_info(path, binaries=None)


def test_unknown_value_type_id_raises(tmp_path):
    data = build_gguf(
        version=3, kvs=[("some.key", 99, b"")], tensors=[]
    )
    path = write_gguf(tmp_path, data)
    with pytest.raises(FiTunaError, match="unknown GGUF value type id 99"):
        read_model_info(path, binaries=None)


def test_missing_architecture_raises(tmp_path):
    data = build_gguf(
        version=3,
        kvs=[("testarch.block_count", _T_UINT32, struct.pack("<I", 32))],
        tensors=[],
    )
    path = write_gguf(tmp_path, data)
    with pytest.raises(FiTunaError, match="general.architecture"):
        read_model_info(path, binaries=None)


def test_missing_block_count_raises(tmp_path):
    data = build_gguf(
        version=3,
        kvs=[("general.architecture", _T_STRING, _string_value(3, "testarch"))],
        tensors=[],
    )
    path = write_gguf(tmp_path, data)
    with pytest.raises(FiTunaError, match="testarch.block_count"):
        read_model_info(path, binaries=None)


def _info_with_file_type(file_type):
    return ModelInfo(
        architecture="testarch",
        n_layers=1,
        n_params=1,
        base_gguf_path=Path("dummy.gguf"),
        file_type=file_type,
    )


@pytest.mark.parametrize("file_type", [0, 1, 32, None])
def test_is_already_quantized_false_for_unquantized_types(file_type):
    assert is_already_quantized(_info_with_file_type(file_type)) is False


def test_is_already_quantized_true_for_quantized_type():
    # 15 = Q4_K_M in llama.cpp's LLAMA_FTYPE enum.
    assert is_already_quantized(_info_with_file_type(15)) is True


def test_model_fingerprint_is_deterministic_and_name_sensitive(tmp_path):
    data = build_gguf(version=3, kvs=valid_kvs(3), tensors=[])
    first = write_gguf(tmp_path, data, "first.gguf")
    assert model_fingerprint(first) == model_fingerprint(first)
    other = write_gguf(tmp_path, data, "other.gguf")
    assert model_fingerprint(other) != model_fingerprint(first)
