# SPDX-License-Identifier: MIT
# Copyright (c) 2026 FiTuna contributors
"""fituna.binaries
==================

Locates required llama.cpp executables (on PATH or under a user-supplied
``--llama-bin-dir``), and introspects them (``--help``/``--version``) so
FiTuna never hardcodes a quant-type list that may drift from the user's
llama.cpp build.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from fituna.config import BinaryNotFoundError, BinaryPaths

REQUIRED_BINARIES = ("llama-quantize", "llama-bench", "llama-perplexity")
"""The three llama.cpp binaries FiTuna cannot run without.

Public (not ``_``-prefixed): ``fituna.doctor`` imports this directly as its
own single source of truth for which binaries are required, the same
reasoning as ``find_exe`` being public (see its docstring).
"""
_OPTIONAL_BIN = "llama-imatrix"
_CONVERT_SCRIPT = "convert_hf_to_gguf.py"

# Matches lines from `llama-quantize --help`, e.g.:
#   "   7  or  Q8_0    :  6.70G, +0.0004 ppl @ 7B"
#   "   0  or  COPY    :  only copy tensors, no quantizing"
_QUANT_LINE_RE = re.compile(r"^\s*\d+\s+or\s+([A-Za-z0-9_]+)\s*:", re.MULTILINE)

# llama.cpp's build banner has drifted in wording across versions
# ("main: build = 3765 (c919d5d)", "version: 3765 (c919d5d)", and current
# builds print a bare "llama.cpp 9960"); try the richer "number (hash)" form
# first, then the bare-banner form, then a looser fallback.
_VERSION_RE_RICH = re.compile(r"(?:build|version)\s*[:=]\s*(\d+\s*\([0-9a-fA-F]+\))")
_VERSION_RE_BARE = re.compile(r"^llama\.cpp\s+(\d+)\s*$", re.MULTILINE)
_VERSION_RE_LOOSE = re.compile(r"(?:build|version)\s*[:=]\s*(\S+)")


def find_exe(name: str, bin_dir: Optional[Path]) -> Optional[Path]:
    """Resolve an executable by name: restricted to bin_dir if given, else PATH.

    Public (not ``_``-prefixed): ``fituna.doctor`` calls this directly for
    its per-binary checks -- a supported single-binary lookup, not just an
    internal helper of ``locate_binaries()``.
    """
    found = shutil.which(name, path=str(bin_dir) if bin_dir is not None else None)
    return Path(found) if found else None


def _find_script(name: str, bin_dir: Optional[Path]) -> Optional[Path]:
    """Resolve a helper script by exact filename (may lack the exec bit).

    ``--llama-bin-dir`` is documented (README) as the compiled binaries'
    directory, e.g. ``/path/to/llama.cpp/build/bin`` -- but
    convert_hf_to_gguf.py lives at the llama.cpp repo *root*, two levels up
    from a standard CMake build's ``build/bin``, not inside it. Checking only
    ``bin_dir / name`` meant the exact --llama-bin-dir value the README's own
    example uses could never find the convert script. Check both the given
    dir and that conventional repo-root location before giving up.
    """
    if bin_dir is not None:
        for candidate in (bin_dir / name, bin_dir.parent.parent / name):
            if candidate.is_file():
                return candidate
        return None
    found = shutil.which(name)  # opportunistic; convert scripts are rarely on PATH
    return Path(found) if found else None


def locate_binaries(bin_dir: Optional[Path] = None) -> BinaryPaths:
    """Find llama-quantize / llama-bench / llama-perplexity (required) and
    llama-imatrix / convert_hf_to_gguf.py (optional) in bin_dir if given,
    else on PATH (shutil.which).

    Raises BinaryNotFoundError with an install-guide message if any of the
    three required binaries is missing.
    """
    resolved = {name: find_exe(name, bin_dir) for name in REQUIRED_BINARIES}
    missing = [name for name, path in resolved.items() if path is None]
    if missing:
        where = f"under --llama-bin-dir {bin_dir}" if bin_dir is not None else "on PATH"
        raise BinaryNotFoundError(
            f"Required llama.cpp binaries not found {where}: {', '.join(missing)}.\n"
            "Build llama.cpp (https://github.com/ggml-org/llama.cpp#building-the-project) "
            "and either add its build output directory (e.g. build/bin) to your PATH, "
            "or pass --llama-bin-dir /path/to/llama.cpp/build/bin."
        )

    return BinaryPaths(
        llama_quantize=resolved["llama-quantize"],
        llama_bench=resolved["llama-bench"],
        llama_perplexity=resolved["llama-perplexity"],
        llama_imatrix=find_exe(_OPTIONAL_BIN, bin_dir),
        convert_script=_find_script(_CONVERT_SCRIPT, bin_dir),
    )


def _run_text(binary: Path, arg: str) -> str:
    """Run `binary arg` and return stdout+stderr concatenated (best-effort;
    llama.cpp tools print --help/--version to either stream depending on
    version, and often exit non-zero for --help, so returncode is ignored)."""
    try:
        # encoding/errors explicit: see hardware.py's _run for why (Windows
        # non-UTF-8 locale can otherwise raise UnicodeDecodeError).
        proc = subprocess.run(
            [str(binary), arg],
            capture_output=True,
            text=True,
            timeout=30,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise BinaryNotFoundError(f"Failed to run '{binary} {arg}': {exc}") from exc
    return (proc.stdout or "") + "\n" + (proc.stderr or "")


def list_supported_quant_types(paths: BinaryPaths) -> list[str]:
    """Run `llama-quantize --help`, parse the listed quant type tokens
    (e.g. Q4_K_M, Q8_0, ...) out of its usage text.
    """
    output = _run_text(paths.llama_quantize, "--help")
    seen: set[str] = set()
    result: list[str] = []
    for token in _QUANT_LINE_RE.findall(output):
        name = token.upper()
        if name not in seen:
            seen.add(name)
            result.append(name)
    return result


def get_llama_cpp_version(paths: BinaryPaths) -> Optional[str]:
    """Best-effort version string from the llama.cpp binaries.
    Returns None if none of them expose one.

    Probes llama-perplexity as well as llama-bench: current builds'
    llama-bench rejects --version outright ("error: invalid parameter") and
    prints no build banner in --help, while llama-perplexity supports
    --version ("version: 9960 (a935fbffe)") -- verified against Homebrew
    build 9960.
    """
    for binary in (paths.llama_bench, paths.llama_perplexity):
        for arg in ("--version", "--help"):
            try:
                text = _run_text(binary, arg)
            except BinaryNotFoundError:
                continue
            match = (
                _VERSION_RE_RICH.search(text)
                or _VERSION_RE_BARE.search(text)
                or _VERSION_RE_LOOSE.search(text)
            )
            if match:
                return match.group(1).strip()
    return None


def _selfcheck() -> None:
    """Minimal runnable check: fakes a llama.cpp bin/ dir with shell-script
    stand-ins and exercises locate_binaries / help-parsing / version-parsing
    without requiring a real llama.cpp build."""
    import stat
    import tempfile

    def _write_fake(directory: Path, name: str, help_text: str, version_text: str = "") -> None:
        script = directory / name
        script.write_text(
            "#!/bin/sh\n"
            'if [ "$1" = "--version" ]; then\n'
            f'  printf %s "{version_text}" 1>&2\n'
            "else\n"
            f"  cat <<'EOF'\n{help_text}EOF\n"
            "fi\n"
        )
        script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        help_text = (
            "usage: llama-quantize ...\n"
            "Allowed quantization types:\n"
            "   7  or  Q8_0    :  6.70G, +0.0004 ppl @ 7B\n"
            "  17  or  Q6_K    :  5.15G, +0.0044 ppl @ 7B\n"
            "  15  or  Q5_K_M  :  4.45G, +0.0136 ppl @ 7B\n"
            "  15  or  Q4_K_M  :  3.80G, +0.0532 ppl @ 7B\n"
            "  12  or  Q3_K_M  :  3.07G, +0.2496 ppl @ 7B\n"
            "  10  or  Q2_K    :  2.63G, +0.6717 ppl @ 7B\n"
            "   0  or  COPY    :  only copy tensors, no quantizing\n"
        )
        _write_fake(tmp_path, "llama-quantize", help_text)
        _write_fake(
            tmp_path,
            "llama-bench",
            "usage: llama-bench ...\n",
            version_text="main: build = 3765 (c919d5d)\n",
        )
        _write_fake(tmp_path, "llama-perplexity", "usage: llama-perplexity ...\n")

        # Missing binaries -> BinaryNotFoundError naming what's absent.
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        try:
            locate_binaries(empty_dir)
        except BinaryNotFoundError as exc:
            assert "llama-quantize" in str(exc), str(exc)
        else:
            raise AssertionError("expected BinaryNotFoundError for empty bin_dir")

        # All required present -> resolved paths; optional ones absent -> None.
        paths = locate_binaries(tmp_path)
        assert paths.llama_quantize == tmp_path / "llama-quantize"
        assert paths.llama_bench == tmp_path / "llama-bench"
        assert paths.llama_perplexity == tmp_path / "llama-perplexity"
        assert paths.llama_imatrix is None
        assert paths.convert_script is None

        quants = list_supported_quant_types(paths)
        assert quants == [
            "Q8_0",
            "Q6_K",
            "Q5_K_M",
            "Q4_K_M",
            "Q3_K_M",
            "Q2_K",
            "COPY",
        ], quants

        version = get_llama_cpp_version(paths)
        assert version == "3765 (c919d5d)", version

        # convert_hf_to_gguf.py resolution: the README's own
        # --llama-bin-dir example points at a standard CMake build's
        # build/bin, where only compiled binaries live -- the convert
        # script sits two levels up, at the repo root. Reproduce that
        # exact layout and confirm _find_script's repo-root fallback
        # finds it (regression test for the bug where only bin_dir/name
        # was checked and this exact documented layout never resolved).
        repo_root = tmp_path / "llama.cpp"
        build_bin = repo_root / "build" / "bin"
        build_bin.mkdir(parents=True)
        _write_fake(build_bin, "llama-quantize", help_text)
        _write_fake(build_bin, "llama-bench", "usage: llama-bench ...\n")
        _write_fake(build_bin, "llama-perplexity", "usage: llama-perplexity ...\n")
        (repo_root / "convert_hf_to_gguf.py").write_text("# fake convert script\n")

        paths_repo = locate_binaries(build_bin)
        assert paths_repo.convert_script == repo_root / "convert_hf_to_gguf.py", (
            paths_repo.convert_script
        )

    print("fituna.binaries self-check OK")


if __name__ == "__main__":
    _selfcheck()
