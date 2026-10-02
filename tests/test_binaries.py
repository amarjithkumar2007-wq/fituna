# SPDX-License-Identifier: MIT
"""Tests for the llama.cpp binary discovery and output parsers."""

from pathlib import Path

import pytest

from fituna.binaries import (
    _find_script,
    get_llama_cpp_version,
    list_supported_quant_types,
    locate_binaries,
)
from fituna.config import BinaryNotFoundError


def _fake_binary(path: Path, output: str) -> None:
    path.write_text(f"#!/bin/sh\nprintf '%s' '{output}'\n")
    path.chmod(0o755)


def test_quant_types_are_unique_and_preserve_help_order(tmp_path: Path) -> None:
    help_text = (
        "  7 or Q8_0 : first\n"
        "  6 or q4_k_m : second\n"
        "  5 or Q8_0 : duplicate\n"
    )
    for name in ("llama-bench", "llama-perplexity"):
        _fake_binary(tmp_path / name, "")
    _fake_binary(tmp_path / "llama-quantize", help_text)

    paths = locate_binaries(tmp_path)
    assert list_supported_quant_types(paths) == ["Q8_0", "Q4_K_M"]


def test_version_parser_accepts_rich_build_banner(tmp_path: Path) -> None:
    for name in ("llama-quantize", "llama-perplexity"):
        _fake_binary(tmp_path / name, "")
    _fake_binary(tmp_path / "llama-bench", "version: 9960 (a1b2c3d)\n")

    assert get_llama_cpp_version(locate_binaries(tmp_path)) == "9960 (a1b2c3d)"


def test_locate_binaries_reports_all_required_missing(tmp_path: Path) -> None:
    with pytest.raises(BinaryNotFoundError, match="llama-quantize.*llama-bench.*llama-perplexity"):
        locate_binaries(tmp_path)


def test_convert_script_is_found_at_standard_build_root(tmp_path: Path) -> None:
    repo = tmp_path / "llama.cpp"
    build_bin = repo / "build" / "bin"
    build_bin.mkdir(parents=True)
    script = repo / "convert_hf_to_gguf.py"
    script.write_text("# helper\n")

    assert _find_script("convert_hf_to_gguf.py", build_bin) == script
