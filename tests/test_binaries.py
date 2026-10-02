# SPDX-License-Identifier: MIT
"""Tests for the llama.cpp binary discovery and output parsers."""

from pathlib import Path
import subprocess
from unittest.mock import Mock

import pytest

from fituna.binaries import (
    _find_script,
    get_llama_cpp_version,
    list_supported_quant_types,
    locate_binaries,
)
from fituna.config import BinaryNotFoundError, BinaryPaths


def _paths(tmp_path: Path) -> BinaryPaths:
    return BinaryPaths(
        llama_quantize=tmp_path / "llama-quantize",
        llama_bench=tmp_path / "llama-bench",
        llama_perplexity=tmp_path / "llama-perplexity",
    )


def test_quant_types_are_unique_and_preserve_help_order(monkeypatch, tmp_path: Path) -> None:
    help_text = (
        "  7 or Q8_0 : first\n"
        "  6 or q4_k_m : second\n"
        "  5 or Q8_0 : duplicate\n"
    )
    paths = _paths(tmp_path)
    run = Mock(return_value=subprocess.CompletedProcess([], 1, stdout="", stderr=help_text))
    monkeypatch.setattr("fituna.binaries.subprocess.run", run)
    assert list_supported_quant_types(paths) == ["Q8_0", "Q4_K_M"]
    run.assert_called_once_with(
        [str(paths.llama_quantize), "--help"], capture_output=True, text=True,
        timeout=30, encoding="utf-8", errors="replace",
    )


@pytest.mark.parametrize("output, expected", [
    ("version: 9960 (a1b2c3d)\n", "9960 (a1b2c3d)"),
    ("main: build = 3765 (c919d5d)\n", "3765 (c919d5d)"),
    ("llama.cpp 9960\n", "9960"),
    ("version: local-build\n", "local-build"),
])
def test_version_parser_accepts_build_banners(monkeypatch, tmp_path: Path, output, expected) -> None:
    monkeypatch.setattr(
        "fituna.binaries.subprocess.run",
        Mock(return_value=subprocess.CompletedProcess([], 0, stdout=output, stderr="")),
    )
    assert get_llama_cpp_version(_paths(tmp_path)) == expected


@pytest.mark.parametrize("output", ["", "usage: llama-quantize", "7 or Q8_0", "or Q4_K_M : missing id"])
def test_malformed_quant_help_has_no_types(monkeypatch, tmp_path: Path, output) -> None:
    monkeypatch.setattr(
        "fituna.binaries.subprocess.run",
        Mock(return_value=subprocess.CompletedProcess([], 0, stdout=output, stderr="")),
    )
    assert list_supported_quant_types(_paths(tmp_path)) == []


def test_version_falls_back_to_perplexity(monkeypatch, tmp_path: Path) -> None:
    run = Mock(side_effect=[
        subprocess.CompletedProcess([], 1, stdout="", stderr="error: invalid parameter: --version"),
        subprocess.CompletedProcess([], 0, stdout="usage: llama-bench", stderr=""),
        subprocess.CompletedProcess([], 0, stdout="", stderr="version: 9960 (a935fbffe)"),
    ])
    monkeypatch.setattr("fituna.binaries.subprocess.run", run)
    paths = _paths(tmp_path)
    assert get_llama_cpp_version(paths) == "9960 (a935fbffe)"
    assert [call.args[0] for call in run.call_args_list] == [
        [str(paths.llama_bench), "--version"], [str(paths.llama_bench), "--help"],
        [str(paths.llama_perplexity), "--version"],
    ]


@pytest.mark.parametrize("error", [FileNotFoundError("missing"), subprocess.TimeoutExpired("llama-bench", 30)])
def test_version_returns_none_when_probes_cannot_run(monkeypatch, tmp_path: Path, error) -> None:
    run = Mock(side_effect=error)
    monkeypatch.setattr("fituna.binaries.subprocess.run", run)
    assert get_llama_cpp_version(_paths(tmp_path)) is None
    assert run.call_count == 4


def test_locate_binaries_reports_all_required_missing(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("fituna.binaries.shutil.which", lambda name, path=None: None)
    with pytest.raises(BinaryNotFoundError, match="llama-quantize.*llama-bench.*llama-perplexity"):
        locate_binaries(tmp_path)


def test_convert_script_is_found_at_standard_build_root(tmp_path: Path) -> None:
    repo = tmp_path / "llama.cpp"
    build_bin = repo / "build" / "bin"
    build_bin.mkdir(parents=True)
    script = repo / "convert_hf_to_gguf.py"
    script.write_text("# helper\n")

    assert _find_script("convert_hf_to_gguf.py", build_bin) == script
