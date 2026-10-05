# SPDX-License-Identifier: MIT
"""Offline regression coverage for quantized-output publication and caching."""

import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

from fituna.config import BinaryNotFoundError, BinaryPaths, FiTunaError
from fituna.quantize import _model_stem, quantize


@pytest.fixture
def inputs(tmp_path):
    base = tmp_path / "model-f16.gguf"
    base.write_bytes(b"synthetic model")
    paths = BinaryPaths(
        llama_quantize=tmp_path / "llama-quantize",
        llama_bench=Path("unused"), llama_perplexity=Path("unused"),
    )
    return base, tmp_path / "out", paths


@pytest.mark.parametrize("name,expected", [
    ("model-f16.gguf", "model"), ("model-F32.gguf", "model"),
    ("model.gguf", "model"), ("-f16.gguf", "-f16"),
])
def test_model_stem(name, expected):
    assert _model_stem(Path(name)) == expected


def test_success_publishes_complete_file_and_cache_skips_subprocess(monkeypatch, inputs):
    base, out, paths = inputs

    def run(cmd, **kwargs):
        assert cmd[0] == str(paths.llama_quantize)
        assert cmd[1] == str(base)
        assert cmd[3] == "Q4_K_M"
        assert kwargs["check"] is False
        assert kwargs["encoding"] == "utf-8"
        assert not (out / Path(cmd[2]).name.split(".tmp.")[0]).exists()
        Path(cmd[2]).write_bytes(b"completed output")
        return subprocess.CompletedProcess(cmd, 0, stderr="")

    runner = Mock(side_effect=run)
    monkeypatch.setattr("fituna.quantize.subprocess.run", runner)
    first = quantize(base, "Q4_K_M", out, paths, "a" * 64)
    assert first.read_bytes() == b"completed output"
    assert list(out.glob("*.tmp.*")) == []
    assert quantize(base, "Q4_K_M", out, paths, "a" * 64) == first
    assert runner.call_count == 1
    second = quantize(base, "Q4_K_M", out, paths, "b" * 64)
    assert second != first
    assert runner.call_count == 2


@pytest.mark.parametrize("returncode,content", [(1, b"partial"), (0, b""), (0, None)])
def test_failed_or_empty_output_never_becomes_a_cache_hit(monkeypatch, inputs, returncode, content):
    base, out, paths = inputs

    def run(cmd, **kwargs):
        if content is not None:
            Path(cmd[2]).write_bytes(content)
        return subprocess.CompletedProcess(cmd, returncode, stderr="synthetic diagnostic")

    monkeypatch.setattr("fituna.quantize.subprocess.run", run)
    with pytest.raises(FiTunaError, match="synthetic diagnostic"):
        quantize(base, "Q4_K_M", out, paths, "a" * 64)
    assert list(out.iterdir()) == []


@pytest.mark.parametrize("error,expected", [
    (FileNotFoundError("missing"), BinaryNotFoundError),
    (PermissionError("not executable"), FiTunaError),
])
def test_launch_errors_have_useful_types(monkeypatch, inputs, error, expected):
    monkeypatch.setattr("fituna.quantize.subprocess.run", Mock(side_effect=error))
    with pytest.raises(expected):
        quantize(inputs[0], "Q4_K_M", inputs[1], inputs[2], "a" * 64)


def test_zero_byte_target_is_rebuilt_and_only_matching_stale_files_are_removed(monkeypatch, inputs):
    base, out, paths = inputs
    out.mkdir()
    target = out / "model-aaaaaaaaaaaa-Q4_K_M.gguf"
    target.touch()
    stale = out / (target.name + ".tmp.old")
    stale.write_bytes(b"partial")
    unrelated = out / "other.gguf.tmp.old"
    unrelated.write_bytes(b"leave alone")

    def run(cmd, **kwargs):
        assert not stale.exists()
        Path(cmd[2]).write_bytes(b"rebuilt")
        return subprocess.CompletedProcess(cmd, 0, stderr="")

    monkeypatch.setattr("fituna.quantize.subprocess.run", run)
    assert quantize(base, "Q4_K_M", out, paths, "a" * 64).read_bytes() == b"rebuilt"
    assert unrelated.read_bytes() == b"leave alone"
