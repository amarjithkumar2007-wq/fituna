# SPDX-License-Identifier: MIT
"""Pytest coverage for llama-bench JSON parsing and command construction."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from fituna.bench import _parse_bench_json, run_bench
from fituna.config import BinaryPaths, BenchTimeoutError, CandidateConfig, FiTunaError, TargetSpec

BENCH_FIXTURE = Path(__file__).parent / "fixtures" / "llama_bench_sample.json"


def _target() -> TargetSpec:
    return TargetSpec(
        model_path=Path("model.gguf"),
        target_tokens_per_sec=20.0,
        max_quality_loss_pct=5.0,
        prompt_tokens=512,
        gen_tokens=128,
    )


def _binaries(tmp_path: Path) -> BinaryPaths:
    return BinaryPaths(
        llama_quantize=tmp_path / "llama-quantize",
        llama_bench=tmp_path / "llama-bench",
        llama_perplexity=tmp_path / "llama-perplexity",
    )


def _successful_process(stdout: str) -> SimpleNamespace:
    return SimpleNamespace(returncode=0, stdout=stdout, stderr="")


def test_parse_captured_llama_bench_fixture() -> None:
    stdout = BENCH_FIXTURE.read_text()
    result = _parse_bench_json(
        stdout,
        Path("llama-3-8b-instruct-Q4_K_M.gguf"),
        ngl=32,
        ctx=4096,
    )

    assert result.candidate == CandidateConfig(quant="Q4_K_M", ngl=32, ctx=4096)
    assert result.prompt_tok_per_sec == 120.5
    assert result.gen_tok_per_sec == 22.8
    assert result.vram_used_mb is None
    assert result.raw_stdout == stdout


def test_run_bench_uses_n_depth_and_passes_timeout(monkeypatch, tmp_path: Path) -> None:
    stdout = BENCH_FIXTURE.read_text()
    run = Mock(return_value=_successful_process(stdout))
    monkeypatch.setattr("fituna.bench.subprocess.run", run)
    binaries = _binaries(tmp_path)

    result = run_bench(
        Path("model-Q4_K_M.gguf"),
        ngl=16,
        ctx=1024,
        target=_target(),
        binaries=binaries,
        timeout_sec=17,
    )

    run.assert_called_once_with(
        [
            str(binaries.llama_bench),
            "-m",
            "model-Q4_K_M.gguf",
            "-ngl",
            "16",
            "-d",
            "384",
            "-p",
            "512",
            "-n",
            "128",
            "-o",
            "json",
        ],
        capture_output=True,
        text=True,
        timeout=17,
        encoding="utf-8",
        errors="replace",
    )
    assert result.candidate.ctx == 1024


def test_parse_output_with_newer_minimal_record_shape() -> None:
    stdout = json.dumps(
        [
            {"test": "pp512", "n_prompt": 512, "avg_ts": 118.0, "future_field": "ignored"},
            {"test": "tg128", "n_gen": 128, "avg_ts": 21.5, "future_field": "ignored"},
        ]
    )

    result = _parse_bench_json(stdout, Path("model-Q5_K_M.gguf"), ngl=0, ctx=2048)

    assert result.candidate.quant == "Q5_K_M"
    assert result.prompt_tok_per_sec == 118.0
    assert result.gen_tok_per_sec == 21.5


def test_missing_optional_metrics_default_to_zero() -> None:
    stdout = json.dumps([{"n_prompt": 512}, {"n_gen": 128}])

    result = _parse_bench_json(stdout, Path("model.gguf"), ngl=0, ctx=4096)

    assert result.prompt_tok_per_sec == 0.0
    assert result.gen_tok_per_sec == 0.0
    assert result.vram_used_mb is None


@pytest.mark.parametrize("stdout", ["not json", '[{"n_prompt": 512'])
def test_run_bench_rejects_malformed_or_truncated_json(
    monkeypatch,
    tmp_path: Path,
    stdout: str,
) -> None:
    monkeypatch.setattr(
        "fituna.bench.subprocess.run",
        Mock(return_value=_successful_process(stdout)),
    )

    with pytest.raises(FiTunaError, match="could not parse llama-bench JSON output"):
        run_bench(
            Path("model-Q4_K_M.gguf"),
            ngl=0,
            ctx=4096,
            target=_target(),
            binaries=_binaries(tmp_path),
        )


def test_run_bench_rejects_empty_records(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "fituna.bench.subprocess.run",
        Mock(return_value=_successful_process("[]")),
    )

    with pytest.raises(FiTunaError, match="llama-bench produced no test records"):
        run_bench(
            Path("model-Q4_K_M.gguf"),
            ngl=0,
            ctx=4096,
            target=_target(),
            binaries=_binaries(tmp_path),
        )


def test_run_bench_translates_timeout(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "fituna.bench.subprocess.run",
        Mock(side_effect=subprocess.TimeoutExpired(cmd="llama-bench", timeout=3)),
    )

    with pytest.raises(BenchTimeoutError, match="timed out after 300s"):
        run_bench(
            Path("model-Q4_K_M.gguf"),
            ngl=0,
            ctx=4096,
            target=_target(),
            binaries=_binaries(tmp_path),
        )
