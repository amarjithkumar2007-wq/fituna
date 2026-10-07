# SPDX-License-Identifier: MIT
"""Unit tests for fituna.quality: perplexity and KL divergence evaluation."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from fituna.config import BinaryPaths, FiTunaError, QualityResult
from fituna.quality import (
    _parse_kld,
    _parse_kld_ppl,
    _parse_perplexity,
    compute_kld,
    compute_perplexity,
    evaluate_quality,
    generate_base_logits,
)


def _binaries(tmp_path: Path) -> BinaryPaths:
    return BinaryPaths(
        llama_quantize=tmp_path / "llama-quantize",
        llama_bench=tmp_path / "llama-bench",
        llama_perplexity=tmp_path / "llama-perplexity",
    )


def test_parse_perplexity_valid():
    sample = (
        "system_info: n_threads = 8\n"
        "[1]4.5987,[2]5.1234\n"
        "Final estimate: PPL = 5.9070 +/- 0.03170\n"
    )
    assert _parse_perplexity(sample) == 5.9070


def test_parse_perplexity_missing():
    assert _parse_perplexity("No perplexity output here") is None


KLD_FIXTURE = Path(__file__).parent / "fixtures" / "llama_perplexity_kld_sample.txt"


def test_parse_kld_valid():
    # Captured from llama.cpp b11342: llama-perplexity --kl-divergence on
    # SmolLM2-135M-Instruct Q4_K_M vs its f16 base, 4 chunks.
    sample = KLD_FIXTURE.read_text(encoding="utf-8")
    assert _parse_kld(sample) == 0.052965
    assert _parse_kld_ppl(sample) == 19.011455
    # The percentile lines ("Median  KLD", "99.9%   KLD", ...) must not match.
    assert _parse_kld("Median  KLD:   0.035235\nMaximum KLD:   3.862033\n") is None


def test_parse_kld_missing():
    assert _parse_kld("Final estimate: PPL = 5.9070") is None
    # A "Final estimate: KLD" line is not something llama.cpp prints.
    assert _parse_kld("Final estimate: KLD = 0.002410 +/- 0.000080") is None


def test_compute_perplexity_missing_files(tmp_path):
    bins = _binaries(tmp_path)
    gguf = tmp_path / "model.gguf"
    wiki = tmp_path / "wiki.txt"

    # Missing GGUF
    with pytest.raises(FiTunaError, match="GGUF file not found"):
        compute_perplexity(gguf, wiki, bins)

    # Missing corpus
    gguf.touch()
    with pytest.raises(FiTunaError, match="wikitext corpus not found"):
        compute_perplexity(gguf, wiki, bins)


def test_compute_perplexity_success(monkeypatch, tmp_path):
    bins = _binaries(tmp_path)
    gguf = tmp_path / "model.gguf"
    wiki = tmp_path / "wiki.txt"
    gguf.touch()
    wiki.touch()

    def fake_run(cmd, **kwargs):
        assert "--chunks" in cmd
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout="Final estimate: PPL = 6.2500 +/- 0.010\n",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    ppl = compute_perplexity(gguf, wiki, bins, chunks=16)
    assert ppl == 6.25


def test_compute_perplexity_failure(monkeypatch, tmp_path):
    bins = _binaries(tmp_path)
    gguf = tmp_path / "model.gguf"
    wiki = tmp_path / "wiki.txt"
    gguf.touch()
    wiki.touch()

    def fake_run_fail(cmd, **kwargs):
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=1,
            stdout="",
            stderr="fatal: out of memory\n",
        )

    monkeypatch.setattr(subprocess, "run", fake_run_fail)
    with pytest.raises(FiTunaError, match="exited with code 1"):
        compute_perplexity(gguf, wiki, bins)


def test_generate_base_logits_success(monkeypatch, tmp_path):
    bins = _binaries(tmp_path)
    base_gguf = tmp_path / "base.gguf"
    wiki = tmp_path / "wiki.txt"
    logits_path = tmp_path / "logits" / "base.kld"
    base_gguf.touch()
    wiki.touch()

    def fake_run(cmd, **kwargs):
        assert "--kl-divergence-base" in cmd
        # Written to a temp path beside logits_path, then moved into place.
        written = Path(cmd[cmd.index("--kl-divergence-base") + 1])
        assert written.parent == logits_path.parent and written != logits_path
        written.write_bytes(b"logits")
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="Final estimate: PPL = 6.0 +/- 0.1")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = generate_base_logits(base_gguf, wiki, logits_path, bins, chunks=32)
    assert result.logits_path == logits_path
    assert result.perplexity == 6.0
    assert logits_path.read_bytes() == b"logits"
    assert list(logits_path.parent.glob("base.kld.tmp.*")) == []


def test_generate_base_logits_missing_files(tmp_path):
    bins = _binaries(tmp_path)
    base_gguf = tmp_path / "base.gguf"
    wiki = tmp_path / "wiki.txt"
    logits_path = tmp_path / "base.kld"

    with pytest.raises(FiTunaError, match="Base GGUF file not found"):
        generate_base_logits(base_gguf, wiki, logits_path, bins)

    base_gguf.touch()
    with pytest.raises(FiTunaError, match="wikitext corpus not found"):
        generate_base_logits(base_gguf, wiki, logits_path, bins)


def test_generate_base_logits_unparseable_preserves_existing_file(monkeypatch, tmp_path):
    base, wiki, logits = tmp_path / "base.gguf", tmp_path / "wiki.txt", tmp_path / "base.kld"
    base.touch()
    wiki.touch()
    logits.write_bytes(b"old logits")

    def fake_run(cmd, **kwargs):
        Path(cmd[cmd.index("--kl-divergence-base") + 1]).write_bytes(b"new logits")
        return subprocess.CompletedProcess(cmd, 0, stdout="unrecognized output", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(FiTunaError, match="could not parse"):
        generate_base_logits(base, wiki, logits, _binaries(tmp_path))
    assert logits.read_bytes() == b"old logits"
    assert list(tmp_path.glob("base.kld.tmp.*")) == []

def test_compute_kld_success(monkeypatch, tmp_path):
    bins = _binaries(tmp_path)
    cand_gguf = tmp_path / "cand.gguf"
    wiki = tmp_path / "wiki.txt"
    base_logits = tmp_path / "base.kld"
    cand_gguf.touch()
    wiki.touch()
    base_logits.touch()

    def fake_run(cmd, **kwargs):
        assert "--kl-divergence-base" in cmd
        assert "--kl-divergence" in cmd
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout=(
                "Mean PPL(Q)                   :   6.120000 ±   0.020000\n"
                "====== KL divergence statistics ======\n"
                "Mean    KLD:   0.003150 ±   0.000040\n"
            ),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    kld, ppl = compute_kld(cand_gguf, wiki, base_logits, bins, chunks=32)
    assert kld == 0.00315
    assert ppl == 6.12


def test_compute_kld_missing_base_logits(tmp_path):
    bins = _binaries(tmp_path)
    cand_gguf = tmp_path / "cand.gguf"
    wiki = tmp_path / "wiki.txt"
    base_logits = tmp_path / "nonexistent.kld"
    cand_gguf.touch()
    wiki.touch()

    with pytest.raises(FiTunaError, match="base logits file not found"):
        compute_kld(cand_gguf, wiki, base_logits, bins)


def test_compute_kld_unparseable(monkeypatch, tmp_path):
    bins = _binaries(tmp_path)
    cand_gguf = tmp_path / "cand.gguf"
    wiki = tmp_path / "wiki.txt"
    base_logits = tmp_path / "base.kld"
    cand_gguf.touch()
    wiki.touch()
    base_logits.touch()

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout="Evaluation finished with no estimate\n",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(FiTunaError, match="could not parse 'Mean    KLD:"):
        compute_kld(cand_gguf, wiki, base_logits, bins)


def test_evaluate_quality_kld(monkeypatch, tmp_path):
    bins = _binaries(tmp_path)
    cand_gguf = tmp_path / "cand.gguf"
    wiki = tmp_path / "wiki.txt"
    base_logits = tmp_path / "base.kld"
    cand_gguf.touch()
    wiki.touch()
    base_logits.touch()

    monkeypatch.setattr(
        "fituna.quality.compute_kld",
        lambda *args, **kwargs: (0.0042, 6.30),
    )

    result = evaluate_quality(
        quant="Q4_K_M",
        quantized_gguf=cand_gguf,
        baseline_ppl=6.00,
        wikitext_path=wiki,
        binaries=bins,
        metric="kld",
        base_logits_path=base_logits,
    )

    assert isinstance(result, QualityResult)
    assert result.candidate_quant == "Q4_K_M"
    assert result.metric == "kld"
    assert result.kld == 0.0042
    assert result.perplexity == 6.30
    assert abs(result.quality_loss_pct - 5.0) < 1e-3


def test_evaluate_quality_kld_requires_logits(tmp_path):
    bins = _binaries(tmp_path)
    cand_gguf = tmp_path / "cand.gguf"
    wiki = tmp_path / "wiki.txt"
    cand_gguf.touch()
    wiki.touch()

    with pytest.raises(FiTunaError, match="base_logits_path is required"):
        evaluate_quality(
            quant="Q4_K_M",
            quantized_gguf=cand_gguf,
            baseline_ppl=6.00,
            wikitext_path=wiki,
            binaries=bins,
            metric="kld",
            base_logits_path=None,
        )


def test_compute_kld_scientific_notation(monkeypatch, tmp_path):
    bins = _binaries(tmp_path)
    cand_gguf = tmp_path / "cand.gguf"
    wiki = tmp_path / "wiki.txt"
    base_logits = tmp_path / "base.kld"
    cand_gguf.touch()
    wiki.touch()
    base_logits.touch()

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout=(
                "Mean PPL(Q)                   :   6.120000 ±   0.020000\n"
                "====== KL divergence statistics ======\n"
                "Mean    KLD:   1.25e-04 ±   0.000010\n"
            ),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    kld, ppl = compute_kld(cand_gguf, wiki, base_logits, bins)
    assert kld == 0.000125
    assert ppl == 6.12



def test_compute_kld_without_quantized_ppl_is_an_error(monkeypatch, tmp_path):
    """quality_loss_pct comes from Mean PPL(Q). If it is missing, falling
    back to the baseline would report 0% loss and pass any quality gate."""
    bins = _binaries(tmp_path)
    cand_gguf, wiki, base_logits = tmp_path / "cand.gguf", tmp_path / "wiki.txt", tmp_path / "base.kld"
    for p in (cand_gguf, wiki, base_logits):
        p.touch()

    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(
        args=cmd, returncode=0, stdout="Mean    KLD:   0.003150 ±   0.000040\n", stderr=""))
    with pytest.raises(FiTunaError, match=r"Mean PPL\(Q\)"):
        compute_kld(cand_gguf, wiki, base_logits, bins)


def test_generate_base_logits_failure_leaves_no_logits_file(monkeypatch, tmp_path):
    """search() reuses the logits file whenever it exists, so an interrupted
    or failed run must never leave a partial file at the final path."""
    bins = _binaries(tmp_path)
    base_gguf, wiki = tmp_path / "base.gguf", tmp_path / "wiki.txt"
    base_gguf.touch()
    wiki.touch()
    logits_path = tmp_path / "base.kld"

    def fake_run(cmd, **kwargs):
        Path(cmd[cmd.index("--kl-divergence-base") + 1]).write_bytes(b"partial")
        return subprocess.CompletedProcess(args=cmd, returncode=1, stdout="", stderr="killed")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(FiTunaError, match="exited with code 1"):
        generate_base_logits(base_gguf, wiki, logits_path, bins)
    assert list(tmp_path.glob("base.kld*")) == []
