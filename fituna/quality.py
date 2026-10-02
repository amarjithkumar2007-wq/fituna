# SPDX-License-Identifier: MIT
"""fituna.quality
=================

Runs ``llama-perplexity`` against a wikitext corpus to measure quality loss
of a quantized GGUF relative to the unquantized baseline.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Optional

from fituna.config import BinaryPaths, FiTunaError, QualityResult

# llama-perplexity has no built-in timeout of its own and a full
# wikitext-2 pass on a large model can run long; 30 min is a generous default
# ceiling. Bump via PPL_TIMEOUT_SEC module attribute if a caller needs more.
PPL_TIMEOUT_SEC = 1800

_FLOAT_PATTERN = r"[+\-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+\-]?\d+)?"
_PPL_RE = re.compile(rf"Final estimate:\s*PPL\s*=\s*({_FLOAT_PATTERN})", re.IGNORECASE)
# `llama-perplexity --kl-divergence` does not print a "Final estimate" line;
# it prints a statistics block (tools/perplexity/perplexity.cpp):
#     ====== Perplexity statistics ======
#     Mean PPL(Q)                   :  19.011455 ±   1.803818
#     ...
#     ====== KL divergence statistics ======
#     Mean    KLD:   0.052965 ±   0.004263
# See tests/fixtures/llama_perplexity_kld_sample.txt for a captured run.
_KLD_RE = re.compile(rf"^\s*Mean\s+KLD\s*:\s*({_FLOAT_PATTERN})", re.MULTILINE)
_KLD_PPL_RE = re.compile(rf"^\s*Mean\s+PPL\(Q\)\s*:\s*({_FLOAT_PATTERN})", re.MULTILINE)


def _parse_perplexity(text: str) -> Optional[float]:
    """Extract the final PPL value from llama-perplexity's combined
    stdout/stderr, e.g. a line like:
        Final estimate: PPL = 5.9070 +/- 0.03170
    Returns None if no such line is present.
    """
    match = _PPL_RE.search(text)
    if match is None:
        return None
    return float(match.group(1))


def _parse_kld(text: str) -> Optional[float]:
    """Extract the mean KLD from `llama-perplexity --kl-divergence` output,
    i.e. the line in the "KL divergence statistics" block:
        Mean    KLD:   0.052965 ±   0.004263
    Returns None if no such line is present.
    """
    match = _KLD_RE.search(text)
    if match is None:
        return None
    return float(match.group(1))


def _parse_kld_ppl(text: str) -> Optional[float]:
    """Extract the quantized model's perplexity from a --kl-divergence run,
    which reports it as `Mean PPL(Q) : <value> ± <err>` instead of the
    "Final estimate: PPL" line a plain perplexity run prints.
    """
    match = _KLD_PPL_RE.search(text)
    if match is None:
        return None
    return float(match.group(1))


def compute_perplexity(
    gguf_path: Path,
    wikitext_path: Path,
    binaries: BinaryPaths,
    chunks: Optional[int] = None,
) -> float:
    """Run `llama-perplexity -m <gguf_path> -f <wikitext_path>` (optionally
    limited to `chunks` chunks to save time) and parse the final PPL value
    from stdout.
    """
    if not gguf_path.exists():
        raise FiTunaError(f"GGUF file not found: {gguf_path}")
    if not wikitext_path.exists():
        raise FiTunaError(
            f"wikitext corpus not found: {wikitext_path} "
            "(see README for the wikitext-2 download link)"
        )

    cmd = [
        str(binaries.llama_perplexity),
        "-m", str(gguf_path),
        "-f", str(wikitext_path),
    ]
    if chunks is not None:
        cmd += ["--chunks", str(chunks)]

    try:
        # encoding/errors explicit: see hardware.py's _run for why.
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=PPL_TIMEOUT_SEC,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError as exc:
        raise FiTunaError(
            f"llama-perplexity binary not found: {binaries.llama_perplexity}"
        ) from exc
    except OSError as exc:
        # e.g. PermissionError -- binary exists but isn't executable.
        raise FiTunaError(
            f"failed to launch llama-perplexity ({binaries.llama_perplexity}): {exc}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise FiTunaError(
            f"llama-perplexity timed out after {PPL_TIMEOUT_SEC}s on {gguf_path.name}"
        ) from exc

    output = proc.stdout + "\n" + proc.stderr
    if proc.returncode != 0:
        tail = output.strip()[-2000:]
        raise FiTunaError(
            f"llama-perplexity exited with code {proc.returncode} on {gguf_path.name}:\n{tail}"
        )

    ppl = _parse_perplexity(output)
    if ppl is None:
        tail = output.strip()[-2000:]
        raise FiTunaError(
            f"could not parse 'Final estimate: PPL = ...' from llama-perplexity output "
            f"for {gguf_path.name}:\n{tail}"
        )
    return ppl


def generate_base_logits(
    base_gguf_path: Path,
    wikitext_path: Path,
    logits_path: Path,
    binaries: BinaryPaths,
    chunks: Optional[int] = None,
) -> Path:
    """Run `llama-perplexity -m <base_gguf_path> -f <wikitext_path> --kl-divergence-base <logits_path>`
    to record reference logits from the unquantized baseline model.
    """
    if not base_gguf_path.exists():
        raise FiTunaError(f"Base GGUF file not found: {base_gguf_path}")
    if not wikitext_path.exists():
        raise FiTunaError(
            f"wikitext corpus not found: {wikitext_path} "
            "(see README for the wikitext-2 download link)"
        )

    logits_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(binaries.llama_perplexity),
        "-m", str(base_gguf_path),
        "-f", str(wikitext_path),
        "--kl-divergence-base", str(logits_path),
    ]
    if chunks is not None:
        cmd += ["--chunks", str(chunks)]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=PPL_TIMEOUT_SEC,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError as exc:
        raise FiTunaError(
            f"llama-perplexity binary not found: {binaries.llama_perplexity}"
        ) from exc
    except OSError as exc:
        raise FiTunaError(
            f"failed to launch llama-perplexity ({binaries.llama_perplexity}): {exc}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise FiTunaError(
            f"llama-perplexity timed out after {PPL_TIMEOUT_SEC}s generating logits for {base_gguf_path.name}"
        ) from exc

    output = proc.stdout + "\n" + proc.stderr
    if proc.returncode != 0:
        tail = output.strip()[-2000:]
        raise FiTunaError(
            f"llama-perplexity exited with code {proc.returncode} generating logits for {base_gguf_path.name}:\n{tail}"
        )
    return logits_path


def compute_kld(
    quantized_gguf: Path,
    wikitext_path: Path,
    base_logits_path: Path,
    binaries: BinaryPaths,
    chunks: Optional[int] = None,
) -> tuple[float, Optional[float]]:
    """Run `llama-perplexity -m <quantized_gguf> -f <wikitext_path> --kl-divergence-base <base_logits_path> --kl-divergence`
    and parse the final KLD value (and PPL if present) from stdout/stderr.
    Returns (kld, ppl).
    """
    if not quantized_gguf.exists():
        raise FiTunaError(f"Quantized GGUF file not found: {quantized_gguf}")
    if not wikitext_path.exists():
        raise FiTunaError(f"wikitext corpus not found: {wikitext_path}")
    if not base_logits_path.exists():
        raise FiTunaError(f"base logits file not found: {base_logits_path}")

    cmd = [
        str(binaries.llama_perplexity),
        "-m", str(quantized_gguf),
        "-f", str(wikitext_path),
        "--kl-divergence-base", str(base_logits_path),
        "--kl-divergence",
    ]
    if chunks is not None:
        cmd += ["--chunks", str(chunks)]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=PPL_TIMEOUT_SEC,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError as exc:
        raise FiTunaError(
            f"llama-perplexity binary not found: {binaries.llama_perplexity}"
        ) from exc
    except OSError as exc:
        raise FiTunaError(
            f"failed to launch llama-perplexity ({binaries.llama_perplexity}): {exc}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise FiTunaError(
            f"llama-perplexity timed out after {PPL_TIMEOUT_SEC}s computing KLD for {quantized_gguf.name}"
        ) from exc

    output = proc.stdout + "\n" + proc.stderr
    if proc.returncode != 0:
        tail = output.strip()[-2000:]
        raise FiTunaError(
            f"llama-perplexity exited with code {proc.returncode} computing KLD for {quantized_gguf.name}:\n{tail}"
        )

    kld = _parse_kld(output)
    if kld is None:
        tail = output.strip()[-2000:]
        raise FiTunaError(
            f"could not parse 'Mean    KLD: ...' from llama-perplexity output "
            f"for {quantized_gguf.name}:\n{tail}"
        )
    ppl = _parse_kld_ppl(output)
    return kld, ppl


def evaluate_quality(
    quant: str,
    quantized_gguf: Path,
    baseline_ppl: float,
    wikitext_path: Path,
    binaries: BinaryPaths,
    chunks: Optional[int] = None,
    metric: str = "ppl",
    base_logits_path: Optional[Path] = None,
) -> QualityResult:
    """Evaluate quality of quantized_gguf.
    If metric == 'kld', runs compute_kld against base_logits_path and returns
    a QualityResult with kld and quality_loss_pct derived from candidate and baseline ppl.
    If metric == 'ppl' (default), runs compute_perplexity and derives quality_loss_pct.
    """
    if metric == "kld":
        if base_logits_path is None:
            raise FiTunaError("base_logits_path is required when metric is 'kld'")
        kld, cand_ppl = compute_kld(
            quantized_gguf, wikitext_path, base_logits_path, binaries, chunks
        )
        ppl = cand_ppl if cand_ppl is not None else baseline_ppl
        loss_pct = (
            (ppl - baseline_ppl) / baseline_ppl * 100.0 if baseline_ppl > 0 else 0.0
        )
        return QualityResult(
            candidate_quant=quant,
            perplexity=ppl,
            baseline_perplexity=baseline_ppl,
            quality_loss_pct=loss_pct,
            metric="kld",
            kld=kld,
        )

    ppl = compute_perplexity(quantized_gguf, wikitext_path, binaries, chunks)
    loss_pct = (ppl - baseline_ppl) / baseline_ppl * 100.0 if baseline_ppl > 0 else 0.0
    return QualityResult(
        candidate_quant=quant,
        perplexity=ppl,
        baseline_perplexity=baseline_ppl,
        quality_loss_pct=loss_pct,
        metric="ppl",
        kld=None,
    )


def _self_check() -> None:
    """Assert-based sanity check, no real llama.cpp binary required.

    Covers: (1) PPL and KLD regex parsing against realistic output; (2)
    quality_loss_pct arithmetic; (3) that a missing binary/corpus/logits
    is surfaced as FiTunaError rather than a raw OSError.
    """
    # 1. Parsing PPL and KLD.
    sample = (
        "system_info: n_threads = 8\n"
        "perplexity: calculating perplexity over 655 chunks\n"
        "[1]4.5987,[2]5.1234,...\n"
        "Final estimate: PPL = 5.9070 +/- 0.03170\n"
    )
    assert _parse_perplexity(sample) == 5.9070
    assert _parse_perplexity("no ppl line here") is None

    kld_sample = (
        "====== Perplexity statistics ======\n"
        "Mean PPL(Q)                   :  19.011455 ±   1.803818\n"
        "Mean PPL(base)                :  18.278568 ±   1.763930\n"
        "====== KL divergence statistics ======\n"
        "Mean    KLD:   0.052965 ±   0.004263\n"
        "Maximum KLD:   3.862033\n"
        "Median  KLD:   0.035235\n"
    )
    assert _parse_kld(kld_sample) == 0.052965
    assert _parse_kld_ppl(kld_sample) == 19.011455
    assert _parse_kld("no kld line here") is None
    assert _parse_kld("Final estimate: KLD = 0.0024") is None
    assert _parse_kld("Mean    KLD:   1.25e-04 ±   0.000010") == 0.000125
    assert _parse_perplexity("Final estimate: PPL = 6.12e+01 +/- 0.02") == 61.2

    # 2. quality_loss_pct arithmetic, mirrored from evaluate_quality's formula.
    baseline = 5.80
    ppl = 5.9070
    expected_loss = (ppl - baseline) / baseline * 100.0
    result = QualityResult(
        candidate_quant="Q4_K_M",
        perplexity=ppl,
        baseline_perplexity=baseline,
        quality_loss_pct=expected_loss,
        metric="kld",
        kld=0.00241,
    )
    assert abs(result.quality_loss_pct - 1.8448) < 1e-3
    assert result.kld == 0.00241

    # 3. Missing wikitext corpus -> FiTunaError, not a bare exception.
    fake_binaries = BinaryPaths(
        llama_quantize=Path("/nonexistent/llama-quantize"),
        llama_bench=Path("/nonexistent/llama-bench"),
        llama_perplexity=Path("/nonexistent/llama-perplexity"),
    )
    try:
        compute_perplexity(
            gguf_path=Path(__file__),  # exists, just not a real gguf
            wikitext_path=Path("/nonexistent/wikitext.txt"),
            binaries=fake_binaries,
        )
        raise AssertionError("expected FiTunaError for missing wikitext corpus")
    except FiTunaError:
        pass

    # 4. Missing binary -> FiTunaError (via FileNotFoundError translation).
    try:
        compute_perplexity(
            gguf_path=Path(__file__),
            wikitext_path=Path(__file__),  # any existing file stands in for corpus
            binaries=fake_binaries,
        )
        raise AssertionError("expected FiTunaError for missing llama-perplexity binary")
    except FiTunaError:
        pass

    # 5. Missing base logits file -> FiTunaError.
    try:
        compute_kld(
            quantized_gguf=Path(__file__),
            wikitext_path=Path(__file__),
            base_logits_path=Path("/nonexistent/base.kld"),
            binaries=fake_binaries,
        )
        raise AssertionError("expected FiTunaError for missing base logits")
    except FiTunaError:
        pass


if __name__ == "__main__":
    _self_check()
    print("fituna.quality self-check OK")
