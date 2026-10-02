# SPDX-License-Identifier: MIT
"""fituna.bench
===============

Runs ``llama-bench`` for a single (quant, ngl, ctx) candidate and parses its
JSON output into a :class:`~fituna.config.BenchResult`. Generation
throughput (``gen_tok_per_sec``) is the primary metric the search algorithm
in :mod:`fituna.search` optimizes against.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from fituna.config import (
    BenchResult,
    BenchTimeoutError,
    BinaryPaths,
    CandidateConfig,
    FiTunaError,
    TargetSpec,
)


def _quant_from_filename(gguf_path: Path) -> str:
    """Best-effort quant-type token from a ``<model>-<QUANT>.gguf`` filename.

    llama-bench's JSON reports ``model_filename``/``model_type`` but no clean
    quant token, so the CandidateConfig attached to the BenchResult is
    reconstructed from the naming convention fituna.quantize.quantize() uses
    (``<model>-<quant>.gguf``).
    naive suffix split on the last "-"; good enough since fituna
    only ever benches files it named itself. Falls back to the full stem for
    anything that doesn't match.
    """
    stem = gguf_path.stem
    return stem.rsplit("-", 1)[1] if "-" in stem else stem


def _parse_bench_json(stdout: str, gguf_path: Path, ngl: int, ctx: int) -> BenchResult:
    """Parse ``llama-bench -o json`` stdout into a BenchResult.

    llama-bench emits one JSON record per sub-test (prompt-processing and
    text-generation). Records are told apart by ``n_prompt``/``n_gen`` being
    nonzero rather than by parsing the ``test`` name string (e.g. "pp512"),
    since that label format has changed across llama.cpp releases.
    A generation record and its ``avg_ts`` are required because generation
    throughput drives the search verdict. Prompt throughput stays optional.
    """
    try:
        records: Any = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise FiTunaError(
            f"could not parse llama-bench JSON output: {exc}\n--- stdout ---\n{stdout}"
        ) from exc

    if not isinstance(records, list) or not records:
        raise FiTunaError(f"llama-bench produced no test records:\n{stdout}")

    prompt_ts = 0.0
    gen_ts = None
    vram_used_mb = None
    for rec in records:
        if not isinstance(rec, dict):
            continue
        if rec.get("n_gen", 0) > 0:
            if "avg_ts" not in rec:
                raise FiTunaError(
                    f"llama-bench text-generation record is missing avg_ts:\n"
                    f"--- stdout ---\n{stdout}"
                )
            gen_ts = float(rec["avg_ts"])
        elif rec.get("n_prompt", 0):
            prompt_ts = float(rec.get("avg_ts", 0.0))
        # llama-bench does not currently report VRAM usage; the field is
        # kept for a future backend/version that might add it.
        if "vram_used_mb" in rec:
            vram_used_mb = int(rec["vram_used_mb"])

    if gen_ts is None:
        raise FiTunaError(
            f"llama-bench produced no text-generation test record:\n"
            f"--- stdout ---\n{stdout}"
        )

    candidate = CandidateConfig(quant=_quant_from_filename(gguf_path), ngl=ngl, ctx=ctx)
    return BenchResult(
        candidate=candidate,
        prompt_tok_per_sec=prompt_ts,
        gen_tok_per_sec=gen_ts,
        vram_used_mb=vram_used_mb,
        raw_stdout=stdout,
    )


def run_bench(
    gguf_path: Path,
    ngl: int,
    ctx: int,
    target: TargetSpec,
    binaries: BinaryPaths,
    timeout_sec: int = 300,
) -> BenchResult:
    """Run:
        llama-bench -m <gguf_path> -ngl <ngl> -d <depth>
                     -p <target.prompt_tokens> -n <target.gen_tokens> -o json
    and parse stdout into a BenchResult.

    Current llama-bench has no direct ``-c``/``--ctx-size`` flag -- it
    allocates the context each test needs from n_prompt + n_gen + n_depth
    instead (confirmed against a real build: passing ``-c`` is a hard
    argument-parse error, not a silently-ignored flag). ``-d``/``--n-depth``
    (tokens already resident in the KV cache before the timed prompt/gen
    phase) is what reproduces "generation speed once the context is filled
    to `ctx` tokens" -- the thing a ``-c <ctx>`` flag used to mean here.
    ``depth = max(0, ctx - prompt_tokens - gen_tokens)`` fills the cache to
    (as close as possible to) ``ctx`` tokens by the time the timed
    generation phase runs.

    Raises BenchTimeoutError on timeout; FiTunaError on non-zero exit,
    failure to launch the binary, or unparsable/empty output.
    """
    depth = max(0, ctx - target.prompt_tokens - target.gen_tokens)
    cmd = [
        str(binaries.llama_bench),
        "-m", str(gguf_path),
        "-ngl", str(ngl),
        "-d", str(depth),
        "-p", str(target.prompt_tokens),
        "-n", str(target.gen_tokens),
        "-o", "json",
    ]
    try:
        # encoding/errors explicit: see hardware.py's _run for why.
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_sec,
            encoding="utf-8", errors="replace",
        )
    except subprocess.TimeoutExpired as exc:
        raise BenchTimeoutError(
            f"llama-bench timed out after {timeout_sec}s: {' '.join(cmd)}"
        ) from exc
    except OSError as exc:
        raise FiTunaError(
            f"failed to launch llama-bench ({binaries.llama_bench}): {exc}"
        ) from exc

    if proc.returncode != 0:
        raise FiTunaError(
            f"llama-bench exited with code {proc.returncode}\n"
            f"command: {' '.join(cmd)}\n--- stderr ---\n{proc.stderr}"
        )

    return _parse_bench_json(proc.stdout, gguf_path, ngl, ctx)


def _self_check() -> None:
    """Minimal assert-based sanity check (no real llama-bench binary needed).

    Exercises the JSON-parsing contract against the checked-in llama-bench
    fixture, plus the three subprocess failure modes run_bench must turn
    into FiTunaError: timeout, non-zero exit, and malformed stdout.
    """
    from unittest import mock

    fixture_path = (
        Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "llama_bench_sample.json"
    )
    fixture_json = fixture_path.read_text()

    binaries = BinaryPaths(
        llama_quantize=Path("/usr/local/bin/llama-quantize"),
        llama_bench=Path("/usr/local/bin/llama-bench"),
        llama_perplexity=Path("/usr/local/bin/llama-perplexity"),
    )
    target = TargetSpec(model_path=Path("model.gguf"), target_tokens_per_sec=20.0,
                         max_quality_loss_pct=5.0)
    gguf = Path("/work/llama-3-8b-instruct-Q4_K_M.gguf")

    # 1. Happy path: JSON fixture parses into the expected BenchResult.
    fake_ok = mock.Mock(returncode=0, stdout=fixture_json, stderr="")
    with mock.patch("subprocess.run", return_value=fake_ok) as run_mock:
        result = run_bench(gguf, ngl=32, ctx=4096, target=target, binaries=binaries)
    assert result.prompt_tok_per_sec == 120.5
    assert result.gen_tok_per_sec == 22.8
    assert result.vram_used_mb is None
    assert result.candidate == CandidateConfig(quant="Q4_K_M", ngl=32, ctx=4096)
    assert result.raw_stdout == fixture_json
    called_cmd = run_mock.call_args[0][0]
    assert called_cmd[:2] == ["/usr/local/bin/llama-bench", "-m"]
    assert "-ngl" in called_cmd and "32" in called_cmd
    assert "-o" in called_cmd and "json" in called_cmd

    # 2. Timeout -> FiTunaError, not a raw TimeoutExpired leaking out.
    with mock.patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="llama-bench", timeout=5)):
        try:
            run_bench(gguf, ngl=0, ctx=4096, target=target, binaries=binaries, timeout_sec=5)
            raise AssertionError("expected FiTunaError on timeout")
        except FiTunaError:
            pass

    # 3. Non-zero exit -> FiTunaError carrying stderr for diagnosis.
    fake_fail = mock.Mock(returncode=1, stdout="", stderr="error: failed to load model")
    with mock.patch("subprocess.run", return_value=fake_fail):
        try:
            run_bench(gguf, ngl=0, ctx=4096, target=target, binaries=binaries)
            raise AssertionError("expected FiTunaError on non-zero exit")
        except FiTunaError as exc:
            assert "failed to load model" in str(exc)

    # 4. Malformed JSON -> FiTunaError, not an uncaught JSONDecodeError.
    fake_bad = mock.Mock(returncode=0, stdout="not json", stderr="")
    with mock.patch("subprocess.run", return_value=fake_bad):
        try:
            run_bench(gguf, ngl=0, ctx=4096, target=target, binaries=binaries)
            raise AssertionError("expected FiTunaError on malformed JSON")
        except FiTunaError:
            pass

    # 5. Quant-from-filename fallback for a name without a "-" separator.
    assert _quant_from_filename(Path("plainname.gguf")) == "plainname"


if __name__ == "__main__":
    _self_check()
    print("fituna.bench self-check OK")
