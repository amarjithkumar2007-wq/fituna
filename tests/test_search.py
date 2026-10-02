# SPDX-License-Identifier: MIT
"""search() orchestration tests -- quantize/bench/quality are faked so this
suite exercises only the early-exit / binary-search / grid-search logic, no
real llama.cpp binaries required.

Every assertion here is checked directly against fituna.search.search()'s
module docstring and the interface contract in fituna/config.py -- no
xfail/skip markers, these are expected to pass outright.

Faking strategy
----------------
search() is expected to call the free functions ``quantize.quantize``,
``bench.run_bench`` and ``quality.evaluate_quality``/``compute_perplexity``.
Depending on whether the implementation does ``import fituna.quantize`` (and
calls ``fituna.quantize.quantize(...)``) or ``from fituna.quantize import
quantize`` (a name bound once, at import time, into ``fituna.search``'s own
namespace), monkeypatching only the source module may not be visible to
search(). ``_patch`` below defends against both import styles by patching
the attribute in both places; the ``raising=False`` half is a no-op if
search.py never imports the name directly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fituna.config import (
    BenchResult,
    BinaryPaths,
    CandidateConfig,
    GPUVendor,
    HardwareProfile,
    ModelInfo,
    NoFeasibleConfigError,
    QualityResult,
    TargetSpec,
)
from fituna.search import search


def _patch(monkeypatch, module: str, name: str, fake) -> None:
    monkeypatch.setattr(f"fituna.{module}.{name}", fake)
    monkeypatch.setattr(f"fituna.search.{name}", fake, raising=False)


def _quant_of(gguf_path: Path) -> str:
    """Recover the quant token from the fake quantize() naming convention
    used below (``model-<quant>.gguf``), mirroring fituna.bench's real
    ``_quant_from_filename`` helper."""
    return gguf_path.stem.rsplit("-", 1)[1]


def _hw(vendor: GPUVendor = GPUVendor.NVIDIA) -> HardwareProfile:
    return HardwareProfile(
        gpu_vendor=vendor, gpu_name="fake-gpu" if vendor != GPUVendor.NONE else None,
        vram_mb=8192 if vendor != GPUVendor.NONE else None, cpu_cores=8, ram_mb=32768,
        os_name="linux",
    )


def _binaries(tmp_path: Path) -> BinaryPaths:
    return BinaryPaths(
        llama_quantize=tmp_path / "llama-quantize",
        llama_bench=tmp_path / "llama-bench",
        llama_perplexity=tmp_path / "llama-perplexity",
    )


def _model_info(tmp_path: Path, n_layers: int = 8) -> ModelInfo:
    # model_fingerprint() (fituna.model_info) stat()s this path for real, so
    # it must actually exist on disk -- an empty placeholder is enough, its
    # *content* is irrelevant to search()'s early-exit/binary-search logic.
    base_gguf = tmp_path / "base-f16.gguf"
    base_gguf.touch()
    return ModelInfo(
        architecture="llama", n_layers=n_layers, n_params=7_000_000_000,
        base_gguf_path=base_gguf,
    )


def test_search_early_exits_on_first_hit(monkeypatch, tmp_path):
    """Highest-quality quant that clears both gates wins; the ngl binary
    search converges on the minimal ngl meeting the speed target; lower
    quality tiers are never even speed-tested (early exit on first hit)."""
    n_layers = 8
    target = TargetSpec(
        model_path=tmp_path / "model.gguf",
        target_tokens_per_sec=20.0,
        max_quality_loss_pct=5.0,
        quant_candidates=("Q8_0", "Q4_K_M", "Q2_K"),
        ngl_max_calls=6,
    )
    model_info = _model_info(tmp_path, n_layers=n_layers)
    hw = _hw(GPUVendor.NVIDIA)
    binaries = _binaries(tmp_path)

    quantize_calls: list[str] = []
    bench_calls: list[tuple[str, int]] = []

    def fake_quantize(base_gguf: Path, quant: str, out_dir: Path, binaries: BinaryPaths, model_fp: str) -> Path:
        quantize_calls.append(quant)
        return out_dir / f"model-{quant}.gguf"

    def fake_compute_perplexity(gguf_path, wikitext_path, binaries, chunks=None) -> float:
        return 6.0  # baseline

    # Every candidate clears the quality gate; Q8_0 is the best (lowest loss).
    _QUALITY_LOSS = {"Q8_0": 1.0, "Q4_K_M": 2.0, "Q2_K": 3.0}

    def fake_evaluate_quality(quant, quantized_gguf, baseline_ppl, wikitext_path, binaries, chunks=None):
        return QualityResult(
            candidate_quant=quant, perplexity=baseline_ppl * (1 + _QUALITY_LOSS[quant] / 100),
            baseline_perplexity=baseline_ppl, quality_loss_pct=_QUALITY_LOSS[quant],
        )

    def fake_run_bench(gguf_path, ngl, ctx, target, binaries, timeout_sec=300):
        quant = _quant_of(gguf_path)
        # Early exit on first hit means Q4_K_M/Q2_K must never be speed-tested
        # once Q8_0 already satisfies the target.
        assert quant == "Q8_0", f"lower-quality quant {quant!r} was speed-tested unnecessarily"
        bench_calls.append((quant, ngl))
        # Monotonic step function: gen tok/s jumps to a passing value at ngl=3.
        gen_tps = 25.0 if ngl >= 3 else 5.0
        cand = CandidateConfig(quant=quant, ngl=ngl, ctx=ctx)
        from fituna.config import BenchResult
        return BenchResult(candidate=cand, prompt_tok_per_sec=100.0, gen_tok_per_sec=gen_tps,
                            vram_used_mb=4096, raw_stdout="{}")

    _patch(monkeypatch, "quantize", "quantize", fake_quantize)
    _patch(monkeypatch, "quality", "compute_perplexity", fake_compute_perplexity)
    _patch(monkeypatch, "quality", "evaluate_quality", fake_evaluate_quality)
    _patch(monkeypatch, "bench", "run_bench", fake_run_bench)

    result = search(
        target=target, model_info=model_info, hw=hw, binaries=binaries,
        work_dir=tmp_path, wikitext_path=tmp_path / "wiki.txt", cache=None,
    )

    assert result.meets_target is True
    assert result.config.quant == "Q8_0"
    assert result.config.ngl == 3, "binary search must converge on the minimal passing ngl"
    assert result.bench.gen_tok_per_sec >= target.target_tokens_per_sec
    assert result.quality.candidate_quant == "Q8_0"
    assert result.quality.quality_loss_pct <= target.max_quality_loss_pct
    assert result.gguf_path == tmp_path / "model-Q8_0.gguf"

    # Never speed-tested the lower tiers.
    assert all(q == "Q8_0" for q, _ in bench_calls)
    # ngl binary search stayed within its call budget (n_layers/0 probes plus
    # the bisection itself).
    assert len(bench_calls) <= 2 + target.ngl_max_calls

    # A copy-pasteable run command reflecting the winning ngl.
    assert isinstance(result.run_command, list) and result.run_command
    assert all(isinstance(part, str) for part in result.run_command)
    assert "-ngl" in result.run_command
    assert str(result.config.ngl) in result.run_command


def test_search_raises_with_closest_when_no_candidate_meets_target(monkeypatch, tmp_path):
    """Every quant clears the quality gate but none reaches the target speed
    even at full GPU offload (early-exit B on every tier) -> per the spec
    ("모든 quant가 조기종료 B로 탈락 -> NoFeasibleConfigError(가장 빨랐던 시도
    정보 포함)") this is a genuine infeasibility: search must raise
    NoFeasibleConfigError carrying the fastest attempt seen as `closest`, not
    return a best-effort result. It must also not needlessly probe ngl=0 or
    bisect once the full-offload ceiling has already failed."""
    n_layers = 8
    target = TargetSpec(
        model_path=tmp_path / "model.gguf",
        target_tokens_per_sec=20.0,
        max_quality_loss_pct=5.0,
        quant_candidates=("Q8_0", "Q4_K_M"),
        ngl_max_calls=6,
    )
    model_info = _model_info(tmp_path, n_layers=n_layers)
    hw = _hw(GPUVendor.NVIDIA)
    binaries = _binaries(tmp_path)

    bench_calls: list[tuple[str, int]] = []
    _TOP_SPEED = {"Q8_0": 15.0, "Q4_K_M": 18.0}  # both below the 20.0 target

    def fake_quantize(base_gguf, quant, out_dir, binaries, model_fp):
        return out_dir / f"model-{quant}.gguf"

    def fake_compute_perplexity(gguf_path, wikitext_path, binaries, chunks=None):
        return 6.0

    def fake_evaluate_quality(quant, quantized_gguf, baseline_ppl, wikitext_path, binaries, chunks=None):
        return QualityResult(candidate_quant=quant, perplexity=baseline_ppl * 1.02,
                              baseline_perplexity=baseline_ppl, quality_loss_pct=2.0)

    def fake_run_bench(gguf_path, ngl, ctx, target, binaries, timeout_sec=300):
        quant = _quant_of(gguf_path)
        # Only the full-offload ceiling should ever be probed for a quant
        # tier that fails it -- no ngl=0 check, no bisection.
        assert ngl == n_layers, (
            f"quant {quant!r} was probed at ngl={ngl} but its full-offload "
            "ceiling already misses the target; no further probing needed"
        )
        bench_calls.append((quant, ngl))
        cand = CandidateConfig(quant=quant, ngl=ngl, ctx=ctx)
        from fituna.config import BenchResult
        return BenchResult(candidate=cand, prompt_tok_per_sec=100.0,
                            gen_tok_per_sec=_TOP_SPEED[quant], vram_used_mb=4096, raw_stdout="{}")

    _patch(monkeypatch, "quantize", "quantize", fake_quantize)
    _patch(monkeypatch, "quality", "compute_perplexity", fake_compute_perplexity)
    _patch(monkeypatch, "quality", "evaluate_quality", fake_evaluate_quality)
    _patch(monkeypatch, "bench", "run_bench", fake_run_bench)

    with pytest.raises(NoFeasibleConfigError) as excinfo:
        search(
            target=target, model_info=model_info, hw=hw, binaries=binaries,
            work_dir=tmp_path, wikitext_path=tmp_path / "wiki.txt", cache=None,
        )

    closest = excinfo.value.closest
    assert closest is not None, "closest must carry the fastest attempt seen"
    assert closest.meets_target is False
    # Best-effort = the fastest attempt seen, i.e. Q4_K_M's 18.0 tok/s.
    assert closest.bench.gen_tok_per_sec == 18.0
    assert closest.config.quant == "Q4_K_M"
    assert {q for q, _ in bench_calls} == {"Q8_0", "Q4_K_M"}, "every tier must be attempted"


def test_search_skips_binary_search_when_cpu_only_already_meets_target(monkeypatch, tmp_path):
    """If ngl=0 (CPU only) already meets the target, the ngl binary search
    must be skipped entirely and the minimal-resource config (ngl=0) adopted
    immediately (early exit C)."""
    n_layers = 8
    target = TargetSpec(
        model_path=tmp_path / "model.gguf",
        target_tokens_per_sec=20.0,
        max_quality_loss_pct=5.0,
        quant_candidates=("Q4_K_M",),
        ngl_max_calls=6,
    )
    model_info = _model_info(tmp_path, n_layers=n_layers)
    hw = _hw(GPUVendor.NVIDIA)
    binaries = _binaries(tmp_path)

    bench_calls: list[int] = []

    def fake_quantize(base_gguf, quant, out_dir, binaries, model_fp):
        return out_dir / f"model-{quant}.gguf"

    def fake_compute_perplexity(gguf_path, wikitext_path, binaries, chunks=None):
        return 6.0

    def fake_evaluate_quality(quant, quantized_gguf, baseline_ppl, wikitext_path, binaries, chunks=None):
        return QualityResult(candidate_quant=quant, perplexity=baseline_ppl * 1.02,
                              baseline_perplexity=baseline_ppl, quality_loss_pct=2.0)

    def fake_run_bench(gguf_path, ngl, ctx, target, binaries, timeout_sec=300):
        bench_calls.append(ngl)
        # Both the full-offload ceiling (n_layers) and CPU-only (0) already
        # clear the target -- any other ngl value means the binary search
        # ran when it should have been skipped.
        assert ngl in (0, n_layers), f"unexpected bisection probe at ngl={ngl}"
        cand = CandidateConfig(quant="Q4_K_M", ngl=ngl, ctx=ctx)
        from fituna.config import BenchResult
        return BenchResult(candidate=cand, prompt_tok_per_sec=100.0, gen_tok_per_sec=25.0,
                            vram_used_mb=None, raw_stdout="{}")

    _patch(monkeypatch, "quantize", "quantize", fake_quantize)
    _patch(monkeypatch, "quality", "compute_perplexity", fake_compute_perplexity)
    _patch(monkeypatch, "quality", "evaluate_quality", fake_evaluate_quality)
    _patch(monkeypatch, "bench", "run_bench", fake_run_bench)

    result = search(
        target=target, model_info=model_info, hw=hw, binaries=binaries,
        work_dir=tmp_path, wikitext_path=tmp_path / "wiki.txt", cache=None,
    )

    assert result.meets_target is True
    assert result.config.ngl == 0, "CPU-only already meets target; must not over-allocate GPU"
    assert sorted(bench_calls) == [0, n_layers], "must not bisect once ngl=0 already passes"


def test_search_raises_when_every_quant_fails_the_quality_gate(monkeypatch, tmp_path):
    """If no quant tier ever clears the quality gate, no bench is ever run at
    all -- per the contract this is the one case that raises
    NoFeasibleConfigError instead of returning a best-effort result."""
    target = TargetSpec(
        model_path=tmp_path / "model.gguf",
        target_tokens_per_sec=20.0,
        max_quality_loss_pct=1.0,  # unattainable ceiling
        quant_candidates=("Q8_0", "Q4_K_M"),
    )
    model_info = _model_info(tmp_path)
    hw = _hw(GPUVendor.NVIDIA)
    binaries = _binaries(tmp_path)

    def fake_quantize(base_gguf, quant, out_dir, binaries, model_fp):
        return out_dir / f"model-{quant}.gguf"

    def fake_compute_perplexity(gguf_path, wikitext_path, binaries, chunks=None):
        return 6.0

    def fake_evaluate_quality(quant, quantized_gguf, baseline_ppl, wikitext_path, binaries, chunks=None):
        # Every tier blows well past the 1.0% ceiling.
        return QualityResult(candidate_quant=quant, perplexity=baseline_ppl * 1.10,
                              baseline_perplexity=baseline_ppl, quality_loss_pct=10.0)

    def fake_run_bench(gguf_path, ngl, ctx, target, binaries, timeout_sec=300):
        raise AssertionError("no bench should run once every quant fails the quality gate")

    _patch(monkeypatch, "quantize", "quantize", fake_quantize)
    _patch(monkeypatch, "quality", "compute_perplexity", fake_compute_perplexity)
    _patch(monkeypatch, "quality", "evaluate_quality", fake_evaluate_quality)
    _patch(monkeypatch, "bench", "run_bench", fake_run_bench)

    with pytest.raises(NoFeasibleConfigError):
        search(
            target=target, model_info=model_info, hw=hw, binaries=binaries,
            work_dir=tmp_path, wikitext_path=tmp_path / "wiki.txt", cache=None,
        )


def test_search_returns_best_effort_on_time_budget_timeout(monkeypatch, tmp_path):
    """max_bench_seconds is the *only* case the contract has return a
    best-effort SearchResult (meets_target=False) instead of raising: "시간
    예산 초과 시 그 시점까지의 최고 속도 결과를 meets_target=False로 반환". A
    fake clock lets the deadline expire right after one successful bench call
    records a best-effort candidate, without needing a real sleep().

    search() now quantizes+measures quality for *every* candidate (stage 1)
    before benching *any* of them in measured-quality order (stage 2), so the
    clock must budget one time_left() check per quant in stage 1 (both Q8_0
    and Q4_K_M pass the quality gate here) plus one more at the top of stage
    2's first iteration, before the check that finally expires."""
    import fituna.search as search_module

    n_layers = 8
    target = TargetSpec(
        model_path=tmp_path / "model.gguf",
        target_tokens_per_sec=20.0,
        max_quality_loss_pct=5.0,
        quant_candidates=("Q8_0", "Q4_K_M"),
        ngl_max_calls=6,
        max_bench_seconds=10,
    )
    model_info = _model_info(tmp_path, n_layers=n_layers)
    hw = _hw(GPUVendor.NVIDIA)
    binaries = _binaries(tmp_path)

    def fake_quantize(base_gguf, quant, out_dir, binaries, model_fp):
        return out_dir / f"model-{quant}.gguf"

    def fake_compute_perplexity(gguf_path, wikitext_path, binaries, chunks=None):
        return 6.0

    def fake_evaluate_quality(quant, quantized_gguf, baseline_ppl, wikitext_path, binaries, chunks=None):
        return QualityResult(candidate_quant=quant, perplexity=baseline_ppl * 1.02,
                              baseline_perplexity=baseline_ppl, quality_loss_pct=2.0)

    bench_calls: list[str] = []

    def fake_run_bench(gguf_path, ngl, ctx, target, binaries, timeout_sec=300):
        quant = _quant_of(gguf_path)
        bench_calls.append(quant)
        # Only Q8_0's full-offload bench should ever run: the clock expires
        # right after it, so Q4_K_M must never be reached.
        assert quant == "Q8_0", f"{quant!r} benched after the time budget should have expired"
        cand = CandidateConfig(quant=quant, ngl=ngl, ctx=ctx)
        from fituna.config import BenchResult
        return BenchResult(candidate=cand, prompt_tok_per_sec=100.0, gen_tok_per_sec=30.0,
                            vram_used_mb=4096, raw_stdout="{}")

    # Clock schedule: start=0 (deadline=10); one time_left() check per quant
    # in stage 1 (Q8_0, Q4_K_M) plus one at the top of stage 2's first
    # iteration all read as "time remaining"; the check right after the
    # full-offload bench reads past the deadline, forcing an immediate
    # timeout break.
    clock = iter([0.0, 1.0, 2.0, 3.0, 100.0])

    def fake_monotonic():
        return next(clock, 100.0)

    _patch(monkeypatch, "quantize", "quantize", fake_quantize)
    _patch(monkeypatch, "quality", "compute_perplexity", fake_compute_perplexity)
    _patch(monkeypatch, "quality", "evaluate_quality", fake_evaluate_quality)
    _patch(monkeypatch, "bench", "run_bench", fake_run_bench)
    monkeypatch.setattr(search_module.time, "monotonic", fake_monotonic)

    result = search(
        target=target, model_info=model_info, hw=hw, binaries=binaries,
        work_dir=tmp_path, wikitext_path=tmp_path / "wiki.txt", cache=None,
    )

    assert result.meets_target is False, "timeout must yield best-effort, not meets_target=True"
    assert result.config.quant == "Q8_0"
    assert result.bench.gen_tok_per_sec == 30.0
    assert bench_calls == ["Q8_0"], "Q4_K_M must never be reached once the budget is spent"


def test_hardware_fingerprint_varies_with_llama_version():
    from fituna.search import _hardware_fingerprint

    hw = _hw()
    a = _hardware_fingerprint(hw, "3765 (c919d5d)")
    b = _hardware_fingerprint(hw, "9960 (abc1234)")
    c = _hardware_fingerprint(hw, None)
    assert len({a, b, c}) == 3, "llama.cpp version must namespace the bench cache"
    assert a == _hardware_fingerprint(hw, "3765 (c919d5d)"), "must stay deterministic"


def test_bench_timeout_treated_as_below_target_not_abort(monkeypatch, tmp_path):
    """A BenchTimeoutError from one bench call must record 0 tok/s and let the
    search continue to other candidates, not kill the whole search."""
    from fituna.config import BenchResult, BenchTimeoutError, NoFeasibleConfigError

    _patch(monkeypatch, "quantize", "quantize",
           lambda base, quant, work, bins, fp: tmp_path / f"m-{quant}.gguf")
    _patch(monkeypatch, "quality", "compute_perplexity",
           lambda *a, **k: 10.0)
    _patch(monkeypatch, "quality", "evaluate_quality",
           lambda quant, gguf, base_ppl, wiki, bins, chunks=None: QualityResult(
               candidate_quant=quant, perplexity=10.0,
               baseline_perplexity=10.0, quality_loss_pct=0.0))
    _patch(monkeypatch, "binaries", "list_supported_quant_types",
           lambda bins: ["Q8_0", "Q4_K_M"])

    def bench(gguf_path, ngl, ctx, target, binaries, timeout_sec=300):
        if _quant_of(gguf_path) == "Q8_0":
            raise BenchTimeoutError("llama-bench timed out after 300s")
        return BenchResult(
            candidate=CandidateConfig(quant=_quant_of(gguf_path), ngl=ngl, ctx=ctx),
            prompt_tok_per_sec=100.0, gen_tok_per_sec=50.0,
            vram_used_mb=None, raw_stdout="{}")

    _patch(monkeypatch, "bench", "run_bench", bench)

    target = TargetSpec(
        model_path=tmp_path / "m.gguf", target_tokens_per_sec=40.0,
        max_quality_loss_pct=5.0, ctx=4096, ctx_candidates=[4096],
        quant_candidates=["Q8_0", "Q4_K_M"])
    result = search(target, _model_info(tmp_path), _hw(GPUVendor.NONE),
                    _binaries(tmp_path), tmp_path, tmp_path / "wiki.txt")
    assert result.meets_target
    assert result.config.quant == "Q4_K_M"


def test_search_with_kld_quality_metric(monkeypatch, tmp_path):
    """When quality_metric='kld', search() generates baseline logits and passes
    metric='kld' and base_logits_path to evaluate_quality()."""
    generated_logits: list[Path] = []
    evaluated_with_metric: list[str] = []

    _patch(monkeypatch, "quantize", "quantize",
           lambda base, quant, work, bins, fp: tmp_path / f"m-{quant}.gguf")
    _patch(monkeypatch, "quality", "compute_perplexity",
           lambda *a, **k: 6.0)

    def fake_gen_logits(base, wiki, logits_path, bins, chunks=None):
        generated_logits.append(logits_path)
        logits_path.touch()
        return logits_path

    _patch(monkeypatch, "quality", "generate_base_logits", fake_gen_logits)

    def fake_eval_quality(quant, gguf, base_ppl, wiki, bins, chunks=None, metric="ppl", base_logits_path=None):
        evaluated_with_metric.append(metric)
        assert base_logits_path is not None
        assert base_logits_path.exists()
        return QualityResult(
            candidate_quant=quant,
            perplexity=6.1,
            baseline_perplexity=6.0,
            quality_loss_pct=1.67,
            metric=metric,
            kld=0.003,
        )

    _patch(monkeypatch, "quality", "evaluate_quality", fake_eval_quality)
    _patch(monkeypatch, "binaries", "list_supported_quant_types", lambda bins: ["Q4_K_M"])
    _patch(
        monkeypatch, "bench", "run_bench",
        lambda gguf_path, ngl, ctx, target, binaries, timeout_sec=300: BenchResult(
            candidate=CandidateConfig(quant=_quant_of(gguf_path), ngl=ngl, ctx=ctx),
            prompt_tok_per_sec=100.0, gen_tok_per_sec=50.0,
            vram_used_mb=None, raw_stdout="{}",
        ),
    )

    target = TargetSpec(
        model_path=tmp_path / "m.gguf",
        target_tokens_per_sec=20.0,
        max_quality_loss_pct=5.0,
        ctx=4096,
        ctx_candidates=[4096],
        quant_candidates=["Q4_K_M"],
        quality_metric="kld",
    )
    result = search(
        target,
        _model_info(tmp_path),
        _hw(GPUVendor.NONE),
        _binaries(tmp_path),
        tmp_path,
        tmp_path / "wiki.txt",
    )
    assert result.meets_target
    assert len(generated_logits) == 1
    assert evaluated_with_metric == ["kld"]
    assert result.quality.metric == "kld"
    assert result.quality.kld == 0.003

