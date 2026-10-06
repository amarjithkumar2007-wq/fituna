# SPDX-License-Identifier: MIT
"""fituna.search
================

The core orchestrator: quality-first filtering over ``quant_candidates``
(highest quality first), grid search over ``ctx_candidates``, binary search
over ``ngl`` (bounded by ``ngl_max_calls``), with early-exit as soon as a
candidate meets both the target throughput and the quality-loss ceiling.

Algorithm (implemented below, matches the spec's two-stage design):

Stage 1 -- quality pre-filter (one ``llama-perplexity`` call per quant):
    baseline_ppl is computed once (and cached) on the unquantized base GGUF.
    Each quant is quantized (idempotent) and its quality_loss_pct measured;
    quants whose loss exceeds ``max_quality_loss_pct`` are skipped entirely
    (early-exit A) -- they never reach the benchmarking stage.

Stage 2 -- speed search, walking quants in quality-descending order:
    - top = bench at ngl=n_layers+1 (full offload: llama.cpp counts the
      output layer as one more layer), or at ngl=0 on no-GPU hardware. If
      even that misses the target throughput, this quant can't work at any
      ngl -- skip to the next, lower-quality (often faster) quant
      (early-exit B).
    - if hardware has a GPU, low = bench at ngl=0. If that alone already
      meets the target, no GPU offload is needed -- adopt it immediately
      (early-exit C), skipping the binary search entirely.
    - otherwise binary-search the minimal ngl in [0, n_layers+1] that meets
      the target (<= ngl_max_calls calls), assuming gen_tok_per_sec is
      monotonically non-decreasing in ngl (documented assumption; the worst
      case if it's violated is simply falling back to the already-known-good
      `top` result, so this stays safe).
    - extra ctx_candidates are part of the ngl predicate: an ngl only
      counts as meeting the target if every ctx candidate does.
    - the first quant (in quality order) that produces a working candidate
      wins immediately -- lower-quality quants are never tried.

If no quant/ngl/ctx combination meets the target, the loop naturally
exhausts and :class:`~fituna.config.NoFeasibleConfigError` is raised,
carrying the best (fastest) attempt seen as ``closest`` if any benchmark
ran at all. If the search is cut short by ``target.max_bench_seconds``
instead, the best-effort result is returned normally with
``meets_target=False`` (a time-out is not proof of infeasibility, so it
does not raise).
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import replace
from pathlib import Path
from typing import Callable, Optional

from fituna.binaries import get_llama_cpp_version, list_supported_quant_types
from fituna.bench import run_bench
from fituna.cache import ResultCache
from fituna.config import (
    BenchResult,
    BenchTimeoutError,
    BinaryPaths,
    CandidateConfig,
    GPUVendor,
    HardwareProfile,
    ModelInfo,
    NoFeasibleConfigError,
    QualityResult,
    SearchResult,
    TargetSpec,
)
from fituna.model_info import model_fingerprint
from fituna.quality import compute_perplexity, evaluate_quality, generate_base_logits
from fituna.quantize import quantize
from fituna.report import build_run_command, build_server_command

# Sentinel quant key used to cache the base-GGUF baseline perplexity inside
# quality_cache (which is keyed by (model_fp, quant)). cache.py's
# contract only exposes get_quality/put_quality keyed by quant name, so
# rather than growing the cache schema for one extra value, the baseline is
# stashed under a quant name real llama-quantize output can never produce.
_BASELINE_QUANT_KEY = "__baseline__"

# Every real llama-quantize quant type (Q4_K_M, IQ2_XXS, F16, ...) is plain
# alphanumerics/underscores. Whitelisting this shape blocks a quant string
# containing "/" or ".." from ever reaching a path-join in quantize.py.
_SAFE_QUANT_RE = re.compile(r"^[A-Za-z0-9_]+$")


def _hardware_fingerprint(hw: HardwareProfile, llama_version: Optional[str]) -> str:
    """Coarse, deterministic cache-key namespace for a hardware profile.

    ``llama_version`` is part of the key: benchmark numbers are a property of
    the llama.cpp build as much as of the hardware (a backend speedup between
    builds changes gen_tok_per_sec on identical hardware), so results measured
    under one build must not be served as --resume cache hits under another.
    ``None`` (version undetectable) still produces a stable key.

    Not part of the cross-module contract (cache.py only takes an opaque
    ``hw_fp`` string) so it stays private to this module.
    """
    raw = (
        f"{hw.gpu_vendor.value}|{hw.gpu_name}|{hw.vram_mb}|{hw.cpu_cores}"
        f"|{hw.ram_mb}|{hw.os_name}|llama={llama_version or 'unknown'}"
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def search(
    target: TargetSpec,
    model_info: ModelInfo,
    hw: HardwareProfile,
    binaries: BinaryPaths,
    work_dir: Path,
    wikitext_path: Path,
    cache: Optional[ResultCache] = None,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> SearchResult:
    """Run the quality-first / grid-search / ngl-binary-search algorithm
    described in the module docstring. Reports progress via
    ``progress_cb(str)`` if provided.
    """
    progress: Callable[[str], None] = progress_cb or (lambda _msg: None)

    start = time.monotonic()
    deadline = start + target.max_bench_seconds if target.max_bench_seconds else None

    def time_left() -> bool:
        return deadline is None or time.monotonic() < deadline

    model_fp = model_fingerprint(model_info.base_gguf_path)
    hw_fp = _hardware_fingerprint(hw, get_llama_cpp_version(binaries))
    # Same cheap name:size:mtime fingerprint as the model's -- perplexity is
    # a property of (model, quant, corpus), so the corpus must namespace the
    # quality cache or switching --quality-corpus under --resume would
    # silently serve measurements from the previous corpus. Only computed
    # when a cache exists: it stats the file, and cache-less runs never
    # touch the corpus until llama-perplexity does.
    corpus_fp = model_fingerprint(wikitext_path) if cache is not None else ""

    # ctx grid: target.ctx is always the primary (recorded) ctx; any other
    # ctx_candidates are re-verified at the winning ngl but never recorded
    # as the chosen CandidateConfig.ctx.
    other_ctxs = [c for c in dict.fromkeys(target.ctx_candidates) if c != target.ctx]

    # Offload ceiling. llama.cpp counts the output layer as one more layer:
    # -ngl n_layers leaves one layer on the CPU ("offloaded 30/31 layers"), and
    # n_layers + 1 is full offload (larger values are clamped). With no GPU
    # the ceiling is 0, the -ngl the result will recommend: a GPU build of
    # llama.cpp still offloads at -ngl n, so benching there would measure a
    # GPU speed for a command that runs on the CPU.
    max_ngl = 0 if hw.gpu_vendor == GPUVendor.NONE else model_info.n_layers + 1

    # Filter quant_candidates down to what this llama-quantize build actually
    # supports. Introspection failures are non-fatal -- fall back to trusting
    # the caller's list rather than blocking the whole search on it.
    try:
        supported = set(list_supported_quant_types(binaries))
    except Exception as exc:  # noqa: BLE001 -- best-effort introspection
        progress(f"could not determine supported quant types ({exc}); "
                  "proceeding without filtering")
        supported = None

    # Every quant token ends up in a filename (quantize.py) and on a
    # subprocess argv (as a plain arg, never via a shell, so no injection
    # risk there) -- but if introspection above failed, an unfiltered quant
    # string could still contain "/" or ".." and land outside out_dir when
    # used to build a path. Whitelist regardless of whether introspection
    # succeeded, rather than only in the fallback branch.
    unsafe = [q for q in target.quant_candidates if not _SAFE_QUANT_RE.match(q)]
    if unsafe:
        raise NoFeasibleConfigError(
            f"invalid quant candidate(s) {unsafe}: must match {_SAFE_QUANT_RE.pattern!r}"
        )

    quant_order = [q for q in target.quant_candidates if supported is None or q in supported]
    if not quant_order:
        raise NoFeasibleConfigError(
            "none of the requested quant candidates "
            f"{target.quant_candidates} are supported by this llama-quantize "
            f"build (supported: {sorted(supported) if supported else 'unknown'})"
        )

    # Baseline perplexity: one call, cached across runs via the sentinel key.
    baseline_ppl: Optional[float] = None
    if cache is not None:
        cached_baseline = cache.get_quality(
            model_fp, _BASELINE_QUANT_KEY, target.ppl_chunks, corpus_fp
        )
        if cached_baseline is not None:
            baseline_ppl = cached_baseline.perplexity
    if baseline_ppl is None:
        progress("computing baseline perplexity on base GGUF")
        baseline_ppl = compute_perplexity(
            model_info.base_gguf_path, wikitext_path, binaries, target.ppl_chunks
        )
        if cache is not None:
            cache.put_quality(
                model_fp,
                QualityResult(
                    candidate_quant=_BASELINE_QUANT_KEY,
                    perplexity=baseline_ppl,
                    baseline_perplexity=baseline_ppl,
                    quality_loss_pct=0.0,
                ),
                target.ppl_chunks,
                corpus_fp,
            )

    base_logits_path: Optional[Path] = None
    if target.quality_metric == "kld":
        # The reference logits depend on the base model, the corpus and the
        # chunk count; key the filename on all three so changing
        # --ppl-chunks or --quality-corpus never reuses a stale file.
        # A missing corpus falls back to its path here so the friendly
        # FiTunaError from generate_base_logits() is what the user sees.
        corpus_key = (
            model_fingerprint(wikitext_path) if wikitext_path.exists() else str(wikitext_path)
        )
        logits_key = hashlib.sha256(
            f"{model_fp}:{corpus_key}:{target.ppl_chunks}".encode("utf-8")
        ).hexdigest()[:12]
        base_logits_path = work_dir / f"{model_info.base_gguf_path.stem}.{logits_key}.kld"
        if not base_logits_path.exists():
            progress("generating baseline logits for KLD measurement on base GGUF")
            generate_base_logits(
                model_info.base_gguf_path,
                wikitext_path,
                base_logits_path,
                binaries,
                target.ppl_chunks,
            )

    best_effort: Optional[SearchResult] = None
    best_effort_speed = float("-inf")
    timed_out = False

    # --- Stage 1: quantize + measure quality for every candidate, then walk
    # Stage 2 in *measured* quality order -- not the caller-supplied
    # quant_candidates order, which is only a hint and can disagree with the
    # actual measured quality_loss_pct (e.g. a build/dataset where Q6_K
    # happens to lose less than Q8_0). Trusting the input order here would
    # silently violate the "highest quality first" guarantee.
    qualified: list[tuple[str, Path, QualityResult]] = []
    for quant in quant_order:
        if not time_left():
            timed_out = True
            break

        progress(f"[{quant}] quantizing")
        cand_gguf = quantize(model_info.base_gguf_path, quant, work_dir, binaries, model_fp)

        quality_res = (
            cache.get_quality(
                model_fp, quant, target.ppl_chunks, corpus_fp, metric=target.quality_metric
            )
            if cache is not None
            else None
        )
        if quality_res is None:
            progress(f"[{quant}] evaluating quality")
            if target.quality_metric == "kld":
                quality_res = evaluate_quality(
                    quant,
                    cand_gguf,
                    baseline_ppl,
                    wikitext_path,
                    binaries,
                    target.ppl_chunks,
                    metric=target.quality_metric,
                    base_logits_path=base_logits_path,
                )
            else:
                quality_res = evaluate_quality(
                    quant,
                    cand_gguf,
                    baseline_ppl,
                    wikitext_path,
                    binaries,
                    target.ppl_chunks,
                )
            if cache is not None:
                cache.put_quality(model_fp, quality_res, target.ppl_chunks, corpus_fp)

        if quality_res.quality_loss_pct > target.max_quality_loss_pct:
            progress(
                f"[{quant}] quality loss {quality_res.quality_loss_pct:.2f}% > "
                f"{target.max_quality_loss_pct:.2f}% cap, skipping (early-exit A)"
            )
            continue

        qualified.append((quant, cand_gguf, quality_res))

    # Re-sort by measured quality_loss_pct ascending (lowest loss = highest
    # quality first). Python's sort is stable, so ties keep their relative
    # quant_candidates order.
    qualified.sort(key=lambda item: item[2].quality_loss_pct)

    for quant, cand_gguf, quality_res in qualified:
        if not time_left():
            timed_out = True
            break

        # --- helpers closed over this iteration's quant/gguf/quality --------
        def cached_bench(ngl: int, ctx: int) -> BenchResult:
            cand = CandidateConfig(quant=quant, ngl=ngl, ctx=ctx)
            if cache is not None:
                hit = cache.get_bench(model_fp, hw_fp, cand)
                if hit is not None:
                    return hit
            try:
                res = run_bench(cand_gguf, ngl, ctx, target, binaries)
            except BenchTimeoutError:
                # A config too slow to finish one bench inside the timeout is
                # a *measurement* ("below any realistic target"), not a search
                # abort -- e.g. ngl=0 on a mid-size model during the minimal-
                # ngl binary search. Record it as 0 tok/s so the search walks
                # on; cached so --resume never pays the timeout twice.
                progress(
                    f"[{quant}] ngl={ngl} bench timed out -- treating as "
                    "0 tok/s (below target)"
                )
                res = BenchResult(
                    candidate=cand,
                    prompt_tok_per_sec=0.0,
                    gen_tok_per_sec=0.0,
                    vram_used_mb=None,
                    raw_stdout="bench timed out",
                )
            if cache is not None:
                cache.put_bench(model_fp, hw_fp, res)
            return res

        def consider_best_effort(bench: BenchResult) -> None:
            nonlocal best_effort, best_effort_speed
            if bench.gen_tok_per_sec > best_effort_speed:
                best_effort_speed = bench.gen_tok_per_sec
                best_effort = SearchResult(
                    config=bench.candidate,
                    bench=bench,
                    quality=quality_res,
                    gguf_path=cand_gguf,
                    run_command=build_run_command(cand_gguf, bench.candidate, binaries),
                    llama_server_command=build_server_command(
                        cand_gguf, bench.candidate, binaries
                    ),
                    meets_target=False,
                )

        def build_result(bench: BenchResult) -> SearchResult:
            return SearchResult(
                config=bench.candidate,
                bench=bench,
                quality=quality_res,
                gguf_path=cand_gguf,
                run_command=build_run_command(cand_gguf, bench.candidate, binaries),
                llama_server_command=build_server_command(
                    cand_gguf, bench.candidate, binaries
                ),
                meets_target=True,
            )

        def verify_other_ctx(ngl: int) -> bool:
            nonlocal timed_out
            for ctx_val in other_ctxs:
                if not time_left():
                    timed_out = True
                    return False
                r = cached_bench(ngl, ctx_val)
                consider_best_effort(r)
                if r.gen_tok_per_sec < target.target_tokens_per_sec:
                    progress(
                        f"[{quant}] ngl={ngl} meets target at ctx={target.ctx} "
                        f"but fails at ctx={ctx_val}"
                    )
                    return False
            return True

        # --- Stage 2: speed search -------------------------------------------
        progress(f"[{quant}] bench full-offload (ngl={max_ngl})")
        top = cached_bench(max_ngl, target.ctx)
        consider_best_effort(top)

        if top.gen_tok_per_sec < target.target_tokens_per_sec:
            progress(
                f"[{quant}] full-offload {top.gen_tok_per_sec:.2f} tok/s < "
                f"target {target.target_tokens_per_sec:.2f}, skipping (early-exit B)"
            )
            continue

        if not time_left():
            timed_out = True
            break

        if hw.gpu_vendor == GPUVendor.NONE:
            # No GPU: ngl search doesn't apply; `top` already ran at ngl=0.
            if verify_other_ctx(0):
                return build_result(top)
            continue

        # Every ctx must hold at the chosen ngl. If even full offload misses
        # one, no smaller ngl will (monotonic assumption) -- next quant.
        if not verify_other_ctx(max_ngl):
            if timed_out:
                break
            continue

        low = cached_bench(0, target.ctx)
        consider_best_effort(low)
        if low.gen_tok_per_sec >= target.target_tokens_per_sec and verify_other_ctx(0):
            progress(f"[{quant}] zero-offload already meets target (early-exit C)")
            return build_result(low)

        if not time_left():
            timed_out = True
            break

        # Binary search the minimal ngl in [0, max_ngl] meeting the target at
        # every ctx candidate (not just the first -- a larger ctx can need
        # more offload than the first ctx's minimum).
        lo, hi = 0, max_ngl
        best = top  # already known to satisfy the target at every ctx
        calls = 0
        while lo < hi and calls < target.ngl_max_calls and time_left():
            mid = (lo + hi) // 2
            r = cached_bench(mid, target.ctx)
            consider_best_effort(r)
            calls += 1
            if r.gen_tok_per_sec >= target.target_tokens_per_sec and verify_other_ctx(mid):
                best = r
                hi = mid
            else:
                lo = mid + 1
        if not time_left():
            timed_out = True

        progress(f"[{quant}] found ngl={best.candidate.ngl} meeting target -- done")
        return build_result(best)

    if timed_out:
        # A time-out is not proof that no config is feasible -- return the
        # best-effort result (or, if nothing was ever benched, raise).
        if best_effort is not None:
            progress("time budget exceeded; returning best-effort result")
            return best_effort
        raise NoFeasibleConfigError(
            "time budget exceeded before any candidate could be benchmarked"
        )

    # The quant sweep ran to exhaustion without a single hit: no config
    # actually meets the target, so this is a genuine infeasibility, not a
    # best-effort situation. Raise, but attach the fastest attempt seen (if
    # any) as `closest` so the caller can still inspect it.
    progress("no candidate met target after exhausting all quant candidates")
    raise NoFeasibleConfigError(
        "no quant/ngl/ctx combination met target_tokens_per_sec within "
        "max_quality_loss_pct",
        closest=best_effort,
    )


# ---------------------------------------------------------------------------
# self-check (run: python -m fituna.search)
# ---------------------------------------------------------------------------


def _self_check() -> None:
    """Assert-based sanity check exercising the early-exit / binary-search /
    grid-search logic with fake quantize/run_bench/quality/etc., no real
    llama.cpp binaries required.

    Monkeypatches this module's own imported names directly (plain global
    reassignment, restored in `finally`) rather than pulling in
    unittest.mock/pytest -- keeps this runnable as a plain script.
    """
    import sys

    # `import fituna.search as _mod` would re-import a *second*,
    # freshly-loaded module object when this file is run as `__main__`
    # (python -m fituna.search runs it under the name "__main__", separate
    # from any "fituna.search" entry in sys.modules) -- patches on that
    # second copy would never be seen by the `search()` actually running
    # here. sys.modules[__name__] always resolves to *this* running module,
    # whatever name it was loaded under.
    _mod = sys.modules[__name__]

    binaries_stub = BinaryPaths(
        llama_quantize=Path("llama-quantize"),
        llama_bench=Path("llama-bench"),
        llama_perplexity=Path("llama-perplexity"),
    )
    model_info_stub = ModelInfo(
        architecture="llama", n_layers=32, n_params=7_000_000_000,
        base_gguf_path=Path("base-f16.gguf"),
    )
    work_dir = Path(".")
    wikitext = Path("wiki.test.raw")

    quality_map = {"Q8_0": 1.0, "Q6_K": 3.0, "Q4_K_M": 10.0}

    def fake_quantize(base_gguf, quant, out_dir, binaries, model_fp):
        return Path(out_dir) / f"model-{quant}.gguf"

    def fake_compute_perplexity(gguf_path, wikitext_path, binaries, chunks=None):
        return 6.0

    def fake_evaluate_quality(quant, quantized_gguf, baseline_ppl, wikitext_path, binaries, chunks=None):
        loss = quality_map[quant]
        return QualityResult(
            candidate_quant=quant, perplexity=baseline_ppl * (1 + loss / 100),
            baseline_perplexity=baseline_ppl, quality_loss_pct=loss,
        )

    def fake_list_supported_quant_types(binaries):
        return ["Q8_0", "Q6_K", "Q4_K_M"]

    def fake_build_run_command(gguf_path, config, binaries):
        return ["llama-cli", "-m", str(gguf_path), "-ngl", str(config.ngl), "-c", str(config.ctx)]

    def fake_model_fingerprint(path):
        return "fake-fp"

    bench_calls = {"n": 0}

    def make_fake_run_bench(speed_fn):
        def fake_run_bench(gguf_path, ngl, ctx, target, binaries, timeout_sec=300):
            bench_calls["n"] += 1
            quant = gguf_path.stem.split("-", 1)[1]
            speed = speed_fn(ngl, ctx)
            return BenchResult(
                candidate=CandidateConfig(quant=quant, ngl=ngl, ctx=ctx),
                prompt_tok_per_sec=speed * 2, gen_tok_per_sec=speed,
                vram_used_mb=None, raw_stdout="fake",
            )
        return fake_run_bench

    saved = (
        _mod.quantize, _mod.run_bench, _mod.evaluate_quality, _mod.compute_perplexity,
        _mod.list_supported_quant_types, _mod.build_run_command, _mod.model_fingerprint,
    )
    try:
        _mod.quantize = fake_quantize
        _mod.compute_perplexity = fake_compute_perplexity
        _mod.evaluate_quality = fake_evaluate_quality
        _mod.list_supported_quant_types = fake_list_supported_quant_types
        _mod.build_run_command = fake_build_run_command
        _mod.model_fingerprint = fake_model_fingerprint

        # --- 1. Early-exit hit: speed(ngl) = ngl, target=20 -> minimal ngl=20,
        #        highest-quality passing quant (Q8_0) wins, GPU binary search path.
        _mod.run_bench = make_fake_run_bench(lambda ngl, ctx: float(ngl))
        hw_gpu = HardwareProfile(gpu_vendor=GPUVendor.NVIDIA, gpu_name="RTX4090",
                                  vram_mb=24000, cpu_cores=16, ram_mb=65536, os_name="linux")
        target = TargetSpec(
            model_path=Path("model.gguf"), target_tokens_per_sec=20.0,
            max_quality_loss_pct=5.0, ctx=4096, ctx_candidates=(4096,),
            quant_candidates=("Q8_0", "Q6_K", "Q4_K_M"), ngl_max_calls=6,
        )
        bench_calls["n"] = 0
        result = search(target, model_info_stub, hw_gpu, binaries_stub, work_dir, wikitext)
        assert result.meets_target is True
        assert result.config.quant == "Q8_0", result.config.quant  # highest quality, first tried
        assert result.config.ngl == 20, result.config.ngl  # minimal ngl meeting target
        assert result.bench.gen_tok_per_sec >= 20.0
        assert bench_calls["n"] <= 2 + target.ngl_max_calls  # top + low + <=6 binary-search calls

        # --- 2. Quality gate skip (early-exit A): tighten max_quality_loss_pct
        #        so Q8_0/Q6_K fail quality (1.0%/3.0% > 0.5%) and are never
        #        benchmarked; only Q4_K_M (10% loss, also > 0.5%) remains, so
        #        it too is skipped without a single bench call -> NoFeasibleConfigError
        #        with closest=None (nothing was ever benchmarked).
        strict_target = replace(target, max_quality_loss_pct=0.5)
        bench_calls["n"] = 0
        try:
            search(strict_target, model_info_stub, hw_gpu, binaries_stub, work_dir, wikitext)
            raise AssertionError("expected NoFeasibleConfigError")
        except NoFeasibleConfigError as e:
            assert e.closest is None
        assert bench_calls["n"] == 0  # quality gate blocked every quant before any bench

        # --- 3. CPU-only hardware: ngl search skipped, top bench reused at ngl=0. ---
        hw_cpu = HardwareProfile(gpu_vendor=GPUVendor.NONE, gpu_name=None, vram_mb=None,
                                  cpu_cores=8, ram_mb=32768, os_name="linux")
        _mod.run_bench = make_fake_run_bench(lambda ngl, ctx: 50.0)  # flat, always meets target
        bench_calls["n"] = 0
        result_cpu = search(target, model_info_stub, hw_cpu, binaries_stub, work_dir, wikitext)
        assert result_cpu.meets_target is True
        assert result_cpu.config.ngl == 0
        assert result_cpu.config.quant == "Q8_0"
        assert bench_calls["n"] == 1  # only the full-offload bench, relabeled -- no extra call

        # --- 4. Full failure: every quant's full-offload speed is below target
        #        -> early-exit B for all, loop exhausts -> NoFeasibleConfigError
        #        with closest = the fastest attempt actually seen.
        _mod.run_bench = make_fake_run_bench(lambda ngl, ctx: 5.0)  # always too slow
        try:
            search(target, model_info_stub, hw_gpu, binaries_stub, work_dir, wikitext)
            raise AssertionError("expected NoFeasibleConfigError")
        except NoFeasibleConfigError as e:
            assert e.closest is not None
            assert e.closest.meets_target is False
            assert e.closest.bench.gen_tok_per_sec == 5.0

        # --- 5. Multi-ctx verification: candidate meets target at ctx=4096 but
        #        not at ctx=8192 -> that quant is rejected, falls through to a
        #        lower-quality quant that meets target at both ctx values.
        multi_ctx_target = replace(target, ctx_candidates=(4096, 8192))

        def speed_multi_ctx(ngl, ctx):
            if ctx == 8192:
                return 0.0  # Q8_0/Q6_K/Q4_K_M all "fail" at the larger ctx here
            return float(ngl)

        _mod.run_bench = make_fake_run_bench(speed_multi_ctx)
        try:
            search(multi_ctx_target, model_info_stub, hw_gpu, binaries_stub, work_dir, wikitext)
            raise AssertionError("expected NoFeasibleConfigError (no quant survives ctx=8192)")
        except NoFeasibleConfigError as e:
            assert e.closest is not None  # best attempt at ctx=4096 was still recorded

        # --- 6. Larger ctx needs more offload: speed(ngl, 8192) = ngl - 5.
        #        Minimal ngl for ctx=4096 alone is 20 (fails 8192 at 15);
        #        the answer must be ngl=25, which meets both.
        _mod.run_bench = make_fake_run_bench(lambda ngl, ctx: float(ngl - (5 if ctx == 8192 else 0)))
        result_mc = search(multi_ctx_target, model_info_stub, hw_gpu, binaries_stub, work_dir, wikitext)
        assert (result_mc.config.quant, result_mc.config.ngl) == ("Q8_0", 25), result_mc.config
    finally:
        (
            _mod.quantize, _mod.run_bench, _mod.evaluate_quality, _mod.compute_perplexity,
            _mod.list_supported_quant_types, _mod.build_run_command, _mod.model_fingerprint,
        ) = saved

    print("fituna.search self-check OK")


if __name__ == "__main__":
    _self_check()
