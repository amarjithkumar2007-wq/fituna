[한국어 원문](./ARCHITECTURE.md)

# FiTuna Architecture

## Overview

FiTuna is a Python 3.11 CLI that orchestrates llama.cpp binaries as subprocesses.

It finds GGUF quantization and execution settings (`quant`, `-ngl`, `-c`) that satisfy the target throughput while staying within the quality-loss budget specified by the user. `quant` candidates are checked in order of quality, and the first one that passes is selected. `-ngl` is the minimum value that allows that quantization to meet the target.

FiTuna itself does not perform tensor operations. Inference, quantization, and perplexity calculation are all handled by llama.cpp C++ binaries, while FiTuna is responsible for execution orchestration, output parsing, search, and caching.

FiTuna has zero Python runtime dependencies and uses only the standard library.

## Pipeline Overview

``````mermaid
flowchart LR
    subgraph Input
        A["model.gguf<br/>(or HF directory)"]
        B["Target tok/s<br/>Quality budget"]
    end

    A --> C[hardware.py<br/>GPU / VRAM / RAM<br/>Auto-detection]
    B --> D

    subgraph "Step 1 · Quality (all candidates)"
        D[quantize.py<br/>llama-quantize] --> E[quality.py<br/>llama-perplexity<br/>Loss vs F16 baseline]
        E --> F{"Loss ≤ budget?"}
        F -- No --> X[Rejected]
    end

    subgraph "Step 2 · Speed (early-termination traversal)"
        F -- Yes, sorted by measured quality --> G[bench.py<br/>llama-bench full-offload]
        G -- Below target --> Y[Skip quant]
        G -- Target achieved --> H["Binary search<br/>minimum -ngl"]
    end

    C --> G

    H --> I[["Result:<br/>quant + ngl + ctx<br/>+ execution command"]]

    E & G <--> K[(cache.py<br/>sqlite3<br/>--resume)]

**Step 1** measures the perplexity loss of *all* candidates. Since Step 2 explores candidates in order of **measured** quality, they cannot be sorted using unmeasured values. In practice, both models tested had a conventional Q8_0-first ordering that turned out to be incorrect. **Step 2** actively performs early termination. A quant that misses the target during the full-offload benchmark is discarded without additional measurements, and the first quant that passes is selected. Quants with lower quality than this are not measured.

All subprocess results are stored in an sqlite3 cache keyed by the model fingerprint, hardware profile, and **llama.cpp build version**. Therefore, `--resume` does not return values measured using a different backend build.

## Repository Structure

`````text
fituna/
├── cli.py         # argparse entry point, exit code mapping (0/1/2/3)
├── quickstart.py  # Interactive wizard for assembling run flags (fituna quickstart)
├── config.py      # Immutable dataclass interface contract (single source of truth)
├── hardware.py    # Automatic GPU/VRAM/CPU/RAM detection + manual override
├── binaries.py    # llama.cpp binary discovery + capability verification
├── doctor.py      # Environment self-diagnostics (fituna doctor subcommand)
├── corpus.py      # Quality corpus download (fituna fetch-corpus, stdlib urllib)
├── errors.py      # Re-export shim for the FiTunaError hierarchy defined in config.py
├── mcp_server.py  # MCP stdio server (JSON-RPC 2.0, fituna-mcp entry point)
├── model_info.py  # Direct GGUF header parsing (struct), HF directory conversion
├── quantize.py    # llama-quantize wrapper (idempotent, atomic writes)
├── quality.py     # llama-perplexity wrapper (quality-loss measurement)
├── bench.py       # llama-bench wrapper (throughput measurement)
├── search.py      # Step 2 search orchestrator
├── cache.py       # sqlite3 result cache (--resume)
└── report.py      # General/JSON result rendering + execution command generation

## Module Relationships

````text
                              ┌───────────┐
                              │  cli.py   │  argparse entry point (run /
                              └─────┬─────┘  quickstart / detect-hw /
                                    │         list-binaries / doctor /
                                    │         fetch-corpus / help);
                                    │         TargetSpec creation & task distribution
        ┌───────────────┬──────────┼───────────┬──────────────────┐
        ▼                ▼          ▼           ▼                  ▼
  hardware.py      binaries.py  model_info.py  quantize.py    report.py
  GPU/CPU/RAM      llama.cpp    GGUF-based     llama-quantize  SearchResult
  detection or     binary       ModelInfo      wrapper,        -> execution
  manual input     discovery    reading        disk reuse      command,
  parsing          & validation (n_layers, etc.)               JSON/general
                                                                report
        │                │          │                │              ▲
        │                │          │                │              │
        └────────┬───────┴────┬─────┴──────┬─────────┘              │
                  ▼            ▼            ▼                        │
           ┌─────────────────────────────────────┐                  │
           │              search.py               │  orchestrator   │
           │  Quality-first filter (quality.py)  │──────────────────┘
           │  + ngl binary search (bench.py)     │
           │  + ctx grid inspection (bench.py)   │
           └───────────┬───────────────┬──────────┘
                        ▼               ▼
                  bench.py         quality.py
                  llama-bench      llama-perplexity
                  wrapper          wrapper
                        │               │
                        └───────┬───────┘
                                ▼
                          cache.py (sqlite3)
                          Reuse BenchResult / QualityResult
                          key: (model_fp, hw_fp, candidate)

           fituna/config.py — Immutable dataclass/Enum/exceptions
           used by all modules above
           (HardwareProfile, TargetSpec, BinaryPaths,
           ModelInfo, CandidateConfig, BenchResult, QualityResult,
           SearchResult, DoctorCheck, CorpusPreset, FiTunaError hierarchy).
           Other modules do not define inter-module types separately;
           they import them from here.

The arrows indicate the direction of calls, not imports. `search.py` calls
`quantize.py`, `bench.py`, `quality.py`, and `cache.py`, but these modules do
not call `search.py` back. `cli.py` imports and calls the largest number of
modules. The modules below it depend only on `config.py` and, when necessary,
`BinaryPaths` from `binaries.py`. The entry points above `cli.py`,
`quickstart.py` and `mcp_server.py`, also access multiple modules, but when
running a search they go through `cli.py`, so there is only one implementation
of the search path.

`fituna/quickstart.py` is positioned *above* `cli.py`, rather than beside it.
The `fituna quickstart` wizard collects answers using `input()`, assembles and
prints `fituna run ...` argv, then parses the same argv again using
`cli._build_parser()` and calls `cli._cmd_run()` within the same process.
Step 1 uses `doctor.run_checks()`, memory-fit calculations use
`hardware.detect_hardware()`, Step 5 uses `corpus.fetch_corpus()`, and all file
sizes use `report.human_size()`. It only orchestrates existing modules and
does not duplicate their logic.

All *search parameters* assembled by the wizard correspond to public `run`
flags. The self-check (`python -m fituna.quickstart --selfcheck`) verifies the
completed argv using assertions, so it fails if wizard-specific search
settings are introduced. This check only covers run argv. Wizard-only
convenience features without corresponding `run` functionality, such as
curated model downloads, HuggingFace search, and corpus downloading, are
outside its scope.

Model downloading is implemented using the same temporary-file + `os.replace`
atomic pattern as `corpus.py`, using the standard-library `urllib`. The
response structure of the HuggingFace `/api/models` search endpoint is
documented in the module docstring and has been verified against the actual
API.

`fituna/mcp_server.py` is a thinner second entry point placed alongside
`cli.py`. It is an MCP stdio server that uses only the standard library without
an SDK dependency and processes line-based JSON-RPC 2.0. It provides the
`fituna_detect_hardware` and `fituna_recommend` tools, allowing an AI agent to
request measured configurations in the same way as `cli.py run`.

It directly calls `binaries.py`, `hardware.py`, `model_info.py`, `search.py`,
`report.py`, and `cache.py` without executing `cli.py` through a shell. It
always enables `cache.ResultCache`, so repeated requests for the same
model and hardware can be answered in approximately one second.

## Runtime Data Flow (`fituna run` once)

1. `cli.py` parses the argv as CLI arguments and assembles a `TargetSpec`.
   `--model`, `--target-tps`, `--max-quality-loss`, and `--ctx` become
   `ctx_candidates`, with the first value used as `.ctx`, while `--quant`
   becomes `quant_candidates`, re-sorted in descending order of quality.

2. `hardware.detect_hardware()` runs the available `nvidia-smi`, `rocm-smi`,
   `system_profiler`, and `platform` commands. When `--gpu` or `--vram-mb` is
   provided, `parse_manual_hardware()` merges the user-provided values with the
   automatically detected results, with user values taking precedence.
   The result is a `HardwareProfile`.

3. `binaries.locate_binaries(bin_dir=...)` searches `PATH` or
   `--llama-bin-dir` for `llama-quantize`, `llama-bench`, and
   `llama-perplexity`, and also checks for the optional
   `llama-imatrix` and `convert_hf_to_gguf.py`. If a required tool is missing,
   it raises a `BinaryNotFoundError` containing installation guidance.
   `list_supported_quant_types()` parses `llama-quantize --help` and filters
   `TargetSpec.quant_candidates` so that only formats actually supported by
   the installed build remain.

4. If `model_path` is not a `.gguf` file, `model_info.ensure_base_gguf()`
   converts the HF directory to `work_dir/base-f16.gguf` using
   `binaries.convert_script`. If conversion fails, it raises a
   `ModelConversionError`. Then `read_model_info()` reads the architecture,
   number of layers, and number of parameters to create a `ModelInfo`.
   Since llama.cpp counts the output layer as one additional layer, the upper
   bound for `-ngl` search is `n_layers + 1` (or 0 when no GPU is available).

5. `search.search()` orchestrates the algorithm described below.
   Through `binaries.BinaryPaths`, it calls `quantize.quantize()`,
   `bench.run_bench()`, and `quality.evaluate_quality()`. When `--resume` is
   specified, all bench and quality calls are stored in `cache.ResultCache`.
   It returns a `SearchResult` containing the actual solution
   (`meets_target=True`) or the best result. If there is no reasonably close
   result, it raises a `NoFeasibleConfigError`.

6. `report.py` converts the `SearchResult` into three ways to use the
   already-generated GGUF. `build_server_command()` creates a
   `llama-server` command for an OpenAI-compatible local API,
   `export_ollama_modelfile()` writes an Ollama `Modelfile` next to the `.gguf`
   atomically when `--export-ollama` is specified, and `build_run_command()`
   creates a `llama-cli` command for interactive testing.
   `to_human()` first displays the output paths and sizes, then lists these
   three usage methods in this order. `to_json()` adds
   `llama_server_command` and `modelfile_path` alongside the existing fields.
   `cli.py` outputs one of the two formats to stdout depending on `--json`.
   `llama-cli` and `llama-server` are *only located and are not executed.*

## Search Algorithm (inside `search.search()`)

This is a two-stage grid search that limits the number of bench calls to
`O(quant × log(n_layers))`. Since perplexity depends only on `quant`, not on
`ngl` or `ctx`, quality and speed can be separated. Quality is calculated only
once per quant and is not measured again while searching for speed.

```text
Step 1 — Quality pre-filter (one llama-perplexity call per quant)
  baseline_ppl = compute_perplexity(base F16 GGUF)      [cache, calculated only once]
  for quant in quant_candidates ∩ list_supported_quant_types():
      gguf = quantize(base_gguf, quant)
      q = evaluate_quality(quant, gguf, baseline_ppl, wikitext_path)
          # --quality-metric kld: generate F16 reference logits once (reuse the file),
          # then read both KLD and PPL(Q) from the same call. KLD is for reporting,
          # while quality_loss_pct is calculated from PPL(Q) (#49). If PPL(Q) is
          # unavailable, raise FiTunaError.
      q.quality_loss_pct <= max_quality_loss_pct → keep the quant
  quality_filtered = sort the passing quants in their original quality order
                      (Q8_0 → Q2_K), i.e. highest quality first

Step 2 — Speed search per quant (highest quality first, first passing quant wins)
  for quant in quality_filtered:
      gguf = quantize(base_gguf, quant)                  # idempotent, reuse Step 1 result
      max_ngl = 0 if hw.gpu_vendor == NONE else n_layers + 1   # includes output layer
      top = run_bench(gguf, ngl=max_ngl, ctx=target.ctx)
      if top.gen_tok_per_sec < target_tps:
          continue                        # Early termination B: move to next lower-quality quant
      # ok(ngl) = all remaining ctx_candidates achieve at least target_tps at this ngl
      if hw.gpu_vendor == NONE:
          if ok(0): return result(quant, ngl=0, top)       # CPU-only, use the measurement at ngl=0 as-is
          continue
      if not ok(max_ngl):
          continue                        # even full offload cannot meet the target for other ctx values
      low = run_bench(gguf, ngl=0, ctx=target.ctx)
      if low.gen_tok_per_sec >= target_tps and ok(0):
          return result(quant, ngl=0, low)                 # Early termination C: GPU not needed
      # Binary-search for the minimum ngl in [0, max_ngl] that satisfies target_tps for all ctx values
      # assuming gen_tok_per_sec does not decrease as ngl increases. In the worst case,
      # fall back to `top`, which has already been confirmed to meet the target.
      lo, hi, best, calls = 0, max_ngl, top, 0
      while lo < hi and calls < target.ngl_max_calls:
          mid = (lo + hi) // 2
          r = run_bench(gguf, ngl=mid, ctx=target.ctx); calls += 1
          if r.gen_tok_per_sec >= target_tps and ok(mid): best, hi = r, mid
          else:                                           lo = mid + 1
      return result(quant, ngl=best.candidate.ngl, best)    # select the first quant that reaches this point
  raise NoFeasibleConfigError(closest=fastest attempt seen)  # all quants failed Early termination B

Early termination happens in three places. **A** — a quant that does not pass the quality gate
never moves on to the speed benchmark. **B** — a quant that misses the target even at full GPU
offload is discarded immediately, and the next lower-quality candidate, which is usually faster,
is tried. **C** — if the target is met at `ngl=0`, the binary search is skipped and the
minimum-resource configuration is returned. Because quants are tried in descending order of
quality and the search returns at the first success, FiTuna always reports the *highest-quality*
configuration among the feasible ones. It never picks a lower-quality candidate just because it
was measured later. If `max_bench_seconds` elapses during the search, it returns the best result
found so far with `meets_target=False` instead of raising an exception.

The upper bound on `llama-bench` calls is `N_quant_survived × (2 + ngl_max_calls) ×
len(ctx_candidates)`. With the `TargetSpec` defaults in `fituna/config.py`, the worst case is
48 calls (6 quants, `ngl_max_calls=6`, 1 ctx candidate: 6 × (2 + 6) × 1). In practice,
early termination usually finishes before 10 calls.

## Filesystem Artifacts

All side effects occur only under `--out` (`work_dir`) or in the llama.cpp binaries. The
remaining functions are pure transformations that take `config.py` dataclasses as input and
return values.

```
<work_dir>/
├── base-f16.gguf            # model_info.ensure_base_gguf() — only for HF directory input
├── <model>-<fp12>-<quant>.gguf  # quantize.quantize() — one per attempted quant, reused if present
├── <base>.<key12>.kld       # quality.generate_base_logits() — only with --quality-metric kld, temp→replace
│                            #   key = sha256(model_fp:corpus:ppl_chunks), a new file is generated if it changes
├── Modelfile                # report.export_ollama_modelfile() — only with --export-ollama, atomic
└── .fituna_cache.sqlite3    # cache.ResultCache — bench_cache / quality_cache, only with --resume
````

`quantize()` and `ensure_base_gguf()` do not recreate a file if one already exists at the
target path. Re-running `fituna run` with the same `--out` is therefore cheap even without
`--resume`. `model_info.model_fingerprint()` is not a full-file hash but a cheap
`sha256(name:size:mtime)`, and is the cache key component that identifies "this model".
Because it is used together with the hardware fingerprint, cache entries from other models or
other computers do not get mixed in. The first 12 hex digits of the same fingerprint also go
into the quantized `.gguf` filename `<model>-<fp12>-<quant>.gguf`. Without it, two different
models converted to the same conventional base filename would collide: every HF directory
input becomes `base-f16.gguf`, so within the same `--out` one model's quantized file could
silently be served as a "cache hit" for a completely different model.

## Error Handling and Exit Codes

`cli.py` maps the `FiTunaError` hierarchy, defined once in `config.py` and re-exposed in
`errors.py`, to process exit codes.

| Exit code | Condition |
|---|---|
| 0 | Success — `search()` returned a `SearchResult` with `meets_target=True` |
| 1 | General error — any other `FiTunaError`, or `meets_target=False` where no configuration met the target and the best result was returned |
| 2 | `BinaryNotFoundError` — a required llama.cpp binary is missing; the message includes installation guidance. **Also** argparse's usage-error code. Since `parser.parse_args()` calls `sys.exit(2)` directly, a missing required flag or an unknown flag exits before `main()`'s `FiTunaError` mapping. If the first line of stderr is `usage: ...`, it is argparse; if it is a `... ERROR fituna: ...` log, it is `BinaryNotFoundError` |
| 3 | `NoFeasibleConfigError` — every quant candidate was eliminated by the quality gate or the full-offload speed check. The closest attempt for diagnostics is stored in `.closest` |

`ModelConversionError`, raised when the HF→GGUF conversion subprocess fails, and the
`FiTunaError` raised on a `llama-bench` or `llama-perplexity` timeout both give exit code 1.

`fituna doctor` does not go through this exception mapping. `_cmd_doctor` computes the exit
code directly from the check results via `doctor.exit_code()`, so it bypasses `main()`'s
`FiTunaError` handling entirely. There is no exception to catch. It reuses the same 0, 1, 2
for related but different conditions: 0 if every check is PASS or WARN, 2 if any of the three
required llama.cpp binaries is FAIL (matching the `BinaryNotFoundError` → 2 convention above),
and 1 for any other FAIL. Doctor has no value corresponding to exit code 3.
`NoFeasibleConfigError` only arises in `run`, and doctor never produces it.

<a id="why-this-shape"></a>

## Why This Shape

- **Single source of truth for types** (`fituna/config.py`) — every value passed between
  modules is a `frozen` dataclass or `Enum` defined exactly once. Even when modules are
  developed separately, their interfaces do not drift apart.
- **Pure functions and explicit side effects** — the only modules that touch the filesystem
  are `quantize.py` (writes `.gguf`), `model_info.py` (writes the converted base `.gguf`),
  `cache.py` (writes sqlite3), `corpus.py` (writes the downloaded corpus), `report.py` (writes
  the Ollama `Modelfile` with `--export-ollama`), and `quickstart.py` (writes downloaded
  `.gguf` files). Everything else returns values. All three download paths use the temporary
  file + `os.replace` pattern, so an interrupted run never leaves an incomplete file behind.
- **Subprocess isolation** — every interaction with a llama.cpp binary goes through exactly
  one wrapper function per binary (`quantize()`, `run_bench()`, `compute_perplexity()`). The
  logic for interpreting each binary's output lives in one place only.
- **Separating quality from speed** — perplexity is independent of `ngl` and `ctx`, so
  `search.py` computes it once per quant rather than once per candidate configuration. This
  reduces the number of benchmark calls from `O(quant × ngl × ctx)` to
  `O(quant × log(n_layers))`.
- **The cache is an optimization, not a dependency** — every module that calls `cache.py`
  works correctly when `cache is None`, that is, when `--resume` is not given, by running the
  subprocess directly. The cache is not required for correctness; it only reduces duplicate
  calls between runs.
- **Recommend, but do not operate a server** — this is an intended scope boundary. FiTuna
  outputs a quantized `.gguf` and the `llama-server` / `llama-cli` commands that the user
  copies and runs. With `--export-ollama`, it also writes an Ollama `Modelfile` next to it. It
  does not execute these or start its own inference server. This is a boundary, not an
  omission. Actually serving inference is the job of llama.cpp, Ollama, and LM Studio, and
  duplicating that would put FiTuna in competition with the very tools the README compares
  against, with nothing to differentiate it. What FiTuna offers is only that it *measures*
  the search rather than guessing. A server process would also conflict with the
  zero-runtime-dependency design. After a recommendation, the Ollama path is already
  provided: `--export-ollama` writes the measured `num_gpu` and `num_ctx` into the Modelfile.
  Otherwise, both Ollama and LM Studio apply fixed per-model presets, which is [exactly the
  difference](https://github.com/ollama/ollama/issues/14674) the README cites. Extension
  candidates that stay within this boundary include running the winning command directly
  (`--launch`) and LM Studio preset export, but neither is in the current release. The MCP
  server already covers the agent path. An agent reads the `fituna_recommend` response and
  decides the next action itself, so no human needs to copy a command.
`````