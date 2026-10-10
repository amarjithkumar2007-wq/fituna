[Korean original](./DEVELOPMENT.md)

# Development and Validation

This document explains how FiTuna is developed and validated in practice. The
practices described here are already followed in the repository and can be checked
against the code, CI configuration, and pull request history. Plans that have not
been carried out are marked (planned).

For *what the code does*, see [ARCHITECTURE.md](ARCHITECTURE.md). For the development
environment and how to contribute, see [CONTRIBUTING.md](../CONTRIBUTING.md). For
release history, see [CHANGELOG.md](../CHANGELOG.md).

## 1. Contract-First Design

Every data structure that crosses a module boundary is defined as an immutable
dataclass or enum in [`fituna/config.py`](../fituna/config.py): `HardwareProfile`,
`TargetSpec`, `BinaryPaths`, `ModelInfo`, `CandidateConfig`, `BenchResult`,
`QualityResult`, `SearchResult`, `DoctorCheck`, and `CorpusPreset`. This module is
the single source of truth for the interface. The surrounding modules (`hardware`,
`binaries`, `quantize`, `quality`, `bench`, `search`, `cache`, and `report`) exchange
only these types.

`frozen=True` is intentional. Results that have been measured, cached, and reported
must not change later in the pipeline. The `config.py` self-check verifies that
attempting to modify a `HardwareProfile` or `SearchResult` raises an exception
rather than silently succeeding. Immutability is a tested property, not a
convention.

As stated in `CONTRIBUTING.md`, a change to a data structure shared between modules
must update the dataclass in `config.py` and all its consumers in the same pull
request.

## 2. Per-Module Self-Checks

Each of the 17 modules in `fituna/` can run on its own and checks its main invariants
with assertions. This count includes all 18 `.py` files except the three-line
`__main__.py` shim.

```bash
python -m fituna.config          # 불변 dataclass 계약
python -m fituna.search          # 탐색 순서와 조기 종료 논리
python -m fituna.cli --selfcheck # entry point는 명시적 flag 사용
```

For `cli` and `mcp_server`, whose `__main__` is an actual entry point, checks run
only when the explicit `--selfcheck` flag is present. Therefore,
`python -m fituna.cli` still runs the CLI. The other modules, including `doctor`
and `corpus`, run their checks under `if __name__ == "__main__":` regardless of
arguments. CI passes `--selfcheck` to `doctor` and `corpus` as well. This does not
change their behavior; it keeps the invocation format consistent across the four
modules that accept arguments. Self-checks are assertions placed beside the code
they test, not a separate test framework. They let a single module be checked in
isolation even without pytest installed.

These checks are neither optional nor informational.
[CI](../.github/workflows/ci.yml) runs the test suite and all 17 self-checks as
required steps on every OS and Python version in the matrix. A failure in any
self-check fails the build.

## 3. Unit Tests with Mocked Subprocesses

The suite in `tests/` covers the package and is designed to pass **on a computer
without llama.cpp installed or network access**. All external effects are
monkeypatched at their boundaries.

- Hardware detection commands (`nvidia-smi`, `rocm-smi`, `system_profiler`) are
  tested by replacing `subprocess.run` itself. Recorded output drives the parsers,
  including cases where the executable cannot be found.
- Tests for `fituna fetch-corpus` replace `urllib.request.urlopen`; tests for the
  atomic write path replace `tempfile.mkstemp`.
- Orchestration tests replace wrapper functions. `search` gets fake `quantize`,
  `bench`, and `quality` functions; `doctor` gets fake `binaries.find_exe`,
  `get_llama_cpp_version`, `detect_hardware`, `shutil.disk_usage`, and `os.access`.

Actual tool output is kept as fixtures.
`tests/fixtures/llama_bench_sample.json` is read by both `_self_check()` in
`fituna/bench.py` and `tests/test_bench.py` (#46).
`tests/fixtures/llama_perplexity_kld_sample.txt`, captured from llama.cpp b11342,
is read by `tests/test_quality.py` (#45). Thus, `pytest -q` tests these two parsers
against text that llama.cpp actually produced, rather than invented strings.
Capture new fixtures from real runs, and record in the test which llama.cpp build
produced them.

This choice has a clear cost. Within the suite, llama.cpp is a mock that behaves
as expected, so **the suite cannot catch communication errors between FiTuna and
real llama.cpp.** The flag and protocol bugs documented in the 0.1.0 changelog
all passed the full suite: `llama-bench` had no `-c` flag, rejected `--version`, and
CPU-only benchmarks did not finish. The KLD interpretation in #45 was another such
case. The mock output used a format that llama.cpp did not actually produce, but
the suite and CI both passed. The problem was found only by running real llama.cpp
before merging. Actual output was then captured as a fixture and the test was
changed to read it. Section 6 describes the procedure that covers this gap and
explains why it is a separate step rather than more tests in the suite.

The suite works well above the subprocess boundary. It checks search order and
early stopping, cache keys and schema migration, hardware output parsing from
recorded `nvidia-smi`, `rocm-smi`, and `system_profiler` text (`test_hardware.py`),
hardware detection fallbacks, error mapping and exit codes, and OS-specific path
handling. `test_bench.py` and `test_quality.py` check `llama-bench` and
`llama-perplexity` output parsing with mocked subprocesses and the fixtures above.
In contrast, `test_binaries.py` uses synthetic output and mocked subprocesses to
check versions, supported quantization types, fallbacks, and error handling. No
actual llama.cpp executable is needed. `quantize.py` is not yet exercised directly
by pytest ([#10](https://github.com/leeyunseokarchive/fituna/issues/10)).
`test_report.py` tests only the pure parts of `report.py`: command generation,
Modelfile export, and rendering, which do not start subprocesses in the first place.
`test_cli.py` replaces `search.search()` and `binaries.locate_binaries()` with fakes,
then drives `cli.main()` end to end. This catches wiring errors, such as those
between `--export-ollama` and `report.export_ollama_modelfile()`, that argparse-only
tests miss. Among the subprocess wrappers, `model_info.py` is the exception:
`test_config.py` checks the `is_already_quantized` guard but does not cover GGUF
header parsing. The remaining parser (`quantize.py`) is tested only by the
per-module self-checks in Section 2.

## 4. CI Matrix: 3 OSes × 2 Python Versions

[`.github/workflows/ci.yml`](../.github/workflows/ci.yml) runs on every pull request
and push to `main`.

| | Python 3.11 | Python 3.13 |
|---|---|---|
| ubuntu-latest | ✅ | ✅ |
| macos-latest | ✅ | ✅ |
| windows-latest | ✅ | ✅ |

There are 6 jobs. With `fail-fast: false`, a failure on one platform does not hide
the results for the others. Each job installs the package with `pip install -e .`
and installs pytest, then runs `pytest -q` and all 17 module self-checks. Importing
something FiTuna has not declared causes the installation step to fail, so this
installation also checks the zero-runtime-dependency constraint.

3.11 is the minimum version declared in `pyproject.toml`, and 3.13 is the current
release. Windows is included in the matrix because FiTuna handles paths and tool
output, both of which differ on Windows. The first actual Windows CI run caught a
test failure caused by `\U` in a Windows path being treated as an invalid regular
expression escape.

Linters, formatters, type checkers, and a coverage gate are deliberately absent
from CI. The package includes `py.typed` and uses type annotations, but there is
currently no step to check them **(planned)**.

### Releases

`.github/workflows/release.yml` handles PyPI distribution. When a release tagged
`vX.Y.Z` is published on GitHub, the workflow checks that the tag matches
`fituna.__version__`, runs the tests, builds an sdist and wheel, and uploads them
to PyPI. Authentication uses PyPI Trusted Publishing (OIDC), so no API token is
stored anywhere. The deployment job runs only in the `pypi` environment, which
permits deployment only from `v*` tags.

Procedure: merge a PR that bumps the versions in `pyproject.toml` and
`fituna/__init__.py` and moves the [Unreleased] entries in CHANGELOG under the new
version, then create a `vX.Y.Z` release at that commit.

## 5. Branch → Pull Request → Review → Merge

Work happens on topic branches (`feat/…`, `fix/…`, `docs/…`, `chore/…`), not on
`main`. Each branch becomes a pull request, and `main` changes only through merge
commits. PRs #1–#6 were merged this way.

FiTuna is a one-person project, so review here means a written self-review
submitted as a formal pull request review before merging. Changes made as a
result of review are pushed as separate commits to the same branch. The aim is
to record in the review body what was verified rather than assumed, and to leave
a history that lets readers trace follow-up commits. Examples from the actual
history:

- PR #1 (`fituna doctor`) → `fix: address doctor review findings (Windows tests,
  dual binary path, naming)` and `fix: close doctor's never-raises gaps`
- PR #2 (`fetch-corpus`) → `fix: address fetch-corpus review findings
  (os.replace guard, rows<=0, docs)` and a final cleanup pass
- PR #6 (Run 5) → three rounds of review successively found overstated conclusions
  in earlier reviews, leading to the final **withdrawal** of the published
  measurement conclusions

The last example shows the review standard. A review that merely confirms the
change is not useful. PR #6 withdrew its own main result when chunk-level traces
did not support the evidence, and PR #4 corrected an incorrect model-license claim
in already-published documentation. Problems too large to fix within the branch
are not quietly dropped: they are recorded in the review and issue history. A
current example is `quality.py` discarding the `±` standard error from
`llama-perplexity` output. Storing it would invalidate the cache used by every
publicly reported figure. This needs an explicit migration of measurement data,
not a fix slipped into unrelated work.

Measured figures in the documentation must be traceable down to a log line or a
cache row. Claims without traceable evidence are deleted, not softened.

## 6. End-to-End Validation on Real Hardware

The test suite deliberately cannot run llama.cpp (Section 3), so changes at the
subprocess boundary are validated separately with real binaries and models.
GitHub runners have neither a GPU nor a llama.cpp build, so this step is manual
and is not included in CI.

A validation run includes:

1. Run `fituna doctor` against a real llama.cpp build, and check the detected build
   version with `fituna list-binaries`.
2. Complete a full `fituna run` using actual quantize, bench, and perplexity
   subprocesses, reaching a successful result (exit code 0).
3. Check the `--resume` cache-hit path on a second run.
4. Check the failure paths for a missing binary (exit code 2) and for reporting
   the best result when no feasible configuration exists (exit code 3).

Run logs, timings, and hardware details are recorded in [RESULTS.md](RESULTS.md).
This includes variation between runs and thermal-throttling outliers retained
rather than discarded, so readers can check the figures instead of taking them
on trust. Actual platform validation currently covers macOS (Apple Silicon/Metal)
and Linux (NVIDIA Tesla T4/CUDA). Anyone can reproduce the Linux results using the
[Colab notebook](../notebooks/colab_nvidia_verification.ipynb). Windows passes unit
tests and CI but has not been validated with real binaries. This is stated in the
README's known limitations rather than overstating support.

[REVIEWERS.md](../REVIEWERS.md) adapts this procedure for third-party reviewers
reproducing the public results.

## 7. Dependencies and Licensing

Zero runtime dependencies is a required constraint, not a preference. The only
development dependency is `pytest`. Even the MCP server implements stdio-based
JSON-RPC directly instead of using an SDK, and corpus downloads use `urllib`
rather than `datasets`. The `pip install -e .` step in CI checks dependency
declarations every time.

Licenses are tracked, not guessed. The project includes an SPDX header in every
Python file, `REUSE.toml`, and compliance records covering the tools FiTuna calls
and the corpora and models used for measurements. See:

[LICENSE_COMPLIANCE.md](LICENSE_COMPLIANCE.md),
[SBOM.md](SBOM.md),
[OPEN_SOURCE_USAGE.md](OPEN_SOURCE_USAGE.md),
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
AI-assisted development is disclosed in [AI_MODEL_USAGE.md](AI_MODEL_USAGE.md).
