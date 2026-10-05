# SPDX-License-Identifier: MIT
"""fituna.cli
=============

argparse-based CLI. Subcommands: ``run``, ``quickstart``, ``detect-hw``,
``list-binaries``, ``doctor``, ``fetch-corpus``, ``help``.

``quickstart`` is an interactive shell over ``run``'s own flags (see
fituna/quickstart.py): it assembles a ``run`` argv, prints it, parses it back
through this module's ``_build_parser()`` and calls ``_cmd_run`` in-process.
It adds no capability ``run`` lacks.

CLI <-> dataclass field mapping (see fituna/config.py for the dataclasses):

    --model              -> TargetSpec.model_path
    --target-tps         -> TargetSpec.target_tokens_per_sec
    --max-quality-loss   -> TargetSpec.max_quality_loss_pct
    --ctx (comma-sep ok) -> TargetSpec.ctx_candidates (first value is .ctx)
    --quant (comma-sep)  -> TargetSpec.quant_candidates
                             (always re-sorted to quality-descending order,
                             using the canonical order documented on
                             TargetSpec.quant_candidates' default)
    --gpu / --vram-mb    -> hardware.parse_manual_hardware(...)
    --llama-bin-dir      -> binaries.locate_binaries(bin_dir=...)
    --wikitext           -> quality module input corpus path (required)
    --out                -> work_dir
    --json                -> report.to_json(...) instead of to_human(...)
    --resume              -> activates a ResultCache at <out>/.fituna_cache.sqlite3
    --export-ollama       -> report.export_ollama_modelfile(...) -> SearchResult
                             .modelfile_path (JSON "modelfile_path"). Attempted
                             on the exit-3 (NoFeasibleConfigError) best-effort
                             path too, using e.closest -- not just on success --
                             and a failed export never eats the report: it's
                             caught, logged as a warning, and the exit code
                             stays whatever the search itself earned.

Exit codes:
    0 = success (meets_target)
    1 = generic error
    2 = BinaryNotFoundError, argparse's own usage-error exit (missing
        required / unrecognized flag, via parser.parse_args() -> sys.exit(2)
        before main()'s FiTunaError mapping runs -- distinguish by whether
        stderr's first line starts with "usage: "), OR `fituna help <cmd>`
        given an unrecognized <cmd> (stderr starts with "fituna help: error: ")
    3 = NoFeasibleConfigError
"""

from __future__ import annotations

import argparse
import contextlib
import io
import logging
import sys
from dataclasses import asdict, fields, replace
from pathlib import Path
from typing import Optional, Sequence

from fituna import __version__, binaries, corpus, doctor, hardware, model_info, quickstart, report, search
from fituna.cache import ResultCache
from fituna.config import BinaryPaths, HardwareProfile, TargetSpec, check_target_ranges
from fituna.errors import BinaryNotFoundError, FiTunaError, NoFeasibleConfigError

logger = logging.getLogger("fituna")

# Canonical quality-descending quant order, reused (not re-hardcoded) from
# TargetSpec.quant_candidates' documented default ("품질 내림차순 고정 순서").
_QUANT_QUALITY_ORDER: tuple[str, ...] = next(
    f.default for f in fields(TargetSpec) if f.name == "quant_candidates"
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fituna",
        description=(
            "Find the highest-quality llama.cpp quantization that meets a "
            "target throughput within a quality-loss budget, with the fewest "
            "GPU layers (-ngl) it needs, by benchmarking on your actual hardware."
        ),
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="enable debug logging"
    )
    parser.add_argument("-V", "--version", action="version", version=f"fituna {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="search for a config meeting the target spec")
    model_src = run.add_mutually_exclusive_group(required=True)
    model_src.add_argument("--model", help="path to .gguf file or HF model dir")
    model_src.add_argument(
        "--hf",
        help=(
            "HuggingFace repo to download an F16/BF16 .gguf from, as "
            "'repo[:filename]' (e.g. bartowski/SmolLM2-135M-Instruct-GGUF); "
            "saved into --out and reused on later runs"
        ),
    )
    run.add_argument("--target-tps", type=float, required=True, dest="target_tps")
    run.add_argument(
        "--max-quality-loss", type=float, required=True, dest="max_quality_loss"
    )
    run.add_argument("--ctx", default="4096", help="comma-separated context length(s)")
    run.add_argument(
        "--quant",
        default="Q8_0,Q6_K,Q5_K_M,Q4_K_M,Q3_K_M,Q2_K",
        help="comma-separated quant type candidates",
    )
    run.add_argument("--gpu", choices=["none", "nvidia", "amd", "apple"], default=None)
    run.add_argument("--vram-mb", type=int, default=None, dest="vram_mb")
    run.add_argument("--llama-bin-dir", default=None, dest="llama_bin_dir")
    run.add_argument(
        "--quality-corpus",
        "--wikitext",  # historical name, kept as an alias
        required=True,
        dest="wikitext",
        help=(
            "plain-text corpus for perplexity-based quality measurement. "
            "Any UTF-8 text works: wikitext-2 (English default), Korean "
            "Wikipedia for Korean models, your own domain text -- quality "
            "loss is only meaningful on text resembling your workload. Run "
            "`fituna fetch-corpus` to download one, or see README 'Get a "
            "quality corpus' for details."
        ),
    )
    run.add_argument(
        "--ppl-chunks",
        type=int,
        default=32,
        dest="ppl_chunks",
        help=(
            "limit llama-perplexity to this many chunks per quant (default: 32; "
            "0 or negative = full corpus, unlimited). Quality loss is a "
            "statistical estimate, not an exact figure -- a full wikitext-2 "
            "test-set pass per quant candidate can take hours on a multi-GB "
            "model, so the default trades some precision for a search that "
            "finishes in minutes. Pass a larger value or 0 for a more rigorous "
            "(slower) estimate."
        ),
    )
    run.add_argument(
        "--quality-metric",
        choices=["ppl", "kld"],
        default="ppl",
        dest="quality_metric",
        help=(
            "quality evaluation metric: 'ppl' (perplexity loss, default) or "
            "'kld' (KL divergence against baseline model logits; writes a "
            "reference logits file to --out that grows with vocabulary size "
            "and --ppl-chunks, roughly 100 MB per 4 chunks for a 49k-token "
            "vocabulary)"
        ),
    )
    run.add_argument("--out", default="./out", help="working/output directory")
    run.add_argument("--json", action="store_true", help="emit JSON report to stdout")
    run.add_argument(
        "--resume", action="store_true", help="reuse cached bench/quality results"
    )
    run.add_argument(
        "--export-ollama",
        action="store_true",
        dest="export_ollama",
        help=(
            "write an Ollama Modelfile (FROM + num_gpu/num_ctx of the winning "
            "config) next to the produced .gguf in --out, ready for "
            "`ollama create <name> -f <out>/Modelfile`"
        ),
    )

    sub.add_parser("detect-hw", help="print auto-detected hardware profile")

    lb = sub.add_parser("list-binaries", help="show resolved llama.cpp binaries")
    lb.add_argument("--llama-bin-dir", default=None, dest="llama_bin_dir")

    doc = sub.add_parser(
        "doctor",
        help="diagnose the environment: Python version, llama.cpp binaries (found "
        "and actually runnable)/version, whether the build can use your GPU, "
        "hardware detection, output directory writability, and free disk space",
    )
    doc.add_argument("--llama-bin-dir", default=None, dest="llama_bin_dir")
    doc.add_argument("--out", default="./out", help="directory to check for write access and free disk space")
    doc.add_argument("--json", action="store_true", help="emit JSON report to stdout")

    fc = sub.add_parser(
        "fetch-corpus",
        help="download a plain-text quality-evaluation corpus (HuggingFace "
        "dataset-viewer API, stdlib urllib only -- no pip install needed)",
    )
    fc.add_argument("--lang", choices=["en", "ko"], default="en")
    fc.add_argument("--out", required=True, help="output UTF-8 text file path")
    fc.add_argument(
        "--rows", type=int, default=None,
        help="number of rows to fetch (default: 1000 for en, 500 for ko)",
    )
    fc.add_argument(
        "--dataset", default=None,
        help="override HuggingFace dataset id (must be given together with --config/--split)",
    )
    fc.add_argument(
        "--config", default=None, dest="hf_config", metavar="CONFIG",
        help="override dataset config name (must be given together with --dataset/--split)",
    )
    fc.add_argument(
        "--split", default=None,
        help="override dataset split name (must be given together with --dataset/--config)",
    )

    qs = sub.add_parser(
        "quickstart",
        help="interactive wizard: environment check -> targets -> license -> "
        "model -> corpus -> the assembled `fituna run` command, executed for you",
        description="Interactive six-step wizard that asks what you want, picks "
        "a model, fetches a corpus, then assembles and runs the equivalent "
        "`fituna run` command in-process -- it adds no capability `run` lacks "
        "and prints the exact command it runs. Requires an interactive "
        "terminal (a non-TTY stdin exits 1).",
    )
    qs.add_argument(
        "--llama-bin-dir", default=None, dest="llama_bin_dir",
        help="directory holding the llama.cpp binaries, passed straight through "
        "to the assembled `fituna run` (default: search PATH)",
    )
    qs.add_argument("--out", default="./out", help="working/output directory")

    hp = sub.add_parser(
        "help",
        help="print a task-oriented overview, or 'fituna help <command>' for "
        "that command's full options",
    )
    hp.add_argument(
        "topic",
        nargs="?",
        default=None,
        metavar="command",
        help="subcommand to show full -h output for",
    )

    return parser


def _parse_ctx_candidates(raw: str) -> tuple[int, ...]:
    """Comma-separated ints -> de-duplicated tuple, order preserved (first
    entry becomes TargetSpec.ctx).

    Raises FiTunaError (not a bare ValueError) on a non-integer entry, so a
    typo here gets `main()`'s clean "log + exit 1" FiTunaError handling
    instead of falling through to the generic-Exception branch, which dumps
    a full Python traceback for what's just a malformed CLI argument.
    """
    seen: set[int] = set()
    ordered: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            value = int(part)
        except ValueError as exc:
            raise FiTunaError(
                f"--ctx: {part!r} is not an integer (expected comma-separated "
                "context lengths, e.g. --ctx 4096 or --ctx 4096,8192)"
            ) from exc
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    if not ordered:
        raise FiTunaError("--ctx must contain at least one context length")
    return tuple(ordered)


def _sort_quants_by_quality(raw: str) -> tuple[str, ...]:
    """Comma-separated quant names -> de-duplicated tuple sorted by the
    canonical quality-descending order. Unrecognized quant strings sort
    last, in their original relative order (stable sort)."""
    order = {q: i for i, q in enumerate(_QUANT_QUALITY_ORDER)}
    seen: set[str] = set()
    quants: list[str] = []
    for part in raw.split(","):
        part = part.strip().upper()  # llama-quantize type names are all upper-case
        if part and part not in seen:
            seen.add(part)
            quants.append(part)
    if not quants:
        raise FiTunaError("--quant must contain at least one quant type")
    return tuple(sorted(quants, key=lambda q: order.get(q, len(order))))


def _format_hardware(hw: HardwareProfile) -> str:
    d = asdict(hw)
    d["gpu_vendor"] = hw.gpu_vendor.value
    return "\n".join(f"{k}: {v}" for k, v in d.items())


def _format_binaries(
    paths: BinaryPaths, quant_types: list[str], version: Optional[str]
) -> str:
    lines = [
        f"llama_quantize: {paths.llama_quantize}",
        f"llama_bench: {paths.llama_bench}",
        f"llama_perplexity: {paths.llama_perplexity}",
        f"llama_imatrix: {paths.llama_imatrix or '(not found)'}",
        f"convert_script: {paths.convert_script or '(not found)'}",
        f"llama.cpp version: {version or 'unknown'}",
        "supported quant types: "
        + (", ".join(quant_types) if quant_types else "unknown"),
    ]
    return "\n".join(lines)


def _cmd_detect_hw(args: argparse.Namespace) -> int:
    hw = hardware.detect_hardware()
    print(_format_hardware(hw))
    return 0


def _cmd_list_binaries(args: argparse.Namespace) -> int:
    bin_dir = Path(args.llama_bin_dir) if args.llama_bin_dir else None
    paths = binaries.locate_binaries(bin_dir=bin_dir)
    quant_types = binaries.list_supported_quant_types(paths)
    version = binaries.get_llama_cpp_version(paths)
    print(_format_binaries(paths, quant_types, version))
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    """Never raises (see fituna/doctor.py): every check is individually
    guarded, and the exit code is computed from the check results rather
    than from a caught exception, so this bypasses main()'s FiTunaError
    handling entirely -- there is nothing for it to catch here."""
    bin_dir = Path(args.llama_bin_dir) if args.llama_bin_dir else None
    out_dir = Path(args.out)
    checks = doctor.run_checks(bin_dir, out_dir)
    print(doctor.to_json(checks) if args.json else doctor.to_human(checks))
    return doctor.exit_code(checks)


def _cmd_fetch_corpus(args: argparse.Namespace) -> int:
    """No try/except of its own: network/HTTP/schema failures surface as a
    plain FiTunaError (see fituna/corpus.py), which propagates up to
    main()'s generic FiTunaError branch (log + exit 1) -- a network failure
    here is a generic error, not one of the special-cased exit codes."""
    out_path = Path(args.out)
    count = corpus.fetch_corpus(
        out_path,
        lang=args.lang,
        rows=args.rows,
        dataset=args.dataset,
        config=args.hf_config,
        split=args.split,
        progress_cb=logger.info,
    )
    print(f"Wrote {count} rows to {out_path}")
    if args.dataset is None:
        print(corpus.PRESETS[args.lang].license_note)
    else:
        # A custom --dataset/--config/--split override may point at a
        # dataset under any license -- printing the preset's CC BY-SA
        # notice here would be an unverified (and likely false) claim
        # about someone else's dataset, so this stays generic instead.
        print(
            f"Corpus: {args.dataset} ({args.hf_config}/{args.split}) -- not "
            "a built-in preset; check this dataset's own license before "
            f"redistributing: https://huggingface.co/datasets/{args.dataset}"
        )
    return 0


# One task-oriented page instead of the argparse dump ("fituna --help" lists
# every flag of every subcommand; this points at the handful of commands
# most sessions actually need, in the order a first-time user hits them).
_HELP_PAGE = """\
FiTuna -- llama.cpp 양자화 설정을 실측으로 찾는 CLI
Find the highest-quality llama.cpp quant that meets your target, with the
fewest GPU layers it needs -- measured on your actual hardware.

처음이라면 (getting started):
  fituna quickstart      대화형 마법사 -- 전 과정을 안내
                          interactive wizard through the whole flow

자주 쓰는 명령 (common commands):
  fituna doctor           환경 점검 (Python/바이너리/하드웨어/디스크)
                          check the environment is ready
  fituna fetch-corpus     품질 측정용 코퍼스 다운로드
                          download a corpus for the quality gate
  fituna run              탐색 실행 -- 목표 tok/s·품질손실을 만족하는 구성 찾기
                          run the search for a config meeting your target
  fituna run --json       스크립트/CI용 JSON 출력
                          machine-readable output for scripts and CI
  fituna detect-hw        자동 감지된 하드웨어 프로파일 출력
                          print the auto-detected hardware profile
  fituna list-binaries    해석된 llama.cpp 바이너리 경로 출력
                          show resolved llama.cpp binaries

에이전트에서 쓴다면 (from an AI agent):
  fituna-mcp              MCP 서버 -- 실측 기반 추천 (별도 실행 파일,
                          fituna 서브커맨드가 아니다)
                          MCP server exposing measured recommendations
                          (separate entry point, not a fituna subcommand)

각 명령의 전체 옵션 (full options per command):
  fituna <command> -h
  fituna help <command>   위와 동일 / same as -h

더 보기 (more): README.md(English) / README.ko.md(한국어), REVIEWERS.md
"""


def _cmd_help(args: argparse.Namespace) -> int:
    """Bare `fituna help` prints the task-oriented page above. `fituna help
    <cmd>` looks `<cmd>` up in the *actual* registered subparsers (not a
    separate hardcoded list) and prints that subparser's own -h text, so it
    can never drift out of sync with what `fituna <cmd> -h` prints. An
    unknown <cmd> is a usage error -> exit 2, argparse convention."""
    if args.topic is None:
        print(_HELP_PAGE)
        return 0

    subparsers_action = next(
        a for a in _build_parser()._actions
        if isinstance(a, argparse._SubParsersAction)
    )
    subparser = subparsers_action.choices.get(args.topic)
    if subparser is None:
        known = ", ".join(sorted(subparsers_action.choices))
        print(
            f"fituna help: error: unknown command {args.topic!r} "
            f"(choices: {known})",
            file=sys.stderr,
        )
        return 2

    print(subparser.format_help())
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    ctx_candidates = _parse_ctx_candidates(args.ctx)
    quant_candidates = _sort_quants_by_quality(args.quant)
    check_target_ranges(args.target_tps, args.max_quality_loss, ctx_candidates)
    if args.vram_mb is not None and args.vram_mb <= 0:
        raise FiTunaError(f"--vram-mb must be positive, got {args.vram_mb}")
    corpus_path = Path(args.wikitext)
    # checked here too (quality.py checks again) so a typo fails before an HF download
    if not corpus_path.is_file() or corpus_path.stat().st_size == 0:
        raise FiTunaError(
            f"--quality-corpus {corpus_path} is missing or empty -- run "
            "`fituna fetch-corpus --out wiki.txt` to download one"
        )

    bin_dir = Path(args.llama_bin_dir) if args.llama_bin_dir else None
    bins = binaries.locate_binaries(bin_dir=bin_dir)

    hw = hardware.parse_manual_hardware(args.gpu, args.vram_mb)

    work_dir = Path(args.out)
    try:
        work_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise FiTunaError(f"--out: cannot create directory {work_dir}: {exc}") from exc

    model_path = quickstart.resolve_hf_model(args.hf, work_dir) if args.hf else Path(args.model)
    base_gguf = model_info.ensure_base_gguf(model_path, work_dir, bins)
    minfo = model_info.read_model_info(base_gguf, bins)
    if model_info.is_already_quantized(minfo):
        print(
            f"WARNING: {base_gguf.name} is already quantized "
            f"(GGUF general.file_type={minfo.file_type}). Quality loss will be "
            "measured against this quantized file, not the original F16/F32 "
            "weights, and re-quantizing it degrades quality twice. Use an "
            "F16/BF16/F32 GGUF (or the original HF directory) as --model.",
            file=sys.stderr,
        )

    target = TargetSpec(
        model_path=model_path,
        target_tokens_per_sec=args.target_tps,
        max_quality_loss_pct=args.max_quality_loss,
        ctx=ctx_candidates[0],
        ctx_candidates=ctx_candidates,
        quant_candidates=quant_candidates,
        # --ppl-chunks 0 (or negative) means "full corpus" -> None, matching
        # compute_perplexity's own "no limit" sentinel.
        ppl_chunks=args.ppl_chunks if args.ppl_chunks > 0 else None,
        quality_metric=args.quality_metric,
    )

    cache = ResultCache(work_dir / ".fituna_cache.sqlite3") if args.resume else None
    wikitext_path = corpus_path

    result = search.search(
        target,
        minfo,
        hw,
        bins,
        work_dir,
        wikitext_path,
        cache=cache,
        progress_cb=logger.info,
    )

    export_error: Optional[FiTunaError] = None
    if args.export_ollama:
        # Export lives in report.py (not inline here) so other front ends --
        # the MCP server, a library caller -- reach the same code path.
        # Guarded: a multi-minute search result must still get reported even
        # if the Modelfile write fails (e.g. --out became read-only in the
        # meantime) -- letting this raise would propagate to main()'s
        # generic FiTunaError handler, which exits 1 *without ever printing
        # the report*, discarding the whole measurement over an unrelated
        # export failure.
        try:
            modelfile = report.export_ollama_modelfile(result.gguf_path, result.config)
            result = replace(result, modelfile_path=modelfile)
        except FiTunaError as exc:
            export_error = exc

    print(report.to_json(result) if args.json else report.to_human(result))
    if export_error is not None:
        logger.warning("could not write Ollama Modelfile: %s", export_error)
    return 0 if result.meets_target else 1


_DISPATCH = {
    "run": _cmd_run,
    # Dispatched straight to fituna.quickstart (no _cmd_ wrapper): run_wizard
    # takes the same argparse.Namespace every other handler here does, and it
    # sets `args.export_ollama` on it so main()'s exit-3 branch below can
    # export the best-effort Modelfile exactly as `run --export-ollama` does.
    "quickstart": quickstart.run_wizard,
    "detect-hw": _cmd_detect_hw,
    "list-binaries": _cmd_list_binaries,
    "doctor": _cmd_doctor,
    "fetch-corpus": _cmd_fetch_corpus,
    "help": _cmd_help,
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Parse argv, dispatch to the requested subcommand, map exceptions to
    exit codes:
        BinaryNotFoundError   -> log + return 2
        NoFeasibleConfigError -> log + return 3
        FiTunaError (other)   -> log + return 1
        unexpected Exception  -> log + return 1
        success               -> return 0 if result.meets_target else 1
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        return _DISPATCH[args.command](args)
    except BinaryNotFoundError as e:
        logger.error(str(e))
        return 2
    except NoFeasibleConfigError as e:
        logger.error(str(e))
        if e.closest is not None:
            closest = e.closest
            # --export-ollama must still export here: e.closest is a real
            # SearchResult with a real gguf_path (the best-effort config the
            # search found), and the best-effort report below already
            # renders the artifact block. Without this, the rendered hint
            # ("re-run with --export-ollama...") would tell the user to redo
            # something they already asked for -- re-running just loops back
            # into the same NoFeasibleConfigError.
            if getattr(args, "export_ollama", False):
                try:
                    modelfile = report.export_ollama_modelfile(
                        closest.gguf_path, closest.config
                    )
                    closest = replace(closest, modelfile_path=modelfile)
                except FiTunaError as exc:
                    logger.warning("could not write Ollama Modelfile: %s", exc)
            try:
                logger.info(
                    "closest best-effort attempt:\n%s", report.to_human(closest)
                )
            except Exception:  # pragma: no cover - reporting must not mask exit code
                pass
        return 3
    except FiTunaError as e:
        logger.error(str(e))
        return 1
    except Exception:
        logger.exception("unexpected error")
        return 1


def _selfcheck() -> None:
    """Pure-logic sanity check for parser wiring and the CLI<->dataclass
    mapping described in the module docstring. No subprocess/filesystem/
    network I/O -- safe to run anywhere.

    Run directly: ``python -m fituna.cli --selfcheck`` (running the file
    path directly, e.g. ``python fituna/cli.py``, fails to import the
    ``fituna`` package itself -- the script's own directory *is* the
    package, so it doesn't appear on ``sys.path``).
    """
    parser = _build_parser()

    run_args = parser.parse_args(
        [
            "run",
            "--model",
            "m.gguf",
            "--target-tps",
            "20",
            "--max-quality-loss",
            "5",
            "--ctx",
            "8192,4096,2048,4096",
            "--quant",
            "Q2_K,Q8_0,FOO,Q6_K",
            "--wikitext",
            "wiki.txt",
        ]
    )
    assert run_args.command == "run"
    assert run_args.target_tps == 20.0
    assert run_args.max_quality_loss == 5.0

    ctxs = _parse_ctx_candidates(run_args.ctx)
    assert ctxs == (8192, 4096, 2048), ctxs  # de-duplicated, order preserved
    assert ctxs[0] == 8192  # first value becomes TargetSpec.ctx

    quants = _sort_quants_by_quality(run_args.quant)
    # quality-descending: Q8_0 before Q6_K before Q2_K; unknown "FOO" sorts last
    assert quants == ("Q8_0", "Q6_K", "Q2_K", "FOO"), quants

    dh_args = parser.parse_args(["detect-hw"])
    assert dh_args.command == "detect-hw"

    lb_args = parser.parse_args(["list-binaries", "--llama-bin-dir", "/opt/llama"])
    assert lb_args.command == "list-binaries"
    assert lb_args.llama_bin_dir == "/opt/llama"

    doc_args = parser.parse_args(["doctor", "--out", "/tmp/o", "--json"])
    assert doc_args.command == "doctor"
    assert doc_args.out == "/tmp/o"
    assert doc_args.json is True
    assert doc_args.llama_bin_dir is None

    fc_args = parser.parse_args(["fetch-corpus", "--out", "corpus.txt"])
    assert fc_args.command == "fetch-corpus"
    assert fc_args.lang == "en"  # default
    assert fc_args.rows is None  # resolved from the --lang preset inside fituna.corpus
    assert fc_args.dataset is None and fc_args.hf_config is None and fc_args.split is None

    fc_ko_args = parser.parse_args(
        ["fetch-corpus", "--lang", "ko", "--out", "kowiki.txt", "--rows", "500"]
    )
    assert fc_ko_args.lang == "ko"
    assert fc_ko_args.rows == 500

    fc_override_args = parser.parse_args(
        [
            "fetch-corpus", "--out", "custom.txt",
            "--dataset", "org/name", "--config", "cfg", "--split", "train",
        ]
    )
    assert fc_override_args.dataset == "org/name"
    assert fc_override_args.hf_config == "cfg"
    assert fc_override_args.split == "train"

    qs_args = parser.parse_args(["quickstart"])
    assert qs_args.command == "quickstart"
    assert qs_args.out == "./out"  # default, same as doctor/run
    assert qs_args.llama_bin_dir is None

    help_args = parser.parse_args(["help"])
    assert help_args.command == "help"
    assert help_args.topic is None

    help_topic_args = parser.parse_args(["help", "run"])
    assert help_topic_args.topic == "run"

    # Every registered subcommand's -h must be reachable through `fituna
    # help <cmd>` (not just `<cmd> -h` directly) -- exercise the actual
    # lookup path _cmd_help uses, for every real subcommand.
    subparsers_action = next(
        a for a in parser._actions if isinstance(a, argparse._SubParsersAction)
    )
    for subparser in subparsers_action.choices.values():
        subparser.format_help()  # must not raise for any registered name

    # _cmd_help prints its "unknown command" diagnostic to stderr -- correct
    # for a real invocation, but here it is the asserted behaviour, not a
    # problem. Swallow it so `--selfcheck` emits only its OK line.
    with contextlib.redirect_stderr(io.StringIO()):
        assert _cmd_help(argparse.Namespace(topic="not-a-real-command")) == 2

    try:
        _parse_ctx_candidates("")
    except FiTunaError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected FiTunaError for empty --ctx")

    # A non-integer entry must also raise FiTunaError (clean "log + exit 1"),
    # not a bare ValueError that falls through to a raw traceback dump.
    try:
        _parse_ctx_candidates("4096,not-a-number")
    except FiTunaError as exc:
        assert "not-a-number" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected FiTunaError for a non-integer --ctx entry")

    print("fituna.cli self-check OK")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selfcheck":
        _selfcheck()
        sys.exit(0)
    sys.exit(main())
