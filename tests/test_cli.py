# SPDX-License-Identifier: MIT
"""CLI-level seam tests for ``fituna run``.

Unlike tests/test_report.py (which calls report.py functions directly), these
drive ``cli.main()`` itself: real argparse, the real GGUF header parser in
model_info.py (via a hand-built synthetic .gguf), and the real
--export-ollama wiring in cli.py -- with only ``search.search`` and
``binaries.locate_binaries`` faked out, so no real llama.cpp binaries or
multi-minute search is needed.

This exists because the previous "CLI wiring" tests (test_report.py) only
checked argparse and dataclasses.replace() in isolation -- deleting the
entire ``if args.export_ollama:`` block in cli.py left all of them green.
These tests call through _cmd_run/main() itself, so they die with that
mutation (verified manually; see the finding this file addresses).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from fituna import cli
from fituna.config import (
    BenchResult,
    BinaryPaths,
    CandidateConfig,
    FiTunaError,
    NoFeasibleConfigError,
    QualityResult,
    SearchResult,
)
from fituna.model_info import _build_synthetic_gguf


def _model_gguf(tmp_path: Path) -> Path:
    p = tmp_path / "model.gguf"
    p.write_bytes(_build_synthetic_gguf())
    return p


def _fake_binaries(tmp_path: Path) -> BinaryPaths:
    # Never actually invoked -- search.search() is faked too -- just needs
    # to satisfy BinaryPaths' shape.
    d = tmp_path / "bin"
    d.mkdir(exist_ok=True)
    return BinaryPaths(
        llama_quantize=d / "llama-quantize",
        llama_bench=d / "llama-bench",
        llama_perplexity=d / "llama-perplexity",
    )


def _fake_result(out_dir: Path, meets_target: bool = True) -> SearchResult:
    cand = CandidateConfig(quant="Q4_K_M", ngl=20, ctx=4096)
    gguf = out_dir / "model-Q4_K_M.gguf"
    gguf.write_bytes(b"x" * 2048)
    return SearchResult(
        config=cand,
        bench=BenchResult(
            candidate=cand, prompt_tok_per_sec=100.0, gen_tok_per_sec=25.0,
            vram_used_mb=None, raw_stdout="{}",
        ),
        quality=QualityResult(
            candidate_quant="Q4_K_M", perplexity=6.1, baseline_perplexity=6.0,
            quality_loss_pct=1.5,
        ),
        gguf_path=gguf,
        run_command=["llama-cli", "-m", str(gguf), "-ngl", "20", "-c", "4096"],
        meets_target=meets_target,
    )


def _run_argv(model: Path, wikitext: Path, out: Path, extra=()) -> list[str]:
    return [
        "run", "--model", str(model), "--target-tps", "20",
        "--max-quality-loss", "5", "--wikitext", str(wikitext),
        "--out", str(out), "--json", *extra,
    ]


def _setup(monkeypatch, tmp_path: Path, search_fn) -> tuple[Path, Path, Path]:
    monkeypatch.setattr(cli.search, "search", search_fn)
    monkeypatch.setattr(
        cli.binaries, "locate_binaries", lambda bin_dir=None: _fake_binaries(tmp_path)
    )
    wikitext = tmp_path / "wiki.txt"
    wikitext.write_text("hello world")
    out = tmp_path / "out"
    out.mkdir()
    return _model_gguf(tmp_path), wikitext, out


# ---------------------------------------------------------------------------
# --export-ollama on the success path
# ---------------------------------------------------------------------------

def test_export_ollama_writes_modelfile_and_sets_the_json_field(
    tmp_path, monkeypatch, capsys
):
    result_holder: dict[str, SearchResult] = {}

    def fake_search(*args, **kwargs):
        result_holder["result"] = _fake_result(args[4])  # positional work_dir
        return result_holder["result"]

    model, wikitext, out = _setup(monkeypatch, tmp_path, fake_search)

    rc = cli.main(_run_argv(model, wikitext, out, extra=["--export-ollama"]))

    assert rc == 0
    modelfile = out / "Modelfile"
    assert modelfile.exists()
    result = result_holder["result"]
    assert modelfile.read_text(encoding="utf-8") == (
        f"FROM ./{result.gguf_path.name}\nPARAMETER num_gpu 20\nPARAMETER num_ctx 4096\n"
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["modelfile_path"] == str(modelfile.resolve())


def test_without_export_ollama_modelfile_path_stays_null_and_no_file_is_written(
    tmp_path, monkeypatch, capsys
):
    def fake_search(*args, **kwargs):
        return _fake_result(args[4])  # positional work_dir

    model, wikitext, out = _setup(monkeypatch, tmp_path, fake_search)

    rc = cli.main(_run_argv(model, wikitext, out))

    assert rc == 0
    assert not (out / "Modelfile").exists()
    payload = json.loads(capsys.readouterr().out)
    assert payload["modelfile_path"] is None


# ---------------------------------------------------------------------------
# --export-ollama on the exit-3 (NoFeasibleConfigError) best-effort path
# ---------------------------------------------------------------------------

def test_export_ollama_on_exit_3_path_writes_modelfile_and_drops_the_rerun_hint(
    tmp_path, monkeypatch, caplog
):
    def fake_search(*args, **kwargs):
        closest = _fake_result(args[4], meets_target=False)  # positional work_dir
        raise NoFeasibleConfigError("no quant met the target", closest=closest)

    model, wikitext, out = _setup(monkeypatch, tmp_path, fake_search)

    with caplog.at_level(logging.INFO, logger="fituna"):
        rc = cli.main(_run_argv(model, wikitext, out, extra=["--export-ollama"]))

    assert rc == 3
    modelfile = out / "Modelfile"
    assert modelfile.exists()
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert f"ollama create <name> -f {modelfile.resolve()}" in logged
    # The user already passed --export-ollama -- telling them to re-run it
    # would just loop, so the hint must not appear once the export worked.
    assert "re-run with --export-ollama" not in logged


def test_export_ollama_failure_does_not_eat_the_report(tmp_path, monkeypatch, capsys, caplog):
    """A failed export (e.g. --out became read-only) must not discard the
    multi-minute search result: the report still prints, and the exit code
    stays whatever the search itself earned -- not a generic 1 from an
    uncaught FiTunaError."""

    def fake_search(*args, **kwargs):
        return _fake_result(args[4], meets_target=True)  # positional work_dir

    model, wikitext, out = _setup(monkeypatch, tmp_path, fake_search)
    monkeypatch.setattr(
        cli.report,
        "export_ollama_modelfile",
        lambda gguf_path, config: (_ for _ in ()).throw(
            FiTunaError("could not write Ollama Modelfile: read-only filesystem")
        ),
    )

    with caplog.at_level(logging.WARNING, logger="fituna"):
        rc = cli.main(_run_argv(model, wikitext, out, extra=["--export-ollama"]))

    assert rc == 0  # meets_target=True -- unaffected by the export failure
    payload = json.loads(capsys.readouterr().out)
    assert payload["meets_target"] is True
    assert payload["modelfile_path"] is None
    assert any("read-only filesystem" in r.getMessage() for r in caplog.records)


def test_export_ollama_failure_on_exit_3_path_still_exits_3_with_a_warning(
    tmp_path, monkeypatch, caplog
):
    def fake_search(*args, **kwargs):
        closest = _fake_result(args[4], meets_target=False)  # positional work_dir
        raise NoFeasibleConfigError("no quant met the target", closest=closest)

    model, wikitext, out = _setup(monkeypatch, tmp_path, fake_search)
    monkeypatch.setattr(
        cli.report,
        "export_ollama_modelfile",
        lambda gguf_path, config: (_ for _ in ()).throw(FiTunaError("disk full")),
    )

    with caplog.at_level(logging.WARNING, logger="fituna"):
        rc = cli.main(_run_argv(model, wikitext, out, extra=["--export-ollama"]))

    assert rc == 3
    assert any("disk full" in r.getMessage() for r in caplog.records)


def _registered_subcommands() -> list[str]:
    """Subcommand names straight from a freshly built parser -- never a
    hardcoded list here, so this can't drift out of sync with cli.py."""
    parser = cli._build_parser()
    subparsers_action = next(
        a for a in parser._actions if isinstance(a, argparse._SubParsersAction)
    )
    return list(subparsers_action.choices)


def test_help_page_mentions_every_registered_subcommand(capsys):
    # Asserts every *registered* name is *mentioned* in the page, never the
    # reverse -- the page may also name things that aren't subcommands at all
    # (e.g. `fituna-mcp`, a separate entry point).
    subcommands = _registered_subcommands()
    assert subcommands  # sanity: the parser actually has subcommands

    rc = cli.main(["help"])
    out = capsys.readouterr().out

    assert rc == 0
    for name in subcommands:
        assert name in out, f"fituna help page does not mention {name!r}"


def test_help_with_topic_delegates_to_that_subcommands_h_text(capsys):
    rc = cli.main(["help", "run"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "usage: fituna run" in out
    assert "--target-tps" in out  # a real `run`-only flag, not boilerplate


def test_help_with_unknown_topic_is_a_usage_error(capsys):
    rc = cli.main(["help", "not-a-real-command"])
    err = capsys.readouterr().err

    assert rc == 2
    assert "not-a-real-command" in err


def test_cli_quality_metric_parsing():
    parser = cli._build_parser()
    args_default = parser.parse_args([
        "run", "--model", "m.gguf", "--target-tps", "20",
        "--max-quality-loss", "5", "--wikitext", "w.txt",
    ])
    assert args_default.quality_metric == "ppl"

    args_kld = parser.parse_args([
        "run", "--model", "m.gguf", "--target-tps", "20",
        "--max-quality-loss", "5", "--wikitext", "w.txt",
        "--quality-metric", "kld",
    ])
    assert args_kld.quality_metric == "kld"

