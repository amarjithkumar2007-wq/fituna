# SPDX-License-Identifier: MIT
"""Artifact-centric result exits: server command, Ollama export, JSON fields,
and the human rendering order.

No subprocess is ever spawned here -- llama-cli/llama-server are *located*,
never executed, so these tests only ever touch the filesystem (tmp_path) and
PATH lookups.
"""

import dataclasses
import json
from pathlib import Path

import pytest

from fituna.config import (
    BenchResult,
    BinaryPaths,
    CandidateConfig,
    FiTunaError,
    QualityResult,
    SearchResult,
)
from fituna.report import (
    build_server_command,
    export_ollama_modelfile,
    to_human,
    to_json,
)


def _binaries(dir_path: Path) -> BinaryPaths:
    return BinaryPaths(
        llama_quantize=dir_path / "llama-quantize",
        llama_bench=dir_path / "llama-bench",
        llama_perplexity=dir_path / "llama-perplexity",
    )


def _result(gguf: Path, **overrides) -> SearchResult:
    cand = CandidateConfig(quant="Q4_K_M", ngl=33, ctx=4096)
    kwargs = dict(
        config=cand,
        bench=BenchResult(
            candidate=cand,
            prompt_tok_per_sec=120.0,
            gen_tok_per_sec=30.81,
            vram_used_mb=None,
            raw_stdout="{}",
        ),
        quality=QualityResult(
            candidate_quant="Q4_K_M",
            perplexity=6.1,
            baseline_perplexity=6.0,
            quality_loss_pct=1.73,
        ),
        gguf_path=gguf,
        run_command=["llama-cli", "-m", str(gguf), "-ngl", "33", "-c", "4096"],
        meets_target=True,
    )
    kwargs.update(overrides)
    return SearchResult(**kwargs)


# ---------------------------------------------------------------------------
# build_server_command
# ---------------------------------------------------------------------------

def test_server_command_shape_and_port():
    cand = CandidateConfig(quant="Q4_K_M", ngl=33, ctx=8192)
    cmd = build_server_command(Path("out/m.gguf"), cand)
    assert cmd == [
        "llama-server",
        "-m",
        str(Path("out/m.gguf")),
        "-ngl",
        "33",
        "-c",
        "8192",
        "--port",
        "8080",
    ]


def test_server_command_prefers_a_binary_next_to_the_known_ones(tmp_path):
    (tmp_path / "llama-quantize").touch()
    (tmp_path / "llama-server").touch()
    cand = CandidateConfig(quant="Q4_K_M", ngl=1, ctx=2048)
    cmd = build_server_command(Path("m.gguf"), cand, _binaries(tmp_path))
    assert cmd[0] == str(tmp_path / "llama-server")


def test_server_command_falls_back_to_the_bare_name_when_nothing_is_found(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("PATH", "")
    cand = CandidateConfig(quant="Q4_K_M", ngl=1, ctx=2048)
    cmd = build_server_command(Path("m.gguf"), cand, _binaries(tmp_path / "nope"))
    assert cmd[0] == "llama-server"  # advisory command, never a crash


# ---------------------------------------------------------------------------
# export_ollama_modelfile
# ---------------------------------------------------------------------------

def test_modelfile_content_is_relative_from_plus_num_gpu_and_num_ctx(tmp_path):
    gguf = tmp_path / "model-Q4_K_M.gguf"
    gguf.touch()
    cand = CandidateConfig(quant="Q4_K_M", ngl=33, ctx=4096)

    path = export_ollama_modelfile(gguf, cand)

    assert path == (tmp_path / "Modelfile").resolve()
    assert path.read_text(encoding="utf-8") == (
        "FROM ./model-Q4_K_M.gguf\n"
        "PARAMETER num_gpu 33\n"
        "PARAMETER num_ctx 4096\n"
    )


def test_modelfile_from_is_relative_so_the_out_dir_can_be_moved(tmp_path):
    """A relocatable --out dir is the whole point of the relative FROM: an
    absolute path baked in here would break the moment the user moves or
    ships the directory."""
    gguf = tmp_path / "sub" / "model-Q8_0.gguf"
    gguf.parent.mkdir()
    gguf.touch()
    path = export_ollama_modelfile(gguf, CandidateConfig(quant="Q8_0", ngl=0, ctx=512))
    first = path.read_text(encoding="utf-8").splitlines()[0]
    assert first == "FROM ./model-Q8_0.gguf"
    assert str(tmp_path) not in path.read_text(encoding="utf-8")


def test_modelfile_write_is_atomic_and_leaves_nothing_behind_on_failure(tmp_path):
    """The temp file must be gone whether the write succeeded (os.replace
    consumed it) or failed (explicit cleanup), so `ollama create` can never
    read a half-written file or trip over a stray Modelfile.tmp."""
    gguf = tmp_path / "m.gguf"
    gguf.touch()
    cand = CandidateConfig(quant="Q4_K_M", ngl=10, ctx=4096)

    export_ollama_modelfile(gguf, cand)
    assert not (tmp_path / "Modelfile.tmp").exists()

    # A directory sitting where the temp file goes makes the write fail.
    (tmp_path / "Modelfile.tmp").mkdir()
    (tmp_path / "Modelfile").unlink()
    with pytest.raises(FiTunaError):
        export_ollama_modelfile(gguf, cand)
    assert not (tmp_path / "Modelfile").exists()  # no partial artifact


def test_modelfile_overwrites_a_previous_export(tmp_path):
    gguf = tmp_path / "m.gguf"
    gguf.touch()
    export_ollama_modelfile(gguf, CandidateConfig(quant="Q8_0", ngl=1, ctx=512))
    path = export_ollama_modelfile(gguf, CandidateConfig(quant="Q8_0", ngl=99, ctx=8192))
    assert path.read_text(encoding="utf-8").endswith(
        "PARAMETER num_gpu 99\nPARAMETER num_ctx 8192\n"
    )


# ---------------------------------------------------------------------------
# to_json -- additive fields only
# ---------------------------------------------------------------------------

def test_json_gains_the_artifact_fields_without_touching_the_existing_ones(tmp_path):
    gguf = tmp_path / "m.gguf"
    result = _result(gguf)
    payload = json.loads(to_json(result))

    # additive
    assert payload["llama_server_command"] is None
    assert payload["modelfile_path"] is None
    # existing, unchanged
    assert payload["gguf_path"] == str(gguf)
    assert payload["run_command"] == result.run_command
    assert payload["meets_target"] is True
    assert payload["config"]["quant"] == "Q4_K_M"


def test_json_serializes_the_populated_artifact_fields(tmp_path):
    gguf = tmp_path / "m.gguf"
    server_cmd = build_server_command(gguf, CandidateConfig("Q4_K_M", 33, 4096))
    result = _result(
        gguf,
        llama_server_command=server_cmd,
        modelfile_path=tmp_path / "Modelfile",
    )
    payload = json.loads(to_json(result))
    assert payload["llama_server_command"] == server_cmd
    assert payload["modelfile_path"] == str(tmp_path / "Modelfile")


# ---------------------------------------------------------------------------
# to_human -- the artifact leads, llama-cli is demoted to a check
# ---------------------------------------------------------------------------

def test_human_leads_with_the_artifact_then_server_then_ollama_then_cli(tmp_path):
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"x" * 1500)
    human = to_human(_result(gguf))

    positions = [
        human.index("artifact:"),
        human.index("1) local API server"),
        human.index("2) import into Ollama"),
        human.index("3) terminal chat"),
    ]
    assert positions == sorted(positions)
    assert str(gguf) in human
    assert "1.5 KB" in human  # size, read off disk
    assert "interactive check" in human  # llama-cli is the check, not the goal


def test_human_keeps_the_same_order_for_a_best_effort_result(tmp_path):
    gguf = tmp_path / "m.gguf"
    gguf.touch()
    human = to_human(_result(gguf, meets_target=False))
    assert "BEST EFFORT" in human
    assert human.index("artifact:") < human.index("1) local API server")
    assert human.index("1) local API server") < human.index("3) terminal chat")


def test_human_survives_a_missing_artifact_file(tmp_path):
    """A report must never crash just because the gguf was moved or deleted
    (e.g. a cached --resume result rendered after a cleanup)."""
    human = to_human(_result(tmp_path / "gone.gguf"))
    assert str(tmp_path / "gone.gguf") in human
    assert "already produced during the search" in human


def test_human_says_not_found_only_when_a_lookup_actually_happened(tmp_path):
    """search() stores llama_server_command after really looking (beside
    known binaries, then PATH); a bare name there means the lookup came up
    empty, so "not found" is a checked fact."""
    gguf = tmp_path / "m.gguf"
    gguf.touch()
    human = to_human(_result(gguf, llama_server_command=["llama-server"]))
    assert "llama-server was not found on this machine" in human


def test_human_does_not_claim_not_found_when_nothing_was_checked(tmp_path, monkeypatch):
    """When llama_server_command is None, to_human()'s own fallback builds
    the command with no BinaryPaths, so it never looks anywhere -- the note
    must not claim "not found" for a lookup that never happened."""
    monkeypatch.setenv("PATH", "")
    gguf = tmp_path / "m.gguf"
    gguf.touch()
    human = to_human(_result(gguf))
    assert "llama-server was not found" not in human
    assert "no binary-location info available" in human


def test_human_hints_at_export_ollama_until_a_modelfile_exists(tmp_path):
    gguf = tmp_path / "m.gguf"
    gguf.touch()
    assert "--export-ollama" in to_human(_result(gguf))

    exported = _result(gguf, modelfile_path=tmp_path / "Modelfile")
    human = to_human(exported)
    assert f"ollama create <name> -f {tmp_path / 'Modelfile'}" in human
    assert "--export-ollama" not in human


def test_human_uses_the_stored_server_command_when_present(tmp_path):
    gguf = tmp_path / "m.gguf"
    gguf.touch()
    stored = ["/opt/llama/llama-server", "-m", str(gguf), "--port", "8080"]
    human = to_human(_result(gguf, llama_server_command=stored))
    assert " ".join(stored) in human
    assert "llama-server was not found" not in human


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------

def test_cli_run_parses_export_ollama_flag():
    from fituna.cli import _build_parser

    args = _build_parser().parse_args(
        [
            "run", "--model", "m.gguf", "--target-tps", "20",
            "--max-quality-loss", "5", "--wikitext", "w.txt",
        ]
    )
    assert args.export_ollama is False
    args2 = _build_parser().parse_args(
        [
            "run", "--model", "m.gguf", "--target-tps", "20",
            "--max-quality-loss", "5", "--wikitext", "w.txt", "--export-ollama",
        ]
    )
    assert args2.export_ollama is True


def test_search_result_replace_carries_the_modelfile_path(tmp_path):
    """cli.py attaches the exported Modelfile with dataclasses.replace on a
    frozen SearchResult -- guard that the field survives it."""
    result = _result(tmp_path / "m.gguf")
    attached = dataclasses.replace(result, modelfile_path=tmp_path / "Modelfile")
    assert attached.modelfile_path == tmp_path / "Modelfile"
    assert attached.run_command == result.run_command


def test_human_includes_kld_when_present(tmp_path):
    gguf = tmp_path / "m.gguf"
    gguf.touch()
    res = _result(gguf)
    res_kld = dataclasses.replace(
        res,
        quality=QualityResult(
            candidate_quant="Q4_K_M",
            perplexity=6.1,
            baseline_perplexity=6.0,
            quality_loss_pct=1.67,
            metric="kld",
            kld=0.003412,
        ),
    )
    human = to_human(res_kld)
    assert "kld             : 0.003412" in human

    payload = json.loads(to_json(res_kld))
    assert payload["quality"]["metric"] == "kld"
    assert payload["quality"]["kld"] == 0.003412

