# SPDX-License-Identifier: MIT
"""fituna.report
================

Turns a :class:`~fituna.config.SearchResult` into ready-to-run ``llama-cli``
/ ``llama-server`` commands, an Ollama ``Modelfile``, and a JSON /
human-readable report.

Almost everything here is a pure function of already-computed dataclasses
(see fituna/config.py). The two exceptions are deliberate and read-only-ish:
``to_human()`` stats the produced GGUF to print its size, and
``export_ollama_modelfile()`` writes the Modelfile (atomically). No
subprocess is ever spawned -- llama-cli/llama-server are *located*, never
executed.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
from pathlib import Path
from typing import Optional

from fituna.config import BinaryPaths, CandidateConfig, FiTunaError, SearchResult

_LLAMA_CLI_NAMES = ("llama-cli", "llama-cli.exe", "main", "main.exe")
_LLAMA_SERVER_NAMES = ("llama-server", "llama-server.exe", "server", "server.exe")


def _find_beside_binaries(binaries: BinaryPaths, names: tuple[str, ...]) -> str:
    """Best-effort locate one of ``names`` next to the known binaries, else PATH.

    llama.cpp does not expose llama-cli/llama-server in BinaryPaths (neither
    is needed for search/bench/quantize/quality), so we look for them:
    1) alongside the binaries we do know about (same install dir), 2) on
    PATH, 3) fall back to ``names[0]``, the bare command name, so the printed
    command is still copy-pasteable even if this machine happens not to have
    it installed. Located only -- FiTuna never executes either binary.
    """
    candidate_dirs = {
        binaries.llama_quantize.parent,
        binaries.llama_bench.parent,
        binaries.llama_perplexity.parent,
    }
    for d in candidate_dirs:
        for name in names:
            p = d / name
            if p.is_file():
                return str(p)

    for name in names:
        found = shutil.which(name)
        if found:
            return found

    # not found anywhere -- return the bare command name rather than raising.
    # The command is advisory (the user copy-pastes it); search/bench/quantize
    # already succeeded without needing it.
    return names[0]


def _find_llama_cli(binaries: BinaryPaths) -> str:
    """Locate llama-cli (see :func:`_find_beside_binaries`)."""
    return _find_beside_binaries(binaries, _LLAMA_CLI_NAMES)


def _find_llama_server(binaries: BinaryPaths) -> str:
    """Locate llama-server (see :func:`_find_beside_binaries`)."""
    return _find_beside_binaries(binaries, _LLAMA_SERVER_NAMES)


def build_run_command(
    gguf_path: Path, config: CandidateConfig, binaries: BinaryPaths
) -> list[str]:
    """Build a ready-to-run llama-cli invocation for the chosen config."""
    llama_cli = _find_llama_cli(binaries)
    return [
        llama_cli,
        "-m",
        str(gguf_path),
        "-ngl",
        str(config.ngl),
        "-c",
        str(config.ctx),
    ]


def build_server_command(
    gguf_path: Path, config: CandidateConfig, binaries: Optional[BinaryPaths] = None
) -> list[str]:
    """Build a ready-to-run llama-server (OpenAI-compatible API) invocation.

    ``binaries`` is optional so a caller holding only a SearchResult (no
    BinaryPaths) still gets a correct command, just with the bare
    ``llama-server`` name instead of a resolved absolute path.
    """
    llama_server = (
        _find_llama_server(binaries) if binaries is not None else _LLAMA_SERVER_NAMES[0]
    )
    return [
        llama_server,
        "-m",
        str(gguf_path),
        "-ngl",
        str(config.ngl),
        "-c",
        str(config.ctx),
        "--port",
        "8080",
    ]


def export_ollama_modelfile(gguf_path: Path, config: CandidateConfig) -> Path:
    """Write an Ollama ``Modelfile`` next to ``gguf_path``; return its path.

    Format verified against Ollama's own documentation on 2026-08-02:
      - https://docs.ollama.com/modelfile -- ``PARAMETER <parameter>
        <parametervalue>`` syntax, ``num_ctx`` = "size of the context
        window", and "The GGUF file location should be specified as an
        absolute path or relative to the Modelfile location" (hence the
        relative ``FROM ./<name>``: the whole --out dir stays relocatable).
      - ``num_gpu`` is not listed on that page; confirmed the same day
        against the API option a PARAMETER maps to, ``Runner.NumGPU
        `json:"num_gpu"``` in
        https://github.com/ollama/ollama/blob/main/api/types.go

    Written atomically (temp file + os.replace) so an interrupted run can
    never leave a half-written Modelfile that `ollama create` would read;
    the temp file is removed if the write fails.
    """
    target = gguf_path.parent / "Modelfile"
    content = (
        f"FROM ./{gguf_path.name}\n"
        f"PARAMETER num_gpu {config.ngl}\n"
        f"PARAMETER num_ctx {config.ctx}\n"
    )
    tmp = target.with_name("Modelfile.tmp")
    try:
        tmp.write_text(content, encoding="utf-8")
        os.replace(tmp, target)
    except OSError as exc:
        try:
            tmp.unlink()
        except OSError:  # pragma: no cover - nothing to clean up
            pass
        raise FiTunaError(f"could not write Ollama Modelfile to {target}: {exc}") from exc
    return target.resolve()


def _to_jsonable(obj):
    """Recursively convert dataclasses/Path into plain JSON-safe values."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    return obj


def to_json(result: SearchResult) -> str:
    """Serialize SearchResult (and nested dataclasses) to a JSON string."""
    return json.dumps(_to_jsonable(result), indent=2, ensure_ascii=False)


def human_size(num_bytes: float) -> str:
    """Decimal (SI) file size, matching how model files are usually quoted."""
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1000:
            return f"{num_bytes:.0f} {unit}" if unit == "B" else f"{num_bytes:.1f} {unit}"
        num_bytes /= 1000
    return f"{num_bytes:.1f} TB"


def _artifact_line(gguf_path: Path) -> str:
    """The headline of the result: the file the search already produced.

    The size is looked up on disk; if the file is gone (or unreadable), the
    line is still printed without it rather than blowing up a report.
    """
    try:
        size = f"{human_size(gguf_path.stat().st_size)} -- "
    except OSError:
        size = ""
    return f"  artifact: {gguf_path}  ({size}already produced during the search)"


def to_human(result: SearchResult) -> str:
    """Render a short human-readable summary: config, throughput, quality
    loss, meets_target, and -- leading the exits -- the produced GGUF plus
    the three ways to consume it (llama-server / Ollama / llama-cli)."""
    cfg = result.config
    bench = result.bench
    quality = result.quality
    status = "MEETS TARGET" if result.meets_target else "BEST EFFORT (target not met)"

    lines = [
        f"FiTuna result: {status}",
        "",
        f"  quant           : {cfg.quant}",
        f"  ngl             : {cfg.ngl}",
        f"  ctx             : {cfg.ctx}",
        "",
        f"  prompt tok/s (pp): {bench.prompt_tok_per_sec:.2f}",
        f"  gen tok/s    (tg): {bench.gen_tok_per_sec:.2f}",
    ]
    if bench.vram_used_mb is not None:
        lines.append(f"  vram used       : {bench.vram_used_mb} MB")
    quality_lines = [
        f"  perplexity      : {quality.perplexity:.4f} "
        f"(baseline {quality.baseline_perplexity:.4f})",
        f"  quality loss    : {quality.quality_loss_pct:.2f}%",
    ]
    if quality.kld is not None:
        quality_lines.append(f"  kld             : {quality.kld:.6f}")

    lines += [
        "",
        *quality_lines,
        "",
        _artifact_line(result.gguf_path),
        "",
    ]

    # Same order for MEETS TARGET and BEST EFFORT: the file is the
    # deliverable either way, and llama-cli is a check, not the destination.
    server_cmd = result.llama_server_command or build_server_command(
        result.gguf_path, cfg
    )
    lines += [
        "  1) local API server (OpenAI-compatible):",
        f"       {' '.join(server_cmd)}",
    ]
    if server_cmd[0] == _LLAMA_SERVER_NAMES[0]:
        # Bare name -- but *why* differs by which branch produced server_cmd,
        # and only one of them actually looked:
        if result.llama_server_command is not None:
            # search() called build_server_command() with real BinaryPaths,
            # i.e. it checked beside the known binaries and on PATH and came
            # up empty. "not found" is a checked fact here.
            lines.append(
                "       (llama-server was not found on this machine -- the "
                "command shows the bare name)"
            )
        else:
            # This is to_human()'s own fallback (no BinaryPaths available to
            # search with), which never looked anywhere -- claiming "not
            # found" would assert a check that never happened.
            lines.append(
                "       (no binary-location info available for this report "
                "-- the command shows the bare name, unchecked)"
            )
    if result.modelfile_path is not None:
        lines += [
            "  2) import into Ollama:",
            f"       ollama create <name> -f {result.modelfile_path}",
        ]
    else:
        lines.append(
            "  2) import into Ollama: re-run with --export-ollama to write a "
            "Modelfile beside the artifact"
        )
    lines += [
        "  3) terminal chat (interactive check):",
        f"       {' '.join(result.run_command)}",
    ]
    return "\n".join(lines)


def _self_check() -> None:
    """Minimal assert-based sanity check for this module's core contract."""
    cand = CandidateConfig(quant="Q4_K_M", ngl=20, ctx=4096)

    from fituna.config import BenchResult, QualityResult

    bench = BenchResult(
        candidate=cand,
        prompt_tok_per_sec=123.45,
        gen_tok_per_sec=30.5,
        vram_used_mb=2048,
        raw_stdout="{}",
    )
    quality = QualityResult(
        candidate_quant="Q4_K_M",
        perplexity=6.15,
        baseline_perplexity=6.0,
        quality_loss_pct=2.5,
    )

    # 1. build_run_command: PATH-fallback branch when llama-cli sits nowhere
    #    we know about and isn't on PATH -- must still return a usable,
    #    non-empty command instead of raising. This must hold regardless of
    #    whatever the machine actually running this self-check happens to
    #    have installed (e.g. llama.cpp via Homebrew), so PATH is blanked
    #    for the duration of this one call rather than assumed empty.
    import os as _os

    binaries = BinaryPaths(
        llama_quantize=Path("/nonexistent/llama-quantize"),
        llama_bench=Path("/nonexistent/llama-bench"),
        llama_perplexity=Path("/nonexistent/llama-perplexity"),
    )
    saved_path = _os.environ.get("PATH")
    try:
        _os.environ["PATH"] = ""
        cmd = build_run_command(Path("out/model-Q4_K_M.gguf"), cand, binaries)
    finally:
        if saved_path is None:
            _os.environ.pop("PATH", None)
        else:
            _os.environ["PATH"] = saved_path
    assert cmd[0] == "llama-cli"  # bare-name fallback, no crash
    assert cmd == ["llama-cli", "-m", "out/model-Q4_K_M.gguf", "-ngl", "20", "-c", "4096"]

    # 2. build_run_command: finds llama-cli sitting next to a known binary.
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        (tdp / "llama-quantize").touch()
        (tdp / "llama-cli").touch()
        binaries2 = BinaryPaths(
            llama_quantize=tdp / "llama-quantize",
            llama_bench=tdp / "llama-bench",
            llama_perplexity=tdp / "llama-perplexity",
        )
        cmd2 = build_run_command(Path("m.gguf"), cand, binaries2)
        assert cmd2[0] == str(tdp / "llama-cli")

        # 2b. build_server_command resolves llama-server the same way, and
        #     without BinaryPaths it falls back to the bare name.
        (tdp / "llama-server").touch()
        srv = build_server_command(Path("m.gguf"), cand, binaries2)
        assert srv[0] == str(tdp / "llama-server")
        assert srv[1:] == ["-m", "m.gguf", "-ngl", "20", "-c", "4096", "--port", "8080"]
        assert build_server_command(Path("m.gguf"), cand)[0] == "llama-server"

        # 2c. export_ollama_modelfile: exact three-line content, relative FROM
        #     (so the out dir stays relocatable), no temp file left behind.
        gguf = tdp / "model-Q4_K_M.gguf"
        gguf.write_bytes(b"x" * 2500)
        mf = export_ollama_modelfile(gguf, cand)
        assert mf == (tdp / "Modelfile").resolve()
        assert mf.read_text(encoding="utf-8") == (
            "FROM ./model-Q4_K_M.gguf\nPARAMETER num_gpu 20\nPARAMETER num_ctx 4096\n"
        )
        assert not (tdp / "Modelfile.tmp").exists()

    result = SearchResult(
        config=cand,
        bench=bench,
        quality=quality,
        gguf_path=Path("out/model-Q4_K_M.gguf"),
        run_command=cmd,
        meets_target=True,
    )

    # 3. to_json: round-trips through json.loads, Path becomes a plain str,
    #    nested dataclasses become nested dicts.
    js = to_json(result)
    parsed = json.loads(js)
    assert parsed["gguf_path"] == "out/model-Q4_K_M.gguf"
    assert parsed["config"]["quant"] == "Q4_K_M"
    assert parsed["bench"]["gen_tok_per_sec"] == 30.5
    assert parsed["quality"]["quality_loss_pct"] == 2.5
    assert parsed["meets_target"] is True
    assert parsed["run_command"] == cmd

    # 3b. the artifact-exit JSON fields are present, additive, and null until
    #     --export-ollama runs.
    assert parsed["llama_server_command"] is None
    assert parsed["modelfile_path"] is None
    with_exits = dataclasses.replace(
        result,
        llama_server_command=build_server_command(result.gguf_path, cand),
        modelfile_path=Path("/tmp/out/Modelfile"),
    )
    parsed2 = json.loads(to_json(with_exits))
    assert parsed2["llama_server_command"][0] == "llama-server"
    assert parsed2["modelfile_path"] == str(Path("/tmp/out/Modelfile"))
    assert parsed2["run_command"] == cmd  # existing fields untouched

    # 4. to_human: key numbers, and the artifact leading the three exits in
    #    order (server, Ollama, llama-cli-as-check).
    human = to_human(result)
    assert "MEETS TARGET" in human
    assert "Q4_K_M" in human
    assert "30.50" in human
    assert " ".join(cmd) in human
    artifact_at = human.index("artifact:")
    assert artifact_at < human.index("1) local API server")
    assert human.index("1) local API server") < human.index("2) import into Ollama")
    assert human.index("2) import into Ollama") < human.index("3) terminal chat")
    assert "--export-ollama" in human  # hint shown when nothing was exported
    assert "/tmp/out/Modelfile" not in human
    human2 = to_human(with_exits)
    assert f"ollama create <name> -f {Path('/tmp/out/Modelfile')}" in human2
    assert "--export-ollama" not in human2

    # 5. best-effort (meets_target=False) must say so, not claim success --
    #    and still lead with the artifact + the same three exits.
    result_bad = dataclasses.replace(result, meets_target=False)
    human_bad = to_human(result_bad)
    assert "BEST EFFORT" in human_bad
    assert human_bad.index("artifact:") < human_bad.index("1) local API server")
    assert json.loads(to_json(result_bad))["meets_target"] is False

    # 6. human_size: decimal units, matching how model files are quoted.
    assert human_size(999) == "999 B"
    assert human_size(2_300_000_000) == "2.3 GB"


if __name__ == "__main__":
    _self_check()
    print("fituna.report self-check OK")
