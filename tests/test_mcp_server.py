import io
import json
from pathlib import Path

import pytest

from fituna import mcp_server
from fituna.config import FiTunaError


def _serve_one(message: object) -> dict:
    stdin = io.StringIO(json.dumps(message) + "\n")
    stdout = io.StringIO()
    mcp_server.serve(stdin, stdout)
    return json.loads(stdout.getvalue())


def test_demo_defaults_reuse_cli_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    model = out / "SmolLM2-135M-Instruct-f16.gguf"
    model.touch()
    (out / "other-f16-Q4_K_M.gguf").touch()
    (out / "not-a-file-f16.gguf").mkdir()
    corpus = tmp_path / "wiki.txt"
    corpus.touch()

    assert mcp_server._resolve_paths({}) == (
        Path("./out"),
        Path("./out") / model.name,
        Path("./wiki.txt"),
    )
    recommend = next(tool for tool in mcp_server._TOOLS if tool["name"] == "fituna_recommend")
    assert recommend["inputSchema"]["required"] == ["target_tps"]
    assert recommend["inputSchema"]["properties"]["quant_candidates"]["default"] == [
        "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M",
    ]


def test_demo_default_rejects_ambiguous_models(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    (out / "a-f16.gguf").touch()
    (out / "b-bf16.gguf").touch()

    with pytest.raises(FiTunaError, match="multiple base GGUF files"):
        mcp_server._resolve_paths({})


def test_explicit_paths_bypass_auto_discovery(tmp_path: Path):
    args = {
        "out_dir": str(tmp_path / "custom-out"),
        "model_path": str(tmp_path / "model.gguf"),
        "wikitext_path": str(tmp_path / "corpus.txt"),
    }

    assert mcp_server._resolve_paths(args) == tuple(map(Path, args.values()))


def test_rejects_non_object_request_without_stopping_server():
    response = _serve_one([])

    assert response["error"]["code"] == -32600


@pytest.mark.parametrize("params", [[], "bad"])
def test_rejects_non_object_params(params):
    response = _serve_one({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": params,
    })

    assert response["error"]["code"] == -32602


def test_rejects_non_object_tool_arguments():
    response = _serve_one({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": "fituna_recommend", "arguments": "bad"},
    })

    assert response["error"]["code"] == -32602


@pytest.mark.parametrize("arguments", [{}, {"target_tps": "fast"}])
def test_recommend_rejects_invalid_target_before_environment_checks(arguments):
    response = mcp_server._tool_call("fituna_recommend", arguments)

    assert response["isError"] is True
    assert "target_tps must be a number" in response["content"][0]["text"]


@pytest.mark.parametrize(
    "arguments, needle",
    [
        ({"target_tps": -1}, "--target-tps"),
        ({"target_tps": 20, "max_quality_loss_pct": 500}, "--max-quality-loss"),
        ({"target_tps": 20, "ctx": -1}, "--ctx"),
    ],
)
def test_recommend_rejects_out_of_range_target_before_environment_checks(
    monkeypatch, arguments, needle
):
    def _must_not_run(*a, **k):
        raise AssertionError("range check must come before locating binaries")

    monkeypatch.setattr(mcp_server.binaries, "locate_binaries", _must_not_run)
    response = mcp_server._tool_call("fituna_recommend", arguments)

    assert response["isError"] is True
    assert needle in response["content"][0]["text"]


def test_initialize_reports_the_package_version():
    import fituna

    response = _serve_one({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
    assert response["result"]["serverInfo"]["version"] == fituna.__version__
