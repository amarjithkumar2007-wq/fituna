# SPDX-License-Identifier: MIT
"""fituna.mcp_server
====================

A Model Context Protocol (MCP) server exposing FiTuna to AI agents, so an
agent asked "which local model config fits this machine?" can answer with a
*measurement* instead of a guess.

Implemented on the stdlib only: the MCP stdio transport is newline-delimited
JSON-RPC 2.0, which ``json`` + ``sys.stdin`` cover fully. No SDK dependency
-- FiTuna's zero-runtime-dependency guarantee extends to the MCP server.

Run: ``fituna-mcp`` (console script) or ``python -m fituna.mcp_server``.

Register with an MCP client (e.g. Claude Code):

    claude mcp add fituna -- fituna-mcp

Tools exposed:

- ``fituna_detect_hardware`` -- auto-detected GPU/VRAM/CPU/RAM profile.
- ``fituna_recommend`` -- run the measured search for a target spec and
  return the winning config + run command. Uses the sqlite3 cache, so
  repeat questions about the same model/hardware answer in about a second.

Protocol notes: implements ``initialize``, ``tools/list``, ``tools/call``;
ignores notifications; answers unknown methods with JSON-RPC error -32601.
Requests are handled sequentially -- a search can take minutes, which is
fine: MCP calls are allowed to be slow, and the cache makes them fast the
second time.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from fituna import __version__, binaries, hardware, model_info, report, search
from fituna.cache import ResultCache
from fituna.config import FiTunaError, NoFeasibleConfigError, TargetSpec, check_target_ranges

PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {"name": "fituna", "version": __version__}
_DEFAULT_QUANTS = ("Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M")
_BASE_GGUF_RE = re.compile(r"(?:^|[-_.])(?:bf16|f16|fp16|f32|fp32)$", re.IGNORECASE)

_TOOLS: list[dict[str, Any]] = [
    {
        "name": "fituna_detect_hardware",
        "description": (
            "Auto-detect the local machine's GPU vendor/name, VRAM, CPU "
            "cores, and RAM as FiTuna sees them. Use this before "
            "fituna_recommend to show the user what hardware the "
            "measurement will run on."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "fituna_recommend",
        "description": (
            "Find the llama.cpp config (quantization level, GPU offload "
            "layers, context length) that meets a target generation speed "
            "within a quality-loss budget -- the highest-quality quant that "
            "passes, with the fewest GPU layers it needs -- by actually benchmarking "
            "on this machine, not by guessing from specs. Slow on first run "
            "(minutes: it quantizes and benchmarks real candidates); "
            "near-instant on repeat runs thanks to the result cache. "
            "Returns the chosen config, measured tok/s, measured quality "
            "loss, and a ready-to-run llama-cli command."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "model_path": {
                    "type": "string",
                    "description": "Path to an F16/BF16 .gguf file (defaults to the only base-precision GGUF in out_dir)",
                },
                "target_tps": {
                    "type": "number",
                    "description": "Target generation speed in tokens/second",
                },
                "max_quality_loss_pct": {
                    "type": "number",
                    "description": "Max acceptable quality loss in percent (relative perplexity increase vs the unquantized baseline)",
                    "default": 5.0,
                },
                "wikitext_path": {
                    "type": "string",
                    "description": "Path to a plain-text perplexity corpus (default: ./wiki.txt)",
                },
                "out_dir": {
                    "type": "string",
                    "description": "Working directory for quantized files and the result cache",
                    "default": "./out",
                },
                "ctx": {
                    "type": "integer",
                    "description": "Context length to validate at",
                    "default": 4096,
                },
                "quant_candidates": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Quant levels to consider",
                    "default": list(_DEFAULT_QUANTS),
                },
                "ppl_chunks": {
                    "type": "integer",
                    "description": "Perplexity corpus chunks to evaluate (default 32; 0 = full corpus, much slower)",
                    "default": 32,
                },
            },
            "required": ["target_tps"],
        },
    },
]


def _detect_hardware() -> dict[str, Any]:
    hw = hardware.detect_hardware()
    payload = asdict(hw)
    payload["gpu_vendor"] = hw.gpu_vendor.value
    return payload


def _resolve_paths(args: dict[str, Any]) -> tuple[Path, Path, Path]:
    work_dir = Path(args.get("out_dir") or "./out")
    if args.get("model_path"):
        model_path = Path(args["model_path"])
    else:
        models = [
            p for p in work_dir.glob("*.gguf")
            if p.is_file() and _BASE_GGUF_RE.search(p.stem)
        ]
        if not models:
            raise FiTunaError(f"no base GGUF file found in {work_dir}; pass model_path")
        if len(models) > 1:
            raise FiTunaError(f"multiple base GGUF files found in {work_dir}; pass model_path")
        model_path = models[0]
    return work_dir, model_path, Path(args.get("wikitext_path") or "./wiki.txt")


def _recommend(args: dict[str, Any]) -> dict[str, Any]:
    try:
        target_tps = float(args["target_tps"])
    except (KeyError, TypeError, ValueError) as exc:
        raise FiTunaError("target_tps must be a number") from exc
    max_loss = float(args.get("max_quality_loss_pct") or 5.0)
    ctx = int(args.get("ctx") or 4096)
    # fail before model conversion, not after (TargetSpec re-checks anyway)
    check_target_ranges(target_tps, max_loss, (ctx,))

    bins = binaries.locate_binaries()
    hw = hardware.detect_hardware()

    work_dir, model_path, wikitext_path = _resolve_paths(args)
    work_dir.mkdir(parents=True, exist_ok=True)

    base_gguf = model_info.ensure_base_gguf(model_path, work_dir, bins)
    minfo = model_info.read_model_info(base_gguf, bins)

    quants = args.get("quant_candidates") or _DEFAULT_QUANTS
    ppl_chunks_raw = args.get("ppl_chunks", 32)
    ppl_chunks: Optional[int] = int(ppl_chunks_raw) if int(ppl_chunks_raw) > 0 else None

    target = TargetSpec(
        model_path=model_path,
        target_tokens_per_sec=target_tps,
        max_quality_loss_pct=max_loss,
        ctx=ctx,
        ctx_candidates=[ctx],
        quant_candidates=list(quants),
        ppl_chunks=ppl_chunks,
    )

    # Always cache: an agent asking twice should pay the benchmark once.
    cache = ResultCache(work_dir / ".fituna_cache.sqlite3")
    try:
        result = search.search(
            target, minfo, hw, bins, work_dir,
            wikitext_path, cache=cache,
        )
        payload = json.loads(report.to_json(result))
        payload["already_quantized_warning"] = model_info.is_already_quantized(minfo)
        return payload
    except NoFeasibleConfigError as exc:
        payload = {
            "meets_target": False,
            "error": str(exc),
        }
        if exc.closest is not None:
            payload["closest_best_effort"] = json.loads(report.to_json(exc.closest))
        return payload
    finally:
        cache.close()


def _tool_call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Dispatch one tools/call. Returns an MCP content payload."""
    try:
        if name == "fituna_detect_hardware":
            data = _detect_hardware()
        elif name == "fituna_recommend":
            data = _recommend(arguments)
        else:
            return {
                "content": [{"type": "text", "text": f"unknown tool: {name}"}],
                "isError": True,
            }
        return {"content": [{"type": "text", "text": json.dumps(data, ensure_ascii=False, default=str)}]}
    except FiTunaError as exc:
        # Tool-level failure (missing binary, bad model file, ...) -- report
        # inside the result so the agent can relay it, not as a protocol error.
        return {"content": [{"type": "text", "text": str(exc)}], "isError": True}


def _error(msg_id: Any, code: int, message: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": msg_id,
        "error": {"code": code, "message": message},
    }


def _handle(msg: object) -> Optional[dict[str, Any]]:
    """Handle one JSON-RPC message; return the response, or None for
    notifications (no id -> no response, per JSON-RPC 2.0)."""
    if not isinstance(msg, dict):
        return _error(None, -32600, "invalid request")

    method = msg.get("method")
    msg_id = msg.get("id")

    if msg_id is None:  # notification (e.g. notifications/initialized)
        return None

    if method == "initialize":
        result: dict[str, Any] = {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": SERVER_INFO,
        }
    elif method == "tools/list":
        result = {"tools": _TOOLS}
    elif method == "tools/call":
        params = msg.get("params")
        if params is None:
            params = {}
        if not isinstance(params, dict):
            return _error(msg_id, -32602, "invalid params")

        arguments = params.get("arguments")
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            return _error(msg_id, -32602, "invalid params")
        result = _tool_call(params.get("name", ""), arguments)
    elif method == "ping":
        result = {}
    else:
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {"code": -32601, "message": f"method not found: {method}"},
        }

    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def serve(stdin=None, stdout=None) -> None:
    """Blocking newline-delimited JSON-RPC loop over stdio.

    ``stdin``/``stdout`` injectable for tests; defaults to the process
    streams. Malformed JSON lines get a -32700 parse error (with null id,
    per spec) instead of killing the server.
    """
    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout

    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            response: Optional[dict[str, Any]] = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": "parse error"},
            }
        else:
            response = _handle(msg)
        if response is not None:
            stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
            stdout.flush()


def main() -> None:
    serve()


def _self_check() -> None:
    """Runnable protocol-level check: initialize -> tools/list -> a bad
    tools/call -> notification -> malformed JSON, all through serve() with
    fake streams. No llama.cpp needed."""
    import io

    requests = "\n".join([
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}),
        json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
        json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
        json.dumps({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                    "params": {"name": "nope", "arguments": {}}}),
        json.dumps({"jsonrpc": "2.0", "id": 4, "method": "no/such/method"}),
        "this is not json",
    ]) + "\n"

    out = io.StringIO()
    serve(stdin=io.StringIO(requests), stdout=out)
    responses = [json.loads(l) for l in out.getvalue().splitlines()]

    # 5 responses: the notification produced none.
    assert len(responses) == 5, responses

    init, tools, badtool, nomethod, parse_err = responses
    assert init["id"] == 1 and init["result"]["protocolVersion"] == PROTOCOL_VERSION
    assert init["result"]["serverInfo"]["name"] == "fituna"

    names = [t["name"] for t in tools["result"]["tools"]]
    assert names == ["fituna_detect_hardware", "fituna_recommend"], names
    for t in tools["result"]["tools"]:
        assert t["inputSchema"]["type"] == "object"

    assert badtool["result"]["isError"] is True
    assert nomethod["error"]["code"] == -32601
    assert parse_err["error"]["code"] == -32700 and parse_err["id"] is None

    # detect_hardware runs for real (pure local introspection, no llama.cpp).
    hw_payload = _detect_hardware()
    assert "gpu_vendor" in hw_payload and "ram_mb" in hw_payload


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selfcheck":
        _self_check()
        print("fituna.mcp_server self-check OK")
    else:
        main()
