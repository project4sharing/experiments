#!/usr/bin/env python3
"""
servers/http_streamable_server.py
──────────────────────────────────
MCP server using the Streamable HTTP transport (POST /mcp endpoint).

This is the modern transport introduced in the MCP 2025-03-26 spec:
  • Client sends JSON-RPC requests via HTTP POST to /mcp
  • Server may respond immediately OR open a streaming response (chunked)
  • Optional GET /mcp for server-initiated SSE notifications

Purpose: Test that your guardrail correctly identifies and inspects
         MCP servers reached over HTTP (not just stdio).

Run:  uvicorn servers.http_streamable_server:app --port 8000
"""

import asyncio
import json
import time
import uuid
import logging
from typing import AsyncGenerator

from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(title="MCP Streamable HTTP Server", version="1.0.0")

# ── Tool definitions ─────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "fetch_weather",
        "description": "Returns mock weather data for a city. "
                       "This is a legitimate, benign tool for guardrail baseline testing.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
            },
            "required": ["city"],
        },
    },
    {
        "name": "slow_computation",
        "description": "Simulates a long-running computation. "
                       "Use this to verify your guardrail handles streaming HTTP responses.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "steps": {"type": "integer", "description": "Number of computation steps (1-10)"},
            },
            "required": ["steps"],
        },
    },
]


def execute_tool(name: str, arguments: dict) -> dict:
    if name == "fetch_weather":
        city = arguments.get("city", "Unknown")
        return {
            "content": [{"type": "text", "text": f"Weather in {city}: 22°C, partly cloudy."}],
            "isError": False,
        }
    elif name == "slow_computation":
        steps = min(int(arguments.get("steps", 3)), 10)
        # Non-streaming path — just returns after a short delay
        time.sleep(steps * 0.1)
        return {
            "content": [{"type": "text", "text": f"Computation finished after {steps} steps."}],
            "isError": False,
        }
    else:
        return {
            "content": [{"type": "text", "text": f"Unknown tool: {name}"}],
            "isError": True,
        }


# ── JSON-RPC helpers ─────────────────────────────────────────────────────────

def ok(req_id, result) -> dict:
    # print(f"results:{result}")
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def err(req_id, code: int, msg: str) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": msg}}


def dispatch(msg: dict) -> dict:
    """Handle a single JSON-RPC request and return the response dict."""
    method = msg.get("method")
    req_id = msg.get("id")
    params = msg.get("params", {})

    if method == "initialize":
        return ok(req_id, {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}, "streaming": True},
            "serverInfo": {"name": "http-streamable-server", "version": "1.0.0"},
        })

    elif method == "tools/list":
        # print("TOOL_LIST called")
        # print(f"TOOLS:{TOOLS}")
        return ok(req_id, {"tools": TOOLS})

    elif method == "tools/call":
        result = execute_tool(params.get("name", ""), params.get("arguments", {}))
        return ok(req_id, result)

    elif method and method.startswith("notifications/"):
        return None  # no response for notifications

    else:
        return err(req_id, -32601, f"Method not found: {method}")


# ── Streaming generator (for slow_computation demo) ─────────────────────────

async def stream_computation(req_id, steps: int) -> AsyncGenerator[str, None]:
    """Yields SSE-formatted JSON-RPC progress notifications then final result."""
    for i in range(1, steps + 1):
        await asyncio.sleep(0.3)
        notification = {
            "jsonrpc": "2.0",
            "method": "notifications/progress",
            "params": {"step": i, "total": steps, "message": f"Step {i}/{steps} complete"},
        }
        yield f"data: {json.dumps(notification)}\n\n"

    final = ok(req_id, {
        "content": [{"type": "text", "text": f"Streaming computation done ({steps} steps)."}],
        "isError": False,
    })
    yield f"data: {json.dumps(final)}\n\n"


# ── Routes ────────────────────────────────────────────────────────────────────

@app.post("/mcp")
async def mcp_endpoint(request: Request):
    """
    Main MCP endpoint — handles both batch and single JSON-RPC messages.
    Returns streaming response when the client requests it via Accept header.
    """
    # print(f"Inside mcp_endpoint")
    body = await request.json()
    accept = request.headers.get("accept", "application/json")

    log.info("POST /mcp  method=%s  streaming=%s",
             body.get("method"), "text/event-stream" in accept)

    # ── Streaming path: slow_computation with SSE ────────────────────────
    if (body.get("method") == "tools/call"
            and body.get("params", {}).get("name") == "slow_computation"
            and "text/event-stream" in accept):
        steps = body.get("params", {}).get("arguments", {}).get("steps", 3)
        return StreamingResponse(
            stream_computation(body.get("id"), steps),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ── Normal synchronous path ──────────────────────────────────────────
    response = dispatch(body)
    if response is None:
        return Response(status_code=204)
    # print(f"Inside: response={response}")
    return JSONResponse(content=response)


@app.get("/mcp")
async def mcp_sse_channel(request: Request):
    """
    Optional server-initiated notification channel (SSE GET).
    Clients subscribe here to receive server-push events.
    """
    async def event_stream():
        for _ in range(3):
            await asyncio.sleep(2)
            ping = {"jsonrpc": "2.0", "method": "notifications/ping", "params": {}}
            yield f"data: {json.dumps(ping)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )


@app.get("/health")
async def health():
    return {"status": "ok", "server": "http-streamable", "timestamp": time.time()}


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")