"""
config.py
──────────
Central configuration for the MCP guardrail tester.
Edit LOCAL_LLM_* to match your local LLM setup.
"""

import os

# ── Local LLM ─────────────────────────────────────────────────────────────────
# Most local LLM servers (Ollama, LM Studio, llama.cpp, vLLM, Jan.ai)
# expose an OpenAI-compatible API at /v1. Adjust as needed.

LOCAL_LLM_BASE_URL  = os.getenv("LOCAL_LLM_BASE_URL",  "http://localhost:8080/v1")
LOCAL_LLM_API_KEY   = os.getenv("LOCAL_LLM_API_KEY",   "not-needed")   # dummy key accepted by most servers
LOCAL_LLM_MODEL     = os.getenv("LOCAL_LLM_MODEL",     "local-model")  # any string — sent in the request
LOCAL_LLM_TIMEOUT   = int(os.getenv("LOCAL_LLM_TIMEOUT", "120"))       # seconds

# ── MCP server ports ──────────────────────────────────────────────────────────
PORTS = {
    "clean_http":    8000,
    # "clean_sse":     8001,
    # "poisoned_tool": 8002,
    # "goal_drift":    8003,
}

MCP_URLS = {k: f"http://localhost:{v}/mcp" for k, v in PORTS.items()}
# SSE_URL  = f"http://localhost:{PORTS['clean_sse']}/sse"

# ── MCPClient server config dicts (passed to MCPClient.from_dict) ─────────────

def mcp_server_config(server_key: str) -> dict:
    """Return an mcpServers dict for a single server."""
    return {"mcpServers": {server_key: {"url": MCP_URLS[server_key]}}}


def multi_server_config(*server_keys: str) -> dict:
    """Return an mcpServers dict for multiple servers."""
    return {"mcpServers": {k: {"url": MCP_URLS[k]} for k in server_keys}}