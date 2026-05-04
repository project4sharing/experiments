"""
config.py
──────────
Central configuration for the MCP guardrail tester.
"""

import os



LOCAL_LLM_BASE_URL  = "http://localhost:8080/v1"
LOCAL_LLM_API_KEY   = "not-needed"
LOCAL_LLM_MODEL     = "local-model"
LOCAL_LLM_TIMEOUT   = 120

mcpServers = [
    {
        'clean_http': {
            'url': 'http://localhost:8000/mcp',
            'auth': None
        }
    }
]


PORTS = {
    "clean_http":    8000,
    # "clean_sse":     8001,
    # "poisoned_tool": 8002,
    # "goal_drift":    8003,
}

MCP_URLS = {k: f"http://localhost:{v}/mcp" for k, v in PORTS.items()}
# SSE_URL  = f"http://localhost:{PORTS['clean_sse']}/sse"

def mcp_server_config(server_key: str) -> dict:
    """Return an mcpServers dict for a single server."""
    return {"mcpServers": {server_key: {"url": MCP_URLS[server_key]}}}


def multi_server_config(*server_keys: str) -> dict:
    """Return an mcpServers dict for multiple servers."""
    return {"mcpServers": {k: {"url": MCP_URLS[k]} for k in server_keys}}