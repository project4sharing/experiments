"""
config.py
──────────
Central configuration for the MCP guardrail tester.
"""

import os
from pathlib import Path
import toml


# SSE_URL  = f"http://localhost:{PORTS['clean_sse']}/sse"
conf = toml.load(f"{str(Path(__file__).parent)}/conf/mcp_simple.toml")
mcpServers = conf["mcpServers"]

def get_server_config(*server_keys: str) -> dict:
    """Return an mcpServers dict for multiple servers."""
    return {"mcpServers": {k: mcpServers[k] for k in mcpServers if k in server_keys}}