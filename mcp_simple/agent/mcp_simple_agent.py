"""
agent/mcp_simple_agent.py
──────────────────────────
Thin wrapper around mcp-use that wires LLM
to any MCP server configuration.

Provides:
  • build_llm()         — ChatOpenAI pointed at localhost:8080
  • build_agent()       — MCPAgent ready to run tasks
  • run_task()          — one-shot task execution
  • list_tools()        — enumerate tools exposed by a server
  • call_tool()         — call a specific tool and return raw result
"""
import sys
from pathlib import Path
import logging
import toml
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from client.mcp_simple_client import *
from langchain_openai import ChatOpenAI
from mcp_use import MCPAgent, MCPClient

# ── LLM factory ──────────────────────────────────────────────────────────────

log = logging.getLogger(__name__)

conf = toml.load(f"{str(Path(__file__).parent.parent)}/conf/mcp_simple.toml")
print(conf["mcpServers"])
print(type(conf["mcpServers"]))
mcpServers = conf["mcpServers"]



def get_llm(
    model: Optional[str] = None,
    temperature: float = 0.0,
    timeout: Optional[int] = None,
) -> ChatOpenAI:
    """
    Returns a ChatOpenAI instance pointed at your LLM.
    """
    return ChatOpenAI(
        model=conf["MODEL"]["LOCAL_LLM_MODEL"],
        base_url=conf["MODEL"]["LOCAL_MODEL_BASE_URL"],
        api_key=conf["MODEL"]["LOCAL_LLM_API_KEY"],
        temperature=temperature,
        timeout=conf["MODEL"]["LOCAL_LLM_TIMEOUT"],
        max_retries=2,
    )

def get_agent(
    system_prompt:    Optional[str]   = None,
    max_steps:        int             = 10,
    disallowed_tools: Optional[list[str]] = None,
    verbose:          bool            = False,
) -> MCPAgent:
    """
    Build an MCPAgent without running the guardrail validator.
    Use initialize_with_guardrail() instead when connecting to untrusted servers.
    """
    llm    = get_llm()
    print(f"mcpServers from conf: {conf["mcpServers"]}")
    client = get_client(conf["mcpServers"])
    return MCPAgent(
        llm=llm,
        client=client,
        system_prompt=system_prompt,
        max_steps=max_steps,
        disallowed_tools=disallowed_tools or [],
        verbose=verbose,
        memory_enabled=True,
        auto_initialize=True
    )

async def main():
    agent = get_agent()

    result = await agent.run(
        "Run the server_filesystem mcp server to list the content of C:/Users/petelam/Downloads"
    )
    print(f"\nResult: {result}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())