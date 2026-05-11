"""
clients/mcp_use_client.py
──────────────────────────
Thin wrapper around mcp-use that wires your local LLM (port 8080)
to any MCP server configuration.

Provides:
  • build_llm()         — ChatOpenAI pointed at localhost:8080
  • build_agent()       — MCPAgent ready to run tasks
  • run_task()          — one-shot task execution
  • list_tools()        — enumerate tools exposed by a server
  • call_tool()         — call a specific tool and return raw result
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any, Optional

from langchain_openai import ChatOpenAI
from mcp_use import MCPAgent, MCPClient

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

log = logging.getLogger(__name__)


# ── LLM factory ──────────────────────────────────────────────────────────────

def build_llm(
    model: Optional[str] = None,
    temperature: float = 0.0,
    timeout: Optional[int] = None,
) -> ChatOpenAI:
    """
    Returns a ChatOpenAI instance pointed at your local LLM.

    Works with any OpenAI-compatible server:
      - Ollama        (ollama serve          → default :11434, or proxy to :8080)
      - LM Studio     (Local Server tab      → :8080)
      - llama.cpp     (llama-server          → :8080)
      - vLLM          (python -m vllm.entrypoints.openai.api_server → :8080)
      - Jan.ai        (API Server settings   → :8080)
      - text-gen-webui (--api --listen-port 8080)
    """
    return ChatOpenAI(
        model=model or config.LOCAL_LLM_MODEL,
        base_url=config.LOCAL_LLM_BASE_URL,
        api_key=config.LOCAL_LLM_API_KEY,
        temperature=temperature,
        timeout=timeout or config.LOCAL_LLM_TIMEOUT,
        max_retries=2,
    )


# ── Client factory ────────────────────────────────────────────────────────────

def build_client(server_config: dict) -> MCPClient:
    """
    Build an MCPClient from a server config dict.

    Example:
        client = build_client({
            "mcpServers": {
                "my-server": {"url": "http://localhost:8000/mcp"}
            }
        })
    """
    # print(f"INside build_client()::{server_config}")
    return MCPClient(server_config, verify=False)


# ── Agent factory ─────────────────────────────────────────────────────────────

def build_agent(
    server_config: dict,
    system_prompt: Optional[str] = None,
    max_steps: int = 10,
    disallowed_tools: Optional[list[str]] = None,
    verbose: bool = False,
) -> MCPAgent:
    """
    Build an MCPAgent wired to your local LLM.

    Args:
        server_config:     mcpServers dict (see config.mcp_server_config)
        system_prompt:     Optional override for the agent system prompt
        max_steps:         Max tool call iterations before giving up
        disallowed_tools:  Tool names the agent must never call
        verbose:           Print LangChain chain trace
    """
    llm    = build_llm()
    client = build_client(server_config)
    return MCPAgent(
        llm=llm,
        client=client,
        system_prompt=system_prompt,
        max_steps=max_steps,
        disallowed_tools=disallowed_tools or [],
        verbose=verbose,
        memory_enabled=True,
    )


# ── One-shot task runner ──────────────────────────────────────────────────────

async def run_task(
    task: str,
    server_config: dict,
    system_prompt: Optional[str] = None,
    max_steps: int = 10,
    disallowed_tools: Optional[list[str]] = None,
    verbose: bool = False,
) -> str:
    """
    Connect to an MCP server, run a task with the local LLM, return the result.

    Example:
        result = await run_task(
            "What is the current time?",
            config.mcp_server_config("clean_http"),
        )
    """
    agent = build_agent(
        server_config=server_config,
        system_prompt=system_prompt,
        max_steps=max_steps,
        disallowed_tools=disallowed_tools,
        verbose=verbose,
    )
    log.info("Running task: %s", task[:80])
    result = await agent.run(task)
    return result


# ── Tool enumeration ──────────────────────────────────────────────────────────

async def list_tools(server_config: dict) -> list[dict]:
    """
    Connect to an MCP server and return its tools/list result.
    Does NOT involve the LLM — pure protocol-level call.
    """
    # print(f"Inside list_tools:{server_config}")
    client = build_client(server_config)
    server_name = next(iter(server_config.get("mcpServers", {})), None)
    # print(f"Inside list_tools, server_name:{server_name}")
    if not server_name:
        raise ValueError("server_config must contain at least one mcpServers entry")

    session = await client.create_session(server_name)
    tools = await session.list_tools()
    # print(f"Inside list_tools, tools={tools}")
    await client.close_session(server_name)
    return [t.model_dump() if hasattr(t, "model_dump") else t for t in tools]


# ── Direct tool call ──────────────────────────────────────────────────────────

async def call_tool(
    tool_name: str,
    arguments: dict,
    server_config: dict,
) -> Any:
    """
    Call a specific tool on an MCP server directly (no LLM routing).
    Returns the raw content list from the tool response.
    """
    client = build_client(server_config)
    server_name = next(iter(server_config.get("mcpServers", {})))
    session = await client.create_session(server_name)
    result = await session.call_tool(tool_name, arguments)
    await client.close_session(server_name)
    return result.content


# ── CLI smoke test ────────────────────────────────────────────────────────────

async def _smoke_test():
    """Quick end-to-end check: local LLM reachable + tools listed."""
    import httpx
    from rich.console import Console
    c = Console()

    c.rule("[bold cyan]mcp-use client smoke test[/]")
    c.print(f"  LLM endpoint : [yellow]{config.LOCAL_LLM_BASE_URL}[/]")
    c.print(f"  Model        : [yellow]{config.LOCAL_LLM_MODEL}[/]")

    # 1. Check LLM health
    c.print("\n[bold]1. LLM reachability[/]")
    try:
        async with httpx.AsyncClient() as hx:
            resp = await hx.get(
                f"{config.LOCAL_LLM_BASE_URL.rstrip('/v1').rstrip('/')}/health",
                timeout=5.0,
            )
        c.print(f"  /health → [green]{resp.status_code}[/]")
    except Exception as e:
        c.print(f"  [yellow]Health check failed (may be normal): {e}[/]")

    # 2. List models
    c.print("\n[bold]2. Available models[/]")
    try:
        llm = build_llm()
        async with httpx.AsyncClient() as hx:
            resp = await hx.get(
                f"{config.LOCAL_LLM_BASE_URL}/models",
                headers={"Authorization": f"Bearer {config.LOCAL_LLM_API_KEY}"},
                timeout=5.0,
            )
            models = resp.json().get("data", [])
            for m in models[:5]:
                c.print(f"  • {m.get('id', m)}")
    except Exception as e:
        c.print(f"  [yellow]Could not list models: {e}[/]")

    # 3. Tool listing (clean server must be running)
    c.print("\n[bold]3. Tool listing via mcp-use (requires clean_http server on :8000)[/]")
    try:
        import config as cfg
        # print(f"cfg::{cfg.mcp_server_config("clean_http")}")
        test_config = {
            'mcpServers': {
                'clean_http': {
                    'url': 'http://127.0.0.1:8000/mcp',
                    'auth': None
                }
            }
        }
        # tools = await list_tools(cfg.mcp_server_config("clean_http"))
        tools = await list_tools(test_config)
        # print(f"tools:{tools}")
        for t in tools:
            c.print(f"  • [cyan]{t['name']}[/] — {t.get('description','')[:60]}")
        c.print(f"  [green]✓ {len(tools)} tool(s) found[/]")
    except Exception as e:
        c.print(f"  [yellow]Skipped (server not running?): {e}[/]")

    c.rule("[green]Smoke test complete[/]")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(_smoke_test())