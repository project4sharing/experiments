import asyncio
import logging
import mcp_use
from mcp_use import MCPAgent, MCPClient
from langchain_openai import ChatOpenAI
from pathlib import Path
import tomllib
from typing import Any, Optional

# Show the handshake for both servers
mcp_use.set_debug(2)

async def main():
    with open(f"{str(Path(__file__).parent.parent)}/conf/mcp_simple.toml", "rb") as f:
        conf = tomllib.load(f)

    llm = ChatOpenAI(
        base_url=conf["MODEL"]["LOCAL_LLM_BASE_URL"],
        api_key=conf["MODEL"]["LOCAL_LLM_API_KEY"],
        model=conf["MODEL"]["LOCAL_LLM_MODEL"]
    )

    print({'mcpServers': conf['mcpServers']})
    print("\n--- INITIALIZING MULTI-SERVER CLIENT ---")
    client = MCPClient(config={'mcpServers': conf['mcpServers']})

    print("\n--- INITIALIZING AGENT ---")
    agent = MCPAgent(llm=llm, client=client, verbose=True)

    # A complex problem requiring information from TWO different MCP servers
    query = (
        "Tell me about the capital city of France. "
        "Include its population and the current weather there."
    )
    
    print(f"\n--- RUNNING MULTI-TOOL QUERY ---")
    result = await agent.run(query)
    
    print("\n[Final Integrated Answer]:")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())