import asyncio
import logging
import mcp_use
from mcp_use import MCPAgent, MCPClient
from langchain_openai import ChatOpenAI
from pathlib import Path
import toml
from typing import Any, Optional

# Show the handshake for both servers
mcp_use.set_debug(2)

async def main():
    conf = toml.load(f"{str(Path(__file__).parent.parent)}/conf/mcp_simple.toml")

    llm = ChatOpenAI(
        base_url=conf["MODEL"]["LOCAL_MODEL_BASE_URL"],
        api_key=conf["MODEL"]["LOCAL_MODEL_API_KEY"],
        model=conf["MODEL"]["LOCAL_MODEL"]
    )

    # DEFINE MULTIPLE SERVERS
    mcp_config = {
        "mcpServers": {
            "knowledge": {
                "url": "http://127.0.0.1:8001/mcp",
                "auth": None
            },
            "weather": {
                "command": "uv",
                "args": ["run", "./server/stdio_weather_server.py"]
            }
        }
    }

    print("\n--- INITIALIZING MULTI-SERVER CLIENT ---")
    client = MCPClient(config=mcp_config)

    print("\n--- INITIALIZING AGENT ---")
    agent = MCPAgent(llm=llm, client=client)

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