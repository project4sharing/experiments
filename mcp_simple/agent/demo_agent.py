import asyncio
import mcp_use
from mcp_use import MCPAgent, MCPClient
from langchain_openai import ChatOpenAI

# Level 2 shows exactly what the Server says to Stderr if it crashes
mcp_use.set_debug(2)

async def main():
    # 1. Setup Local LLM (pointing to your port 8080)
    llm = ChatOpenAI(
        base_url="http://localhost:8080/v1",
        api_key="local-llm",
        model="local-model"
    )

    # 2. Configuration
    # Ensure the 'command' is 'python' and 'args' points to your filename
    mcp_config = {
        "mcpServers": {
            "weather": {
                "command": "python",
                "args": ["./server/stdio_weather_server.py"]
            }
        }
    }

    print("\n--- PHASE 1: INITIALIZING CLIENT (Handshake & Discovery) ---")
    # This is where the 'initialize' and 'tools/list' steps happen
    client = MCPClient(config=mcp_config)

    print("\n--- PHASE 2: INITIALIZING AGENT (LLM Integration) ---")
    agent = MCPAgent(llm=llm, client=client)

    # 3. Solve the problem
    query = "Is it raining in London? Use your weather tool to find out."
    print(f"\n--- EXECUTING: {query} ---")
    
    try:
        result = await agent.run(query)
        print("\n[LLM Response]:")
        print(result)
    except Exception as e:
        print(f"\n[Error during execution]: {e}")

if __name__ == "__main__":
    asyncio.run(main())