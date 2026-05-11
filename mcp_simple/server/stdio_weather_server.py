import httpx
import sys
from mcp.server.fastmcp import FastMCP

# Initialize the server
mcp = FastMCP("WeatherService")

@mcp.tool()
async def get_weather(location: str) -> str:
    """Get real-time weather for any location (city or landmark)."""
    # Note: We use wttr.in/location?format=j1 for a clean JSON response
    query = location.strip().replace(" ", "+")
    url = f"https://wttr.in/{query}?format=j1"
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=10.0)
            response.raise_for_status()
            data = response.json()
            
            current = data['current_condition'][0]
            temp = current['temp_C']
            desc = current['weatherDesc'][0]['value']
            return f"The weather in {location} is {desc} at {temp}°C."
    except Exception as e:
        # Any errors returned here are sent back to the LLM as text
        return f"Could not get weather for '{location}': {str(e)}"

if __name__ == "__main__":
    # IMPORTANT: Do not use print() anywhere in this file. 
    # MCP uses stdout for data. Anything else crashes the initialization.
    mcp.run()