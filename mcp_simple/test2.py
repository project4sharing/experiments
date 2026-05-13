import httpx
import os
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("WeatherService")

# Enterprise Proxy Configuration
PROXY_URL = "http://localhost:9000"

@mcp.tool()
async def get_weather(location: str) -> str:
    """Get real-time weather for any location (city or landmark)."""
    query = location.strip().replace(" ", "+")
    url = f"https://wttr.in/{query}?format=j1"
    
    # Configure the client to route through the Zscaler proxy
    # 'verify=True' is standard, but if you get SSL errors, see the tip below.
    async with httpx.AsyncClient(proxy=PROXY_URL, verify=True, timeout=10.0) as client:
        try:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()
            
            current = data['current_condition'][0]
            temp = current['temp_C']
            desc = current['weatherDesc'][0]['value']
            return f"The weather in {location} is {desc} at {temp}°C."
        except httpx.ProxyError:
            return "Error: Could not connect to the proxy at localhost:9000."
        except Exception as e:
            return f"Could not get weather for '{location}': {str(e)}"

if __name__ == "__main__":
    mcp.run()