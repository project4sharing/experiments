import httpx
import logging
from fastmcp import FastMCP
import os
from pathlib import Path
import toml

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
conf = toml.load(f"{str(Path(__file__).parent.parent)}/conf/mcp_simple.toml")

cfg = conf.get("OFFICE" if os.environ.get('HTTP_PROXY') else "NOT_OFFICE", {})


# Initialize FastMCP
mcp = FastMCP("KnowledgeBase")

@mcp.tool()
async def get_country_info(country_name: str) -> str:
    """Get population, capital, and region for a country."""
    url = f"https://restcountries.com/v3.1/name/{country_name}?fullText=true"
    
    # Configure httpx to use the proxy and enterprise certificate
    async with httpx.AsyncClient(proxy=cfg.get("PROXY", None), verify=cfg.get("CA_PATH", False)) as client:
        try:
            response = await client.get(url, timeout=10.0)
            if response.status_code == 404:
                return f"Country '{country_name}' not found."
            
            data = response.json()[0]
            name = data['name']['common']
            cap = data.get('capital', ['N/A'])[0]
            pop = data.get('population', 0)
            return f"{name} has a population of {pop:,} and its capital is {cap}."
        except Exception as e:
            return f"Error: {str(e)}"

@mcp.tool()
async def get_wikipedia_summary(topic: str) -> str:
    """Get a short summary of a topic from Wikipedia."""
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic.replace(' ', '_')}"
    
    async with httpx.AsyncClient(proxy=cfg.get("PROXY", None), verify=cfg.get("CA_PATH", False)) as client:
        try:
            response = await client.get(url, timeout=10.0)
            if response.status_code == 404:
                return "Topic not found on Wikipedia."
            data = response.json()
            return data.get('extract', 'No summary available.')
        except Exception as e:
            return f"Error fetching Wikipedia: {str(e)}"

if __name__ == "__main__":
    # This turns the server into an HTTP endpoint
    mcp.run(host="0.0.0.0",transport="http", port=8001)