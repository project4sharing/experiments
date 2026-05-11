import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("KnowledgeBase")

@mcp.tool()
async def get_country_info(country_name: str) -> str:
    """Get population, capital, and region for a country."""
    url = f"https://restcountries.com/v3.1/name/{country_name}?fullText=true"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=10.0)
            if response.status_code == 404:
                return f"Country '{country_name}' not found."
            
            data = response.json()[0]
            name = data['name']['common']
            cap = data.get('capital', ['N/A'])[0]
            pop = data.get('population', 0)
            region = data.get('region', 'N/A')
            
            return f"{name} (Region: {region}) has a population of {pop:,} and its capital is {cap}."
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
async def get_wikipedia_summary(topic: str) -> str:
    """Get a short summary of a topic from Wikipedia."""
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic.replace(' ', '_')}"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=10.0)
            if response.status_code == 404:
                return "Topic not found on Wikipedia."
            
            data = response.json()
            return data.get('extract', 'No summary available.')
    except Exception as e:
        return f"Error fetching Wikipedia: {str(e)}"

if __name__ == "__main__":
    mcp.run()