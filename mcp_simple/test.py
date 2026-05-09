PORTS = {
    "clean_http":    
    {'port': 8000},
    "clean_sse":     {'port': 8001},
    "poisoned_tool": {'port': 8002},
    "goal_drift":    {'port': 8003},
}

mcpServers = {

    "clean_http":    
    {'port': 8000},
    "clean_sse":     {'port': 8001},
    "poisoned_tool": {
        'url': 'http://localhost:8002/mcp',
        'auth': None
    },
    "goal_drift": {
        'url': 'http://localhost:8003/mcp',
        'auth': None
    },
}

def multi_server_config(*server_keys: str) -> dict:
    """Return an mcpServers dict for multiple servers."""
    return {"mcpServers": {k: mcpServers[k] for k in mcpServers if k in server_keys}}

MCP_URLS = {k: f"http://localhost:{v}/mcp" for k, v in PORTS.items()}

MCP_URLS2 = {k: v for k, v in mcpServers.items()}
# print(MCP_URLS2)
print(multi_server_config( "poisoned_tool"))

# return {"mcpServers": {k: {"url": MCP_URLS[k]} for k in server_keys}}