"""
start_registry_server.py
--------------------
Starts a python-a2a AgentRegistry that other agents register with.
Run this first, before starting any agent servers.
"""
import python_a2a
from pathlib import Path
import tomllib


from python_a2a.discovery import AgentRegistry, run_registry

def main():
    with open(f"{str(Path(__file__).parent)}/conf/agent_registry.toml", "rb") as f:
        conf = tomllib.load(f)

    registry = AgentRegistry(
        name="MCP-A2A Registry",
        description="Central discovery registry for agents",
    )
    print(f"Registry running at http://{conf['REGISTRY']['HOST']}:{conf['REGISTRY']['PORT']}")
    print("Agents register via POST /register")
    print("Clients discover via GET  /agents\n")
    run_registry(registry, host=conf['REGISTRY']['HOST'], port=conf['REGISTRY']['PORT'])


if __name__ == "__main__":
    main()
