from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm

# Define the local model configuration
local_model = LiteLlm(
    model="openai/local-model", # Prefix with openai/
    api_base="http://127.0.0.1:8080/v1",
    api_key="not-needed" # Usually required by the SDK but ignored by local servers
)

def get_current_time(city: str):
    """Returns the current time for the specified city."""
    # Placeholder implementation - replace with actual time retrieval logic
    return {"status": "success", "city": city, "current_time": "12:00 PM"}

root_agent = LlmAgent(
    name="root_agent",
    model=local_model,
    instruction="You are a helpful assistant running locally that tells the current time.  Use the 'get_current_time' tool for this purpose.",
    tools=[get_current_time]
)

## Command to run: uv run adk run agent