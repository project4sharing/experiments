from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm

# Define the local model configuration
local_model = LiteLlm(
    model="openai/local-model", # Prefix with openai/
    api_base="http://127.0.0.1:8080/v1",
    api_key="not-needed" # Usually required by the SDK but ignored by local servers
)

# Pass the configured model wrapper to your LlmAgent
root_agent = LlmAgent(
    # model=LiteLlm(model="gemini/gemini-2.5-pro-exp-03-25"),
    # model=LiteLlm(model="vertex_ai/gemini-2.5-pro-exp-03-25"),
    # model=LiteLlm(model="vertex_ai/claude-3-5-haiku"),
    model=local_model,
    name="my_governed_agent",
    instruction="You are a helpful assistant powered by Gemini and governed by Apigee.",
    # ... other agent parameters
)

# run "uv run adk run agent_v2_01" in the adk_v2 folder to execute this agent