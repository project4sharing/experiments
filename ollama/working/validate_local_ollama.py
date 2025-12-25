from openai import OpenAI

# Point to your server's IP
client = OpenAI(
    base_url='http://127.0.0.1:11434/v1',
    api_key='ollama', # Required but ignored
)

print("Sending request to server...")

response = client.chat.completions.create(
  model="gemma3:1b",
  messages=[
    {"role": "system", "content": "You are a helpful AI assistant running on a powerful server."},
    {"role": "user", "content": "Why is having 512GB of RAM useful for AI?"}
  ]
)

print(response.choices[0].message.content)