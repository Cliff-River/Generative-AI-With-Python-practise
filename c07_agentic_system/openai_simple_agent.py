# %% packages
import os
import asyncio
from agents import Agent, Runner
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# Check if API key is loaded and configure OpenRouter
openai_api_key = os.environ.get("OPENAI_API_KEY")
openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")

print(f"OPENAI_API_KEY loaded: {bool(openai_api_key)}")
print(f"OPENROUTER_API_KEY loaded: {bool(openrouter_api_key)}")

# Prefer OpenRouter to avoid quota issues
if openrouter_api_key:
    os.environ["OPENAI_API_KEY"] = openrouter_api_key
    os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
    print("Using OpenRouter configuration")

# %%
agent = Agent(
    name="Simple Agent",
    instructions="You are a simple agent that answers questions about general knowledge. Provide concise and accurate answers.",
    model="openai/gpt-4o-mini"
)

# %%
async def main():
    response = await Runner.run(agent, "What is the capital of France?")
    print(response.final_output)

asyncio.run(main())

# %%