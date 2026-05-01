# %% packages
import asyncio
from agents import Agent, Runner, set_default_openai_client 
from dotenv import load_dotenv, find_dotenv
from openai import AsyncOpenAI
import os

load_dotenv(find_dotenv())

# 1. 创建 OpenRouter 客户端
client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# 2. 设置为默认客户端
set_default_openai_client(client)

# %%
agent = Agent(
    name="Simple Agent",
    instructions="You are a simple agent that answers questions about general knowledge. Provide concise and accurate answers.",
    model="openrouter/openai/gpt-4o-mini"
)

# %%
async def main():
    response = await Runner.run(agent, "What is the capital of France?")
    print(response.final_output)

asyncio.run(main())

# %%