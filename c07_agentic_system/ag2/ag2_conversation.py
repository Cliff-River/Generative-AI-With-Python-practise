# %% Packages
from autogen import ConversableAgent
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv())

# %% LLM configuration
llm_config = {
    "config_list": [
        {
            "model": "minimax/minimax-m2.5",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": os.environ.get("OPENROUTER_API_KEY"),
            "temperature": 0.9,
        }
    ]
}

# %%
jack_flat_earther = ConversableAgent(
    "Jack",
    llm_config=llm_config,
    system_message="""
    You believe that the earth is flat. 
    You try to convince others of this. 
    With every answer, you are more frustrated and angry that they don't see it.
    """,
    human_input_mode="NEVER"
)

# %%
alice_scientist = ConversableAgent(
    "Alice",
    llm_config=llm_config,
    system_message="""
    You are a scientist who believes that the earth is round. 
    Answer very polite, short and concise.
    """,
    human_input_mode="NEVER"
)

# %% Initiate the conversation
result = jack_flat_earther.initiate_chat(
    alice_scientist, 
    message="Hello, how can you not see that the earth is flat?",
    max_turns=5)

# %% Print the conversation history
result.chat_history

# %%