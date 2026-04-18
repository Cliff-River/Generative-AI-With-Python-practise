# %% packages
import os
from autogen import ConversableAgent
from dotenv import load_dotenv, find_dotenv
from datetime import datetime

load_dotenv(find_dotenv())

# %% LLM configuration
llm_config = {
    "config_list": [
        {
            "model": "gpt-4o-mini",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": os.environ.get("OPENROUTER_API_KEY"),
            "temperature": 0.9,
        }
    ]
}

# %% A function to get current date
def get_current_date() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def is_task_completed_message(msg: dict) -> bool:
    content = msg.get("content")
    print(f"Received message content: {content}")
    return content is not None and "task completed" in str(content).lower()


# %%
my_assisistant = ConversableAgent(
    name="my_assisistant",
    description="An assistant that can help you with various tasks.",
    system_message="""
    You are a helpful assistant that can perform various tasks. You have access to the following tool:
    1. get_current_date: This tool returns the current date in YYYY-MM-DD format.

    Return 'Task Completed' after then the task is done.
    """,
    llm_config=llm_config,
    human_input_mode="NEVER",
)

# %% register the tool
my_assisistant.register_for_llm(
    name="get_current_date", description="Get the current date in YYYY-MM-DD format."
)(get_current_date)

# %%
user_proxy = ConversableAgent(
    name="user_proxy",
    description="A proxy for the user to interact with the assistant.",
    llm_config=llm_config,
    human_input_mode="NEVER",
    is_termination_msg=is_task_completed_message,
)
user_proxy.register_for_execution(name="get_current_date")(get_current_date)

# %%
result = user_proxy.initiate_chat(my_assisistant, message="What is the current date?")

# %% print the result
result

# %%