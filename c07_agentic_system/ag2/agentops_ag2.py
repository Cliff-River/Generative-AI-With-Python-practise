# %% Packages
import agentops
from dotenv import load_dotenv, find_dotenv
import os
from agentops import track_agent, record_action, init as init_agentops
import logging
from openai import OpenAI

load_dotenv(find_dotenv())
init_agentops()
logging.basicConfig(level=logging.DEBUG)

# %% config for OpenAI API
openai_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ.get("OPENROUTER_API_KEY"))

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

# %% Define the agent jack with agentops tracking
@track_agent(name="jack")
class FlatEarthAgent:
    def completion(self, prompt: str) -> str:
        response = openai_client.chat.completions.create(
            model="minimax/minimax-m2.7",
            messages=[
                {
                    "role": "system",
                    "content": """
                    You believe that the earth is flat. 
                    You try to convince others of this. 
                    With every answer, you are more frustrated and angry that they don't see it.
                    """
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
        )
        return response.choices[0].message.content

# %% Define the agent alice with agentops tracking
@track_agent(name="alice")
class ScientistAgent:
    def completion(self, prompt: str) -> str:
        response = openai_client.chat.completions.create(
            model="minimax/minimax-m2.7",
            messages=[
                {
                    "role": "system",
                    "content": """
                    You are a scientist who believes that the earth is round. 
                    Answer very polite, short and concise.
                    """
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
        )
        return response.choices[0].message.content

# %% Initiate the two agents
jack = FlatEarthAgent()
alice = ScientistAgent()

flat_earth_arguement = jack.completion("Hello, how can you not see that the earth is flat?")

# %%
@record_action(event_name="argue_flat_earth")
def argue_flat_earth():
    return jack.completion("Hello, how can you not see that the earth is flat?")

@record_action(event_name="respond_flat_earth")
def respond_flat_earth():
    return alice.completion(f"Respond to the argument that the earth is flat with convincing evidence: \n{flat_earth_arguement}")

argue_flat_earth()
respond_flat_earth()

# %% End monitoring
agentops.end_session("Success")

# %%