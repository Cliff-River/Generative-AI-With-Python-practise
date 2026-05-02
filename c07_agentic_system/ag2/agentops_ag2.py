# %% Packages
import agentops
from dotenv import load_dotenv, find_dotenv
import os
from agentops import agent, task, trace
import logging
from openai import OpenAI

load_dotenv(find_dotenv())
logging.basicConfig(level=logging.DEBUG)

# Initialize AgentOps with the v4 tracing API.
try:
    agentops.init(
        api_key=os.environ.get("AGENTOPS_API_KEY"),
        default_tags=["ag2-demo"],
        instrument_llm_calls=False,
        auto_start_session=True,
        fail_safe=True,
        log_level="ERROR",
    )
    logging.info("AgentOps initialized successfully")
except Exception as e:
    logging.error(f"AgentOps initialization error: {e}", exc_info=True)

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
@agent(name="jack")
class FlatEarthAgent:
    @task(name="completion")
    def completion(self, prompt: str) -> str:
        response = openai_client.chat.completions.create(
            model="openai/gpt-4o-mini",
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
@agent(name="alice")
class ScientistAgent:
    @task(name="completion")
    def completion(self, prompt: str) -> str:
        response = openai_client.chat.completions.create(
            model="openai/gpt-4o-mini",
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
@task(name="argue_flat_earth")
def argue_flat_earth(jack: FlatEarthAgent) -> str:
    return jack.completion("Hello, how can you not see that the earth is flat?")


@task(name="respond_flat_earth")
def respond_flat_earth(alice: ScientistAgent, flat_earth_arguement: str) -> str:
    return alice.completion(
        f"Respond to the argument that the earth is flat with convincing evidence: \n{flat_earth_arguement}"
    )


@trace(name="ag2-demo")
def run_demo():
    jack = FlatEarthAgent()
    alice = ScientistAgent()
    flat_earth_arguement = argue_flat_earth(jack)
    return respond_flat_earth(alice, flat_earth_arguement)


try:
    run_demo()
    agentops.end_all_sessions()
    logging.info("Demo completed successfully and all sessions ended")
except Exception as e:
    logging.error(f"Demo execution error: {e}", exc_info=True)
    agentops.end_all_sessions()

# %%