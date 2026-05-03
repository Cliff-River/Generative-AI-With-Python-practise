# %% packages
from langchain_community.document_loaders import WikipediaLoader
from dotenv import find_dotenv, load_dotenv
from pydantic_ai import Agent
import nest_asyncio
from pydantic import BaseModel, Field
import logfire

nest_asyncio.apply()
logfire.configure()
load_dotenv(find_dotenv())

# %%  Enable AI instrumentation so Logfire monitors OpenAI / agent calls
# (calls available on the default instance: instrument_openai, instrument_openai_agents)
try:
    logfire.instrument_openai()
    logfire.instrument_openai_agents()
except Exception:
    # If instrumentation fails, continue but emit an error-level log
    try:
        logfire.error("Logfire AI instrumentation failed to initialize")
    except Exception:
        pass

# %% load documents
loader = WikipediaLoader("Alan Turing", load_all_available_meta=True, doc_content_chars_max=100_000, load_max_docs=1)
docs = loader.load()
page_content = docs[0].page_content
page_content[:500]  # show the first 500 characters of the page content

# %% pydantic model
class PersonInfo(BaseModel):
    name: str = Field(..., description="The full name of the person")
    birth_date: str = Field(..., description="The birth date of the person")
    death_date: str = Field(..., description="The death date of the person")
    known_for: str = Field(..., description="What the person is known for")
    publications: list[str] = Field(..., description="A list of notable publications of the person")
    achievements: list[str] = Field(..., description="A list of achievements of the person")

# %%
agent = Agent(model="openrouter:minimax/minimax-m2.7", output_type=PersonInfo)

# %%
result = agent.run_sync(page_content)
print(result.output.model_dump())

# %%