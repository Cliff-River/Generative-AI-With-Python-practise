# %% package
import anthropic
from langchain_community.document_loaders import TextLoader
from rich.console import Console
from rich.markdown import Markdown
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# %% client
client = anthropic.Anthropic()

# %% class to prompt caching
class PromptCaching:
    def __init__(self, initial_context: str):
        self.messages = []
        self.context = None
        self.initial_context = initial_context

    def run_model(self, prompt: str) -> str:
        self.messages.append({"role": "user", "content": prompt})
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1024,
            messages=self.messages
        )
        self.messages.append({"role": "assistant", "content": response.content[0].text})
        return response.content[0].text