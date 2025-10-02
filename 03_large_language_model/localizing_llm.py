#%%
import ollama
from pprint import pprint

response = ollama.generate("gemma2:2b", "What's an LLM")

#%%
pprint(response["response"])