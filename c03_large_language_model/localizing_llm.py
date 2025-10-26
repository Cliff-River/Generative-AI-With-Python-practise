#%%
import ollama
from pprint import pprint
from langchain_community.llms.ollama import Ollama

#%%
response = ollama.generate("gemma2:2b", "What's an LLM")

#%%
pprint(response["response"])
print("=======================================================")

#%%
llm = Ollama(model="gemma2:2b")
response = llm.invoke("What is a cat")
print(response)

#%%