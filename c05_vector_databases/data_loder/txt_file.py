#%%
from os import path
from langchain.document_loaders import TextLoader

#%%
file_path = path.join(path.dirname(__file__), "..", "data", "HoundOfBaskerville.txt")
loader = TextLoader(file_path, encoding="utf-8")
doc = loader.load()
doc[0].metadata
#%%
doc[0].page_content[:500]  # Preview the first 500 characters
#%%