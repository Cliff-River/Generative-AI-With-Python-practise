#%%
from os import path
from langchain.document_loaders import TextLoader, DirectoryLoader

directory_path = path.join(path.dirname(__file__), "..", "data")
loader = DirectoryLoader(directory_path, glob="*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})

#%%
loader.load()

#%%