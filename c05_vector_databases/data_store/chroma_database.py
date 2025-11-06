#%% packages
import os
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

#%% Get path of HoundOfBaskerville.txt
file_path = os.path.join(os.path.abspath(__file__), "..", "..", "data", "HoundOfBaskerville.txt")

#%% Load test file
loader = TextLoader(file_path, encoding="utf8")
documents = loader.load()

#%% Splite the document
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ".", ","])
chunks = text_splitter.split_documents(documents)
len(chunks)

#%% Create instance of embedding model
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#%% Create Chroma vector database
db_path = os.path.join(os.path.abspath(__file__), "..", "..", "db")
chroma_db = Chroma.from_documents(documents=chunks, embedding=embeddings_model, collection_name="hound_of_baskerville_db")

#%%
chroma_db.add_documents(documents=chunks)

#%% Get data from database
chroma_db.get()["ids"]

#%%