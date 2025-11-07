#%% packages
from dotenv import load_dotenv, find_dotenv
import os
from pinecone import Pinecone, ServerlessSpec

load_dotenv(find_dotenv())

#%%
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

#%%
pc.list_indexes().names()

#%% Create index if it doesn't exist
index_name = "sherlock"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        serverless_spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1",
        ),
    )

#%%
