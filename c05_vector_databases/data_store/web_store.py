# %% packages
from dotenv import load_dotenv, find_dotenv
import os
from pinecone import Pinecone, ServerlessSpec
from data_prep import create_chunks
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv(find_dotenv())

# %%
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# %%
pc.list_indexes().names()

# %% Create index if it doesn't exist
index_name = "sherlock"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1",
        ),
    )

# %% Split chunks from text file
chunks = create_chunks("HoundOfBaskerville.txt")
texts = [chunk.page_content for chunk in chunks]

# %% Embedding with model all-MiniLM-L6-v2
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
embeddings = embedding_model.embed_documents(texts)

# %% Create vectors
vectors = [
    {
        "id": f"vec{i}",
        "values": embedding,
        "metadata": {
            **chunks[i].metadata,  # Include original metadata
            "text": chunks[i].page_content  # Add the text content to metadata
        }
    } for i, embedding in enumerate(embeddings)
]

#%%
index = pc.Index(index_name)
index.upsert(vectors)

# %%
print(index.describe_index_stats())

#%%