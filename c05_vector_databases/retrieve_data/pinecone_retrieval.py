# %% packages
from dotenv import load_dotenv, find_dotenv
import os
from pinecone import Pinecone
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv(find_dotenv())

# %% Set up Pinecone connection
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "sherlock"
index = pc.Index(index_name)

# %% Set up embeddings model for query encoding
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# %% Find information using the same queries as chroma_retrieval.py
# # Character
query = "How does the hound look like?"
query_embedding = embeddings_model.embed_query(query)
results = index.query(
    vector=query_embedding,
    top_k=2,
    include_metadata=True
)

# %% Display text content of matches
for match in results.matches:
    # Access the text from metadata
    text = match.metadata.get('text', 'No text found')
    print(f"Text: {text}")
    print("=================")

# %% Alternative queries (commented out)
# query = "Who is the sidekick of Sherlock Holmes in the book?"
# query = "Find passages that describe the moor or its atmosphere."
# query = "Which chapters or passages convey a sense of fear or suspense?"
# query = "Identify all conversations between Sherlock Holmes and Dr. Watson."