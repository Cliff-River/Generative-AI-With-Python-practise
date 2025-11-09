# %% packages
from datasets import load_dataset
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

# %% Load dataset
ds = load_dataset("MongoDB/embedded_movies", split="train")
ds[0].keys(), len(ds)

# %%  Create list of Document instances with fullplot as page_content
docs = []
for item in ds:
    if item["fullplot"] is None:
        continue
    docs.append(Document(
        page_content=item["fullplot"],
        metadata={
            "title": item["title"] if item["title"] is not None else "",
            "poster": item["poster"] if item["poster"] is not None else "",
            "genre": "; ".join(item["genres"]) if item["genres"] is not None else "",
            "imdb_rating": item["imdb"]["rating"] if item["imdb"]["rating"] is not None else ""
        }
    ))

# %% Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", ",", " ", ""],
)
chunks = text_splitter.split_documents(docs)
len(chunks)

# %% Store embeddings into Chroma DB
db_path = os.path.join(os.path.abspath(__file__), "..", "db")
# Initialize embeddings model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create Chroma vector store first
vector_store = Chroma(
    embedding_function=embedding_model,
    persist_directory=db_path,
    collection_name="movies",
)

# Add documents to the existing vector store
vector_store.add_documents(documents=chunks)

# %%  Get all unique genres
all_genres = set()
for meta in docs:
    if meta.metadata["genre"]:
        genres = meta.metadata["genre"].split(";")
        all_genres.update(genres)
all_genres

# %%  Get all genrees from database
# Get all unique genres from the database
db_genres = set()
for meta in vector_store.get()["metadatas"]:
    if meta["genre"]:
        genres = meta["genre"].split(";")
        db_genres.update(genres)
db_genres

# %%