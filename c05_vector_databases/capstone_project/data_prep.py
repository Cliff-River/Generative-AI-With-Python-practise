# %% packages
from datasets import load_dataset
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

# %% function to prepare data
# Initialize embeddings model
def prepare_data() -> Chroma:
    """
    Prepare data for the movie database.
    This function initializes the embeddings model, creates a Chroma instance,
    and either loads an existing database or creates a new one if it doesn't exist.
    """
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store : Chroma = None
    db_path = os.path.join(os.path.abspath(__file__), "..", "db")
    def create_chroma_instance():
        """
        Create a Chroma instance with the specified embedding model and database path.
        """
        vector_store = Chroma(
            embedding_function=embedding_model,
            persist_directory=db_path,
            collection_name="movies",
        )
        return vector_store
    # Check if the database already exists
    if os.path.exists(db_path):
        vector_store = create_chroma_instance()
    else:
        vector_store = create_chroma_instance()
        create_database(vector_store)
    return vector_store

# %% function to create database
def create_database(vector_store : Chroma):
    # Load dataset
    ds = load_dataset("MongoDB/embedded_movies", split="train")
    #  Create list of Document instances with fullplot as page_content
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
    
    #  Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", ",", " ", ""],
    )
    chunks = text_splitter.split_documents(docs)

    # Store embeddings into Chroma DB
    # Add documents to the existing vector store
    vector_store.add_documents(documents=chunks)

# %%