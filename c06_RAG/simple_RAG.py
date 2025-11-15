"""
Fetch documents about 'Human History' from Wikipedia and set up a simple Retrieval-Augmented Generation (RAG) pipeline using LangChain and Groq AI.
"""

# %% packages
from langchain.document_loaders import WikipediaLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
import os
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# %% Load dataset, if rag_store directory does not exist, create it, otherwise load it as Chroma vector store
# Set up embedding model using OpenAI Embeddings via OpenRouter
embeddings = OpenAIEmbeddings(
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=os.environ.get("OPENROUTER_API_KEY"),
)

db_path = os.path.join(os.path.abspath(__file__), "..", "rag_store")
# Check if rag_store directory exists
if os.path.exists(db_path):
    print("Loading existing vector store...")
    vectorstore = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings
    )
    print("Vector store loaded successfully")
else:
    print("Creating vector store...")
    
    # Load Wikipedia documents about 'Human History'
    docs = WikipediaLoader(
        query="Human History",
        load_max_docs=50,
        doc_content_chars_max=1_000_000
    ).load()
    print(f"Loaded {len(docs)} documents")
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")
    
    # Create and persist vector store
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_path
    )
    vectorstore.persist()
    print("Vector store created and persisted")

# %% Use the vector store as retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)
relevant_docs = retriever.get_relevant_documents("What happened in the first world war?")
[doc.page_content[:100] for doc in relevant_docs]

# %%
