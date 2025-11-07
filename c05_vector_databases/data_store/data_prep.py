"""
Overview:
This module contains functions for preparing and processing data to be stored in vector databases.
"""

#%% packages
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import os

#%% functions to create chunks
def create_chunks(file_name : str) -> list[Document]:
    """Create chunks from a text file.

    Args:
        file_name (str): The name of the text file to be processed. 
    Returns:
        list[Document]: A list of Document objects representing the text chunks.
    """
    # get file path from data folder
    file_path = os.path.join(os.path.abspath(__file__), "..", "..", "data", file_name)

    # Load the text file
    loader = TextLoader(file_path, encoding="utf8")
    documents = loader.load()

    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ".", "!", ",", ""],
    )
    docs = text_splitter.split_documents(documents)

    return docs
