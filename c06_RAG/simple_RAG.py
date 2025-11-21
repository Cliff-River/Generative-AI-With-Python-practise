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
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseRetriever
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

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
question = "What happened in the first world war?"
relevant_docs = retriever.get_relevant_documents(question)
[doc.page_content[:100] for doc in relevant_docs]

# %% prompt engineering
context = "\n".join([doc.page_content for doc in relevant_docs])
prompt_template = ChatPromptTemplate.from_messages([
    ("system", """
     You are an AI assistant that answers questions about the history of human civilization. You are given a question and a list of documents and need to answer the question. Ansser the question only based on the provided documents. These document can help you answer the question: 
     <context>
        {context}
     </context>
     If you don't know the answer, just say you don't know or say 'Sorry, I don't know the answer to this question.'. Do not try to make up an answer.
     """),
    ("human", "{question}")
])

# %% Create a chain and invoke it
chat_model = ChatOpenAI(
    model="google/gemma-3-27b-it:free",
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=os.environ.get("OPENROUTER_API_KEY"),
)
chain = prompt_template | chat_model | StrOutputParser()
response = chain.invoke({
    "context": context,
    "question": question
})
print(response)

# %%# Test with another question
response = chain.invoke({
    "context": context,
    "question": "What is cosine similarity?"
})
print(response)

# %% boudle RAG process as a function
def simple_RAG(question: str, retriever: BaseRetriever, chat_model: BaseChatModel) -> str:
    """
    Execute a simple RAG pipeline: retrieve relevant documents and generate an answer.
    
    Args:
        question: The user's question
        retriever: LangChain retriever for document search
        chat_model: LLM chat model for answer generation
        
    Returns:
        Generated answer based on retrieved context
    """
    # Retrieve relevant documents
    relevant_docs = retriever.get_relevant_documents(question)
    
    # Build context from retrieved documents
    context = "\n".join([doc.page_content for doc in relevant_docs])
    
    # Create prompt template
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """
         You are an AI assistant that answers questions about the history of human civilization. You are given a question and a list of documents and need to answer the question. Answer the question only based on the provided documents. These documents can help you answer the question: 
         <context>
            {context}
         </context>
         If you don't know the answer, just say you don't know or say 'Sorry, I don't know the answer to this question.'. Do not try to make up an answer.
         """),
        ("human", "{question}")
    ])
    
    # Create chain and invoke
    chain = prompt_template | chat_model | StrOutputParser()
    response = chain.invoke({
        "context": context,
        "question": question
    })
    
    return response

# %% Example usage
answer = simple_RAG(question, retriever, chat_model)
print(answer)

# %% Test with another question
answer2 = simple_RAG("What is cosine similarity?", retriever, chat_model)
print(answer2)

# %%