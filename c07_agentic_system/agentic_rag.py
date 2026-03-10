# %% packages
import os
from langchain_community.tools.tavily_search.tool import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import WikipediaLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# %% load documents
loader = WikipediaLoader("Principle of relativity", load_max_docs=18)
docs  = loader.load()

# %% split documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(docs)

# %%
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, base_url="https://openrouter.ai/api/v1", api_key=os.environ.get("OPENROUTER_API_KEY"))
embeddings = OpenAIEmbeddings(base_url="https://openrouter.ai/api/v1", api_key=os.environ.get("OPENROUTER_API_KEY"))

# %%
vectorstore = FAISS.from_documents(chunks, embeddings)

# %%