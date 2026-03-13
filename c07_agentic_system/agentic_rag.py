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
retriever = vectorstore.as_retriever(return_similarities=True)

# %%
query = "What is relativity?"
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that answers questions about the principle of relativity.You can get contextial information from the retrieved documents. If you don't know the answer, just say 'insufficient information'. Always use the retrieved documents to answer the question."),
    ("user", "<context>{context}</context>\n\n<question>{question}</question>")
])
retrieved_docs = retriever.invoke(query)
retrieved_docs_str = "; ".join([doc.page_content for doc in retrieved_docs])
chain = prompt_template | llm
rag_response = chain.invoke({"context": retrieved_docs_str, "question": query})

# %%
rag_response.content

# %% a simple test
search_tool = TavilySearchResults(max_results=5, include_answer=True)
search_response = search_tool.invoke({"query": query})
[doc["content"] for doc in search_response]

# %%
if rag_response.content == "insufficient information":
    print("RAG model did not have enough information. Using search tool to find the answer.")
    search_response = search_tool.invoke({"query": query})
    context = "; ".join([doc["content"] for doc in search_response])
    final_response = chain.invoke({"context": context, "question": query})
else:
    final_response = rag_response.content

final_response
# %%