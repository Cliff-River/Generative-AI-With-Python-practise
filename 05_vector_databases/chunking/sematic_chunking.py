#%%
from langchain_experimental.text_splitter import SemanticChunker
from langchain.document_loaders import WikipediaLoader
from dotenv import load_dotenv, find_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv(find_dotenv())

#%%
ai_article_title = "Artificial_intelligence"
loader = WikipediaLoader(ai_article_title, load_max_docs=1, doc_content_chars_max=10000, load_all_available_meta=True)
docs = loader.load()
docs[0].page_content

#%%
spliter = SemanticChunker(embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"), )
chunks = spliter.split_documents(docs)
chunks

#%%
len(chunks)

#%%
chunks[0].page_content

#%%