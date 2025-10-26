#%%
from langchain_experimental.text_splitter import SemanticChunker
from langchain.document_loaders import WikipediaLoader
from dotenv import load_dotenv, find_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv(find_dotenv())

#%%
ai_article_title = "Artificial_intelligence"
loader = WikipediaLoader(ai_article_title, load_max_docs=1, doc_content_chars_max=1000, load_all_available_meta=True)
docs = loader.load()
docs[0].page_content

#%%
spliter = SemanticChunker(embeddings=OpenAIEmbeddings(), 
                          breakpoint_threshold_type="cosine", breakpoint_threshold_amount=0.5)
chunks = spliter.split_documents(docs)
chunks

#%%