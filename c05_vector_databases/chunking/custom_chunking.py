#%% packages
from langchain.document_loaders import GutenbergLoader
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv, find_dotenv
from custom_chunking_utils import custom_spliter, catch_title
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv(find_dotenv())

#%%
book_details = {
    "title": "The Adventures of Sherlock Holmes",
    "author": "Arthur Conan Doyle",
    "year": 1892,
    "language": "English",
    "genre": "Detective Fiction",
    "url": "https://www.gutenberg.org/cache/epub/1661/pg1661.txt"
}
loader = GutenbergLoader(book_details["url"])
raw_doc = loader.load()
raw_doc[0].metadata = book_details

#%%
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200, is_separator_regex=False, length_function=len)
text_splitter.split_text = custom_spliter
books = text_splitter.split_documents(raw_doc)
len(books)

#%%
books[0].page_content

#%%
books = books[1:]

for book in books:
    title = catch_title(book.page_content)
    book.metadata["title"] = title

#%%
spliter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, is_separator_regex=False, length_function=len, 
    separators=["\n\n", "\n", ".", "!", "?", " "])
chunks = spliter.split_documents(books)
len(chunks)

#%%