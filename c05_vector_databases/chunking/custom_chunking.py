#%% packages
from langchain.document_loaders import GutenbergLoader
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv, find_dotenv
from custom_chunking_utils import custom_spliter

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
docs = loader.load()

#%%

