#%%
from langchain.document_loaders import GutenbergLoader

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
docs[0]

#%%
docs[0].metadata = book_details

#%%