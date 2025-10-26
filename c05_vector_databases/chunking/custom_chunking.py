#%% packages
import re
from langchain.document_loaders import GutenbergLoader
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv, find_dotenv

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
def custom_spliter(text):
    """
    Split text based on Roman numeral chapter headers.
    
    Pattern matches newlines followed by Roman numerals (I, II, III, IV, V, etc.)
    followed by a period and a space, and then a capital letter.
    
    Args:
        text (str): The text to split
        
    Returns:
        list: List of text chunks split by Roman numeral chapter headers
    """
    pattern = r'\n(?=[IVX]+\.\s[A-Z])' 
    return re.split(pattern, text)
