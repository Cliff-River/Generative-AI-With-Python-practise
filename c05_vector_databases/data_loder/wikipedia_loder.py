#%%
from langchain_community.document_loaders import WikipediaLoader
import urllib3

# Disable SSL warnings if needed
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

#%%
articles = [
    {'title': 'Artificial Intelligence'},
    {'title': 'Artificial General Intelligence'},
    {'title': 'Superintelligence'},
]

docs = []
for article in articles:
    try:
        loader = WikipediaLoader(
            query=article['title'], 
            load_all_available_meta=True, 
            doc_content_chars_max=100_000, 
            load_max_docs=1
        )
        doc = loader.load()
        docs.append(doc)
    except Exception as e:
        print(f"Error loading {article['title']}: {e}")
        continue

docs
#%%