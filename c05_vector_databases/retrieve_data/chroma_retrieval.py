# %% packages
import os
from pprint import pprint
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

# %% Set up chrona database connection
db_path = os.path.join(os.path.abspath(__file__), "..", "..", "db")
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
chroma_db = Chroma(
    persist_directory=db_path,
    embedding_function=embeddings_model,
)
retriever = chroma_db.as_retriever()

# %% Find infomations
# query = "Who is the sidekick of Sherlock Holmes in the book?"

# # thematic search
# query = "Find passages that describe the moor or its atmosphere."

# # Emotion
# query = "Which chapters or passages convey a sense of fear or suspense?"

# # Dialogue Analysis
# query = "Identify all conversations between Sherlock Holmes and Dr. Watson."

# Character
query = "How does the hound look like?"
most_simular_docs = retriever.invoke(query)
pprint(most_simular_docs[0].page_content)

# %%
