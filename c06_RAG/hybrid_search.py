# %% packages
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())  # read local .env file

# %% docs
docs = [
    "The weather tomorrow will be sunny with a slight chance of rain.",
    "Dogs are known to be loyal and friendly companions to humans.",
    "The climate in tropical regions is warm and humid, often with frequent rain.",
    "Python is a powerful programming language used for machine learning.",
    "The temperature in deserts can vary widely between day and night.",
    "Cats are independent animals, often more solitary than dogs.",
    "Artificial intelligence and machine learning are rapidly evolving fields.",
    "Hiking in the mountains is an exhilarating experience, but it can be unpredictable due to weather changes.",
    "Winter sports like skiing and snowboarding require specific types of weather conditions.",
    "Programming languages like Python and JavaScript are popular choices for web development."
]

# %% remove stop words
docs_without_stopwords = [
    ' '.join([word for word in doc.split() if word.lower() not in ENGLISH_STOP_WORDS])
    for doc in docs
]

# %% tf-idf embeddings
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(docs_without_stopwords)

# %% Set up user query
user_query = "Which weather is good for outdoor activities?"
query_sparse_vec = vectorizer.transform([user_query])
sparse_search = cosine_similarity(query_sparse_vec, tfidf_matrix).flatten()

# %% Function to filter documnents below threshold
def get_filtered_indices(similarities, threshold=0.1):
    indeces_and_values = [
        (i, sim) for i, sim 
        in enumerate(similarities) 
        if sim >= threshold
    ]
    return [i for i, sim in indeces_and_values]

# %%
selected_indices_sparse = get_filtered_indices(sparse_search, threshold=0.1)
selected_indices_sparse

# %%
embeddings = OpenAIEmbeddings(
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=os.environ.get("OPENROUTER_API_KEY"),
)

# %% Create Chroma vector database if not exists
db_path = os.path.join(os.path.dirname(__file__), "hybrid_store")
if os.path.exists(db_path):
    chroma_db = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings,
        collection_name="hybrid_search_docs"
    )
    print("Vector database already exists. Skipping document addition.")
else:
    chroma_db = Chroma.from_documents(
        persist_directory=db_path,
        documents=[Document(page_content=doc, metadata={"index": i}) for i, doc in enumerate(docs)],
        embedding=embeddings,
        collection_name="hybrid_search_docs"
    )
    print("Documents added to vector database.")

# %% Retrieve indices documents from database
retrieved_docs = chroma_db.similarity_search(user_query, k=len(selected_indices_sparse))
selected_indices_dense = [doc.metadata["index"] for doc in retrieved_docs]
selected_indices_dense

# %%
