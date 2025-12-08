# %% packages
from langchain_openai import OpenAIEmbeddings
import scipy as sp
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
selected_indices = get_filtered_indices(sparse_search, threshold=0.1)
selected_indices

# %%
embeddings = OpenAIEmbeddings(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model="openrouter/auto"
)
embedded_docs = [embeddings.embed_documents(doc) for doc in docs]
query_dense_vec = embeddings.embed_query(user_query)

# %% dense search
dense_similarities = cosine_similarity(
    [query_dense_vec], 
    embedded_docs
)
dense_similarities

# %%
