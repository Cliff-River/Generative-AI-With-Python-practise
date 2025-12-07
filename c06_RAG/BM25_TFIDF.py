# %% packages
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

# %% Function to preprocess text
def preprocess_text(text: str) -> list[str]:
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.split()

# %% tokenized corpus
corpus = [
    "Artificial intelligence is a field of artificial intelligence. The field of artificial intelligence involves machine learning. Machine learning is an artificial intelligence field. Artificial intelligence is rapidly evolving.",
    "Artificial intelligence robots are taking over the world. Robots are machines that can do anything a human can do. Robots are taking over the world. Robots are taking over the world.",
    "The weather in tropical regions is typically warm. Warm weather is common in these regions, and warm weather affects both daily life and natural ecosystems. The warm and humid climate is a defining feature of these regions.",
    "The climate in various parts of the world differs. Weather patterns change due to geographic features. Some regions experience rain, while others are dry."
]
tokenized_corpus = [preprocess_text(doc) for doc in corpus]
tokenized_corpus_tfidf = [' '.join(words) for words in tokenized_corpus]

# %%
bm25 = BM25Okapi(tokenized_corpus)
tfidf_vectorizer = TfidfVectorizer()

# %% Define function to calculate similarity scores
def calculate_similarity_scores(user_query: str):
    """
    Calculate BM25 scores for a given user query using a pre-initialized BM25 object.
    
    Args:
        user_query (str): The user's query string
        
    Returns:
        dict: Dictionary containing tokenized queries and BM25 scores
    """
    tokenized_query_bm25 = user_query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query_bm25)

    tokenized_query_tfidf = ' '.join(tokenized_query_bm25)
    tfidf_matrix = tfidf_vectorizer.fit_transform(tokenized_corpus_tfidf)
    tfidf_query_vector = tfidf_vectorizer.transform([tokenized_query_tfidf])
    cosine_similarities = cosine_similarity(tfidf_query_vector, tfidf_matrix)
    tfidf_scores = cosine_similarities.tolist()
    
    # Print results
    print("BM25 Scores:", bm25_scores)
    print("Tokenized Query (BM25):", tokenized_query_bm25)
    print("Tokenized Query (TF-IDF string):", tokenized_query_tfidf)
    print("TF-IDF Scores:", tfidf_scores)
    
    # Return results as a dictionary for potential further use
    return {
        "tokenized_query_bm25": tokenized_query_bm25,
        "tokenized_query_tfidf": tokenized_query_tfidf,
        "bm25_scores": bm25_scores,
        "tfidf_scores": tfidf_scores
    }

# %% Example usage 1
user_query = "What is the weather like in tropical regions?"
results = calculate_similarity_scores(user_query)

# %% Example usage 2
user_query = "Humid climate"
results = calculate_similarity_scores(user_query)

# %%