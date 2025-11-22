# %% packages
import string
import numpy as np
from tqdm import tqdm
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

# %%