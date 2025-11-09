"""
An application to search for movies based on a query.
- Use streamlit to create a web interface for the application.
"""

# %% packages
import streamlit as st
from data_prep import prepare_data, get_all_genres

# %% Prepare data
vector_store = prepare_data()
st.title("Movie Finder")

# A slider to select the number of imdb ratings
num_ratings = st.slider("Select the number of imdb ratings", min_value=0.0, max_value=10.0, value=7.0, step=0.1)
# A selectbox to select the genre
genres = get_all_genres(vector_store)
selected_genre = st.selectbox("Select the genre", options=genres)

user_query = st.chat_input("What happens in the movie?")
if user_query:
    with st.spinner("Searching for similar movies..."):
        metadata_filter = { "imdb_rating": {"$gte": num_ratings} }
        similar_movies = vector_store.similarity_search_with_score(user_query, k=100, filter=metadata_filter)
        # Further filter by genre if selected
        similar_movies = [movie for movie in similar_movies if selected_genre in movie[0].metadata["genres"]]

    st.header("Most Similar Movies: ")
    st.subheader(f"Query: {user_query}")

    for movie in similar_movies:
        st.write(movie.metadata["title"])