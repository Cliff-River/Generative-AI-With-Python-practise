"""
An application to search for movies based on a query.
- Use streamlit to create a web interface for the application.
"""

# %% packages
import streamlit as st
from data_prep import prepare_data, get_all_genres
import pandas as pd

# %% Page configuration
st.set_page_config(
    page_title="Movie Finder",
    page_icon="🎬",
    layout="wide"
)

# %% Prepare data
@st.cache_resource
def init_vector_store():
    return prepare_data()

vector_store = init_vector_store()
st.title("🎬 Movie Finder")
st.markdown("### Find movies based on your preferences and search query")

# %% Sidebar for filters
st.sidebar.header("Search Filters")

# A slider to select the number of imdb ratings
num_ratings = st.sidebar.slider("Minimum IMDB Rating", min_value=0.0, max_value=10.0, value=7.0, step=0.1)
st.sidebar.caption(f"Showing movies with rating ≥ {num_ratings}")

# A selectbox to select the genre
genres = ["all"] + sorted(list(get_all_genres(vector_store)))
selected_genre = st.sidebar.selectbox("Select Genre", options=genres)
if selected_genre != "all":
    st.sidebar.caption(f"Filtering by: {selected_genre}")

# Number of results to show
num_results = st.sidebar.slider("Number of Results", min_value=5, max_value=50, value=20)
st.sidebar.caption(f"Showing top {num_results} results")

# %% Main content
col1, col2 = st.columns([3, 1])

with col1:
    user_query = st.text_input(
        "What happens in the movie?", 
        placeholder="e.g., A group of friends go on an adventure, A detective investigates a mystery, etc.",
        help="Describe the plot or theme you're looking for"
    )

with col2:
    st.write("")  # Spacer
    search_button = st.button("🔍 Search Movies", type="primary")

# Process search
if user_query and search_button:
    with st.spinner("Searching for similar movies..."):
        metadata_filter = { "imdb_rating": {"$gte": num_ratings} }
        similar_movies = vector_store.similarity_search_with_score(user_query, k=100, filter=metadata_filter)
        
        # Further filter by genre if selected
        if selected_genre != "all":
            similar_movies = [movie for movie in similar_movies if selected_genre in movie[0].metadata["genres"]]

    st.header("🎯 Most Similar Movies")
    
    # Prepare data for table display
    if similar_movies:
        movie_data = []
        for doc, score in similar_movies[:num_results]:
            # Clean up data
            title = doc.metadata.get("title", "N/A")
            genres_str = doc.metadata.get("genres", "N/A")
            rating = doc.metadata.get("imdb_rating", "N/A")
            similarity_score = round(score, 3)
            
            # Format rating
            if rating != "N/A":
                try:
                    rating = f"{float(rating):.1f}"
                except Exception:
                    rating = "N/A"
            
            movie_data.append({
                "Title": title,
                "Genres": genres_str.replace("; ", ", ") if genres_str != "N/A" else "N/A",
                "IMDB Rating": rating,
                "Similarity Score": similarity_score
            })
        
        # Display results in a styled table
        df = pd.DataFrame(movie_data)
        
        # Style the dataframe
        def style_rating(val):
            if val != "N/A":
                try:
                    rating = float(val)
                    if rating >= 8.0:
                        color = "background-color: #d4edda; color: #155724"  # Green
                    elif rating >= 7.0:
                        color = "background-color: #fff3cd; color: #856404"  # Yellow
                    else:
                        color = "background-color: #f8d7da; color: #721c24"  # Red
                    return color
                except Exception:
                    pass
            return ""
        
        def style_similarity(val):
            if val >= 0.8:
                color = "background-color: #e2e3e5; font-weight: bold"
            elif val >= 0.6:
                color = "background-color: #f8f9fa"
            else:
                color = ""
            return color
        
        # Apply styling
        styled_df = df.style.applymap(style_rating, subset=['IMDB Rating'])
        styled_df = styled_df.applymap(style_similarity, subset=['Similarity Score'])
        
        st.dataframe(
            styled_df,
            use_container_width=True,
            hide_index=True
        )
        
        # Display statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Movies Found", len(movie_data))
        
        with col2:
            if movie_data:
                valid_ratings = [float(movie["IMDB Rating"]) for movie in movie_data if movie["IMDB Rating"] != "N/A"]
                if valid_ratings:
                    avg_rating = sum(valid_ratings) / len(valid_ratings)
                    st.metric("Average IMDB Rating", f"{avg_rating:.1f}")
        
        with col3:
            if movie_data:
                top_genres = {}
                for movie in movie_data:
                    if movie["Genres"] != "N/A":
                        for genre in movie["Genres"].split(", "):
                            top_genres[genre] = top_genres.get(genre, 0) + 1
                
                if top_genres:
                    most_common = max(top_genres, key=top_genres.get)
                    st.metric("Most Common Genre", most_common)
        
        # Show top 3 results with more detail
        if len(movie_data) >= 3:
            st.subheader("🏆 Top 3 Recommendations")
            for i, movie in enumerate(movie_data[:3], 1):
                with st.container():
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col1:
                        st.write(f"**#{i}**")
                    with col2:
                        st.write(f"**{movie['Title']}**")
                        st.write(f"Genres: {movie['Genres']}")
                    with col3:
                        st.write(f"⭐ {movie['IMDB Rating']}")
                        st.write(f"Similarity: {movie['Similarity Score']}")
                    
                    if i < 3:  # Don't add line after last item
                        st.divider()
    
    else:
        st.info("No movies found matching your criteria. Try adjusting your search query or filters.")
        
        # Suggest alternatives
        st.markdown("**💡 Suggestions:**")
        st.markdown("- Try a different search query with more general terms")
        st.markdown("- Lower the minimum IMDB rating threshold")
        st.markdown("- Select 'all' genres to see more results")

elif not user_query and search_button:
    st.warning("Please enter a search query to find movies.")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "Movie Finder • Powered by Vector Search & Streamlit"
    "</div>", 
    unsafe_allow_html=True
)
    