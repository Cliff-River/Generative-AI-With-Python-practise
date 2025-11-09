"""
An application to search for movies based on a query.
- Use streamlit to create a web interface for the application.
"""

# %% packages
import streamlit as st
from data_prep import prepare_data

# %% Prepare data
vector_store = prepare_data()
st.title("Movie Finder")

# %%
