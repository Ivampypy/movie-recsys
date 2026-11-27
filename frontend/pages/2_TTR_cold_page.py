import os
import sys

sys.path.append("..")

import requests
import streamlit as st
from src.data.load import load_movies

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")


@st.cache_data
def get_movies() -> dict[int, str]:
    movies = load_movies()
    return dict(zip(movies["movie_id"], movies["title"]))



st.subheader("Cold-Start Recommendations (New User)")

movies_dict = get_movies()
title_to_id = {title: m_id for m_id, title in movies_dict.items()}

selected_titles = st.multiselect(
    "Select movies you like:",
    options=list(title_to_id.keys()),
)

btn = st.button("Get recommendations", type="primary")

if btn:
    if not selected_titles:
        st.warning("Please select at least one movie.")
    else:
        liked_ids = [title_to_id[title] for title in selected_titles]
        with st.spinner("Loading..."):
            try:
                response = requests.post(
                    f"{API_URL}/recommend/cold_start",
                    json={"liked_movies": liked_ids},
                )
                if response.status_code == 200:
                    recs = response.json()["recommendations"]
                    st.subheader("Recommendations for you:")
                    for i, movie in enumerate(recs, 1):
                        title = movies_dict.get(movie)
                        st.write(f"{i}. {title}")
                else:
                    st.error(f"Error from API: status code {response.status_code}")
            except requests.exceptions.ConnectionError:
                st.error("Server is down")