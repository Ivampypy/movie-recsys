import sys
sys.path.append(".")  


import streamlit as st
import requests
import pandas as pd
from src.data.load import load_movies, load_ratings


@st.cache_data
def get_movies() -> dict:
    movies = load_movies()
    return dict(zip(movies["movie_id"], movies["title"]))


@st.cache_data
def get_ratings() -> pd.DataFrame:
    return load_ratings()


st.title("Movie Recommender")
st.markdown(""" 
    End-to-End Movie Recommender System

    This is a demonstration project showcasing the full ML development lifecycle: from data exploration (EDA) to deployment.

    Model: Hybrid recommender (SVD + Popularity bias).

    Backend: REST API with FastAPI.

    Frontend: Streamlit.

    Enter a User ID (from 1 to 943) in the sidebar to request personalized recommendations via the API and explore the user's rating history.
""")


with st.sidebar:
    st.header("Settings")
    user_id = st.number_input(
        "Please input user id: ", min_value=1, max_value=943, value=1
    )
    btn = st.button("Get recommendations", type="primary")

with st.expander(" View user's rating history"):
    ratings_df = get_ratings()
    user_hist = ratings_df[ratings_df["user_id"] == user_id].copy()

    if not user_hist.empty:
        movies_dict = get_movies()
        user_hist["title"] = user_hist["movie_id"].map(movies_dict)
        user_hist = user_hist.sort_values("rating", ascending=False)
        st.dataframe(user_hist[["title", "rating"]], use_container_width=True)
    else:
        st.info("This user hasn't rated any movies yet.")

if btn:
    with st.spinner("Loading..."):
        try:
            response = requests.get(f"http://127.0.0.1:8000/recommend/{user_id}")
            if response.status_code == 200:
                recs = response.json()["recommendations"]
                st.subheader("Recommendations for user: ")
                movies_dict = get_movies()
                for i, movie in enumerate(recs, 1):
                    title = movies_dict.get(movie)
                    st.write(f"{i}. {title} ")

            else:
                st.error(f"Error from API: status code {response.status_code}")
        except requests.exceptions.ConnectionError:
            st.error("Server is down")

