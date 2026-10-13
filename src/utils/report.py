import pandas as pd

from src.utils.config import RATING_SCALE

class ShowInfo:

    def __init__(self,
                 ratings_data: pd.DataFrame,
                 movies_data: pd.DataFrame,
                 users_data: pd.DataFrame,
                 genres_data: pd.DataFrame):
        
        self.ratings_data=ratings_data
        self.movies_data=movies_data
        self.users_data=users_data
        self.genres_data=genres_data

    def info_ratings(self) -> None:
        ratings=self.ratings_data

        print()
        print(" === Ratings === ")
        print(ratings.head())
   
        print(f"Total rates: {len(ratings)}" )
        print(f"Unique users: {len(ratings['user_id'].unique())}")
        print(f"Unique movies: {len(ratings['movie_id'].unique())}")
        print(f"Range of rates: {RATING_SCALE[0]}-{RATING_SCALE[1]}")

    def info_movies(self) -> None:
        movies=self.movies_data

        print()
        print(" === MOVIES === ")
        print(movies.head())

        print()
        print(f"Total movies: {len(movies)}")

    def info_users(self) -> None:
        users=self.users_data 

        print()
        print(" === USERS === ")
        print(users.head())

    def info_genres(self) -> None:
        genres=self.genres_data

        print()
        print(" === GENRES === ")
        print(genres.head())


    def push_info(self) -> None:
        self.info_ratings()
        self.info_movies()
        self.info_users()
        self.info_genres()