from pathlib import Path
import pandas as pd


from src.utils.config import (
    RATINGS_FILE,
    MOVIES_FILE,
    USERS_FILE,
    GENRES_FILE,
)


def load_ratings(path: Path = RATINGS_FILE) -> pd.DataFrame:

    if not path.exists():
        raise FileNotFoundError(f"This {path} does not exist!")

    return pd.read_csv(
        path,
        sep="\t",
        names=["user_id", "movie_id", "rating", "timestamp"],
        engine="python",
    )


def load_movies(path: Path = MOVIES_FILE) -> pd.DataFrame:

    if not path.exists():
        raise FileNotFoundError(f"This {path} does not exist!")

    return pd.read_csv(
        path,
        sep="|",
        header=None,
        encoding="latin-1",
        usecols=[0, 1, 2],
        names=["movie_id", "title", "release_date"],
    )


def load_users(path: Path = USERS_FILE) -> pd.DataFrame:

    if not path.exists():
        raise FileNotFoundError(f"This {path} does not exist")

    return pd.read_csv(
        path, sep="|", names=["user_id", "age", "gender", "occupation", "zip_code"]
    )


def load_genres(path: Path = GENRES_FILE) -> pd.DataFrame:

    if not path.exists():
        raise FileNotFoundError(f"This {path} does not exist")

    return pd.read_csv(path, sep="|", header=None, names=["genre", "genre_id"])


if __name__ == "__main__":

    from src.utils.report import ShowInfo

    ratings = load_ratings()
    movies = load_movies()
    users = load_users()
    genres = load_genres()

    report = ShowInfo(
        ratings_data=ratings, movies_data=movies, users_data=users, genres_data=genres
    )

    report.push_info()
