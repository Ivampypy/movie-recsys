import pandas as pd
from surprise import SVD, Dataset, Reader
import numpy as np


class SVDRecommender:
    def __init__(self, n_factors=100, n_epochs=20, random_state=42):
        self.model = SVD(
            n_epochs=n_epochs, n_factors=n_factors, random_state=random_state
        )
        self.trainset = None  # Здесь будем хранить обучающую выборку surprise

    def fit(self, train_df: pd.DataFrame) -> None:
        """
        Обучает модель на train_df.
        Ожидаемые колонки: user_id, movie_id, rating.
        """
        # 1. Создай Reader со шкалой оценок (1, 5)
        reader = Reader(rating_scale=(1, 5))

        # 2. Загрузи данные в формат surprise (только 3 колонки: user, item, rating)
        data = Dataset.load_from_df(train_df[["user_id", "movie_id", "rating"]], reader)

        # 3. Преврати в полный trainset (build_full_trainset) и обучи модель
        self.trainset = data.build_full_trainset()
        self.model.fit(self.trainset)

    def predict(self, user_id: int, movie_id: int) -> float:
        """Предсказывает оценку для одной пары (юзер, фильм)."""
        return self.model.predict(user_id, movie_id).est

    def recommend(
        self, user_id: int, top_n: int = 10, candidate_movies: list = None
    ) -> list:
        if candidate_movies is None:
            candidate_movies = list(self.trainset._raw2inner_id_items.keys())

        predictions = []
        for movie_id in candidate_movies:
            rate = self.predict(user_id, movie_id)
            predictions.append((movie_id, rate))

        predictions.sort(key=lambda x: x[1], reverse=True)

        return [movie_id for movie_id, rate in predictions[:top_n]]

    def recommend_hybrid(
        self,
        user_id: int,
        top_n: int = 10,
        candidate_movies: list = None,
        movie_popularity: dict = None,
    ) -> list:

        if candidate_movies is None:
            candidate_movies = list(self.trainset._raw2inner_id_items.keys())

        predictions = []
        for movie_id in candidate_movies:
            svd_score = self.predict(user_id, movie_id)
            if movie_popularity is None:
                final_score = svd_score
            else:
                pop_score = np.log1p(movie_popularity.get(movie_id, 0))
                final_score = (svd_score * 0.8) + (pop_score * 0.2)
            predictions.append((movie_id, final_score))

        predictions.sort(key=lambda x: x[1], reverse=True)

        return [movie_id for movie_id, rate in predictions[:top_n]]
