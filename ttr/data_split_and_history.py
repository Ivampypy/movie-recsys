from src.data.load import load_ratings
from torch.utils.data import Dataset
import pandas as pd
import torch


def prep_hist_n_data(emb_movies: torch.Tensor):
    train_d, test_d = [], []
    ratings = load_ratings()
    ratings = ratings[ratings["rating"] >= 4]

    ratings["movie_id"] = ratings["movie_id"] - 1
    ratings["user_id"] = ratings["user_id"] - 1
    ratings = ratings.sort_values(by="timestamp")

    us_films = ratings.groupby("user_id")

    for user_id, gr in us_films:
        movies = gr["movie_id"].to_list()
        if len(movies) < 3:
            continue
        split_b = int(len(movies) * 0.8)

        for idx in range(1, len(movies)):
            past_movies = movies[max(0, idx - 5) : idx]
            history_embs = emb_movies[past_movies, 1:].mean(dim=0)

            sample = (user_id, movies[idx], history_embs)
            if idx < split_b:
                train_d.append(sample)
            else:
                test_d.append(sample)

    return train_d, test_d


class MovieRetrievalDataset(Dataset):
    def __init__(self, samples: list):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        user_id, movies, history_embs = self.samples[idx]
        return (
            torch.tensor(user_id, dtype=torch.long),
            torch.tensor(movies, dtype=torch.long),
            history_embs,
        )
