from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ML_100K = PROJECT_ROOT / "data" / "raw" / "ml-100k"
EMBS = PROJECT_ROOT / "models"

RATINGS_FILE = ML_100K / "u.data"
MOVIES_FILE = ML_100K / "u.item"
USERS_FILE = ML_100K / "u.user"
GENRES_FILE = ML_100K / "u.genre"
EMBS_ITEM = PROJECT_ROOT / "models" / "movie_features.pt"
EMBS_USER = PROJECT_ROOT / "models" / "user_static_features.pt"
INDEX_FILE = EMBS / "movies.index"
TTR_CONFIG = {
    "temperature": 0.07,
    "num_users": 943,
    "num_items": 1682,
    "id_lay_sz": 16,
    "lay_sz": 64,
    "output_sz": 64,
    "hist_dim": 384,
    "proj_hist_sz": 64,
}

RATING_SCALE = (1, 5)
RANDOM_STATE = 42
TEST_SIZE = 0.2
MIN_RATINGS_PER_MOVIE = 50
TOP_K = 10
