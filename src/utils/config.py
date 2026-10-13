from pathlib import Path

PROJECT_ROOT=Path(__file__).resolve().parents[2]
ML_100K=PROJECT_ROOT/"data"/"raw"/"ml-100k"

RATINGS_FILE=ML_100K/"u.data"
MOVIES_FILE=ML_100K/"u.item"
USERS_FILE=ML_100K/"u.user"
GENRES_FILE=ML_100K/"u.genre"

RATING_SCALE=(1,5)
RANDOM_STATE=42
TEST_SIZE=0.2
MIN_RATINGS_PER_MOVIE=50
TOP_K=10
