from fastapi import FastAPI
import joblib as jlb
from src.models.svd import SVDRecommender

app = FastAPI(title="Movie RecSys API")

model = None
popularity_dict = None


@app.on_event("startup")
def load_artefact() -> None:
    global model, popularity_dict
    print("Loading model....")
    model = jlb.load("models/svd_model.pkl")
    popularity_dict = jlb.load("models/train_popularity.pkl")
    print("Loading completed successfully!")


@app.get("/health")
def get_status() -> dict:
    return {"status": "ok"}


@app.get("/recommend/{user_id}", response_model=SVDRecommender)
def get_recommendations(user_id: int) -> dict:
    recs = model.recommend_hybrid(user_id=user_id, movie_popularity=popularity_dict)
    return {"user_id": user_id, "recommendations": recs}
