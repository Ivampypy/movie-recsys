import sys
sys.path.append('.')

from contextlib import asynccontextmanager
from fastapi import FastAPI
import joblib as jlb
from src.models.svd import SVDRecommender
from api.schemas import RecommendationResponse,ColdStartRequest,ColdStartResponse
from ttr.TwoTowerRecs import TwoTowerRecommender
from src.data.load import load_ratings


models={}



@asynccontextmanager

async def lifespan(app: FastAPI) -> None:
    global model, popularity_dict,modelcold
    print("Loading model....")
    models["svd"] = jlb.load("models/svd_model.pkl")
    models["cold"]=TwoTowerRecommender()
    models['popularity_dict'] = jlb.load("models/train_popularity.pkl")
    models['cold'].load(load_ratings())

    print("Loading completed successfully!")

    models.clear()
    yield

app=FastAPI(title='Movie RecSys', lifespan=lifespan)

@app.get("/health")
def get_status() -> dict:
    return {"status": "ok"}


@app.get("/recommend/{user_id}", response_model=RecommendationResponse)
def get_recommendations(user_id: int) -> dict:
    recs = models["svd"].recommend_hybrid(user_id=user_id, movie_popularity=models["popularity_dict"])
    return {"user_id": user_id, "recommendations": recs}



