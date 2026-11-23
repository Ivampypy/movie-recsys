from pydantic import BaseModel,Field


class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: list[int]

class ColdStartRequest(BaseModel):
    liked_movies: list[int] = Field(...,min_length=1)

class ColdStartResponse(BaseModel):
    liked_movies: list[int]
    recommendations: list[int]




