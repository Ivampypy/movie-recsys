from pydantic import BaseModel,Field


class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: list[int]

class ColdStartRequesr(BaseModel):
    liked_movies: list[int] = Field(...,min_length=1)



