from pydantic import BaseModel


class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: list[int]
