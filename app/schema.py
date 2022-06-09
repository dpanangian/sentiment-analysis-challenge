from typing import Dict, List

from pydantic import BaseModel


class SentimentRequest(BaseModel):
    text: str

class SentimentRequests(BaseModel):
    text: List[str]


class SentimentResponse(BaseModel):
    text: str 
    probabilities: Dict[str, float]
    sentiment: str
    confidence: float

class SentimentResponses(BaseModel):
    __root__: List[SentimentResponse]
