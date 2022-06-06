from fastapi import APIRouter, Depends

from app.auth import authenticate
from app.schema import SentimentRequest, SentimentResponse

from .classifier.model import Model, get_model

router = APIRouter()



@router.get("/initialized")
def check_status(authenticated=Depends(authenticate)):
    """
    This endpoint can be used during startup to understand if the
    server is ready to take any requests, or is still loading.

    The recommended approach is to call this endpoint with a short timeout,
    like 500ms, and in case of no reply, consider the server busy.
    """
    return True
    
@router.post("/predict", response_model=SentimentResponse)
def predict(request: SentimentRequest, authenticated=Depends(authenticate), model: Model = Depends(get_model)):
    sentiment, confidence, probabilities = model.predict(request.text)
    return SentimentResponse(
        sentiment=sentiment, confidence=confidence, probabilities=probabilities
    )
