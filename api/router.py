import json

from fastapi import APIRouter, Depends

from api.auth import authenticate
from api.classifier.bert import BERT
from api.classifier.svm import SVM
from api.schema import (SentimentRequest, SentimentRequests, SentimentResponse,
                        SentimentResponses)

router = APIRouter()

with open("config.json") as json_file:
    config = json.load(json_file)

def get_model():
    if config["MODEL"] == "BERT":
        return BERT() 
    if config["MODEL"] == "SVM":
        return SVM()     

model = get_model()

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
def predict(request: SentimentRequest, authenticated=Depends(authenticate)):
    """
    Predict the sentiment for the given text.
    """
    text, sentiment, confidence, probabilities = model.predict(request.text)
    return SentimentResponse(
        text=text, sentiment=sentiment, confidence=confidence, probabilities=probabilities
    )
@router.post("/predict_on_batch", response_model=SentimentResponses)
def predict_on_batch(request: SentimentRequests, authenticated=Depends(authenticate)):
    """
    Predict the sentiment for multiple texts.
    """
    result = model.predict_on_batch(request.text)
    return  SentimentResponses.parse_obj(result)
