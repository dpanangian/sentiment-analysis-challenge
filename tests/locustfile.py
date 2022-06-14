import json

from locust import HttpUser, task
from dotenv import load_dotenv
import os

load_dotenv()

class CelonisUser(HttpUser):
    @task
    def test_predict(self):
        payload = json.dumps({
                    "text": "This is crap. I realize that the profit turned out to be lower than expected because of the huge budget"
                    })
        headers = {
                    'access_token': os.getenv("API_KEY"),
                    'Content-Type': 'application/json'
                    }
        self.client.post("/predict", headers=headers, data=payload)
        
