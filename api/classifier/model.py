import json
from abc import ABC

import torch

with open("config.json") as json_file:
    config = json.load(json_file)

class Model(ABC):
    def __init__(self, text):
        raise NotImplementedError()
    
    def predict_on_batch(self,texts):
        #TODO: refactor for effeciency
        result = []
        for text in texts:
            keys= ["text","sentiment","confidence","probabilities"]
            result.append(dict(zip(keys,list(self.predict(text)))))
        return result
            
    def predict(self,text):
        probabilities = self.predict_probabilities(text)
        confidence, predicted_class = torch.max(probabilities, dim=1)
        predicted_class = predicted_class.cpu().item()
        probabilities = probabilities.flatten().cpu().numpy().tolist()
        return (
                text,
                config["CLASS_NAMES"][predicted_class],
                confidence.cpu().numpy(),
                dict(zip(config["CLASS_NAMES"], probabilities)),
            )
    def predict_probabilities(self):
        raise NotImplementedError()
   
