import json
import pickle

import torch
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from app.classifier.preprocessing import text_preprocessing

from .model import Model

with open("config.json") as json_file:
    config = json.load(json_file)


class SVM(Model):
    def __init__(self, pretrained=True):
        if pretrained:
            model_path = config["PARAMS"]["PRE_TRAINED_MODEL"]
            with open(model_path , 'rb') as f:
                self.model = pickle.load(f)
        else:
            self.cv= CountVectorizer(ngram_range=(1, 3), binary=True)
            self.tfidf= TfidfTransformer(smooth_idf=False)
            self.tfidf= SVC(C=1.0, kernel='linear', degree=3, gamma='auto',probability=True)
            self.components = [ ("vect", self.cv), 
                                ("tfidf", self.tfidf),
                                ("svm", self.svc)]
            
            self.model = Pipeline(self.components)

    def predict_on_batch(self, texts):
        return super().predict_on_batch(texts)


    def predict(self, text):
        return super().predict(text)

    def predict_probabilities(self, text):
        preprocessed_text = text_preprocessing(text)
        probabilities = self.model.predict_proba([preprocessed_text])
        probabilities = torch.from_numpy(probabilities)
        return probabilities
