  

# Sentiment Analysis Challenge

## Demo
_____

The model is trained to classify sentiments (Irrelevant, Negative, Neutral and Positive).

 Here's a sample request to the API:

```bash
http POST http://127.0.0.1:8000/predict text="This is crap. I realize that the profit turned out to be lower than expected because of the huge budget"
```

The response:

```js
 {
    "text": "This is crap. I realize that the profit turned out to be lower than expected because of the huge budget",
    "probabilities": {
        "Irrelevant": 0.07386413961648941,
        "Negative": 0.7712374925613403,
        "Neutral": 0.07921858131885529,
        "Positive": 0.07567978650331497
    },
    "sentiment": "Negative",
    "confidence": 0.7712374925613403
}
```

A running API also can be accessed here https://celonis-sentiment-analysis.azurewebsites.net/docs with the access token

```sh
932ea2a0-29b9-44ea-8f08-6c42f761b205
```


## Installation
_____

Clone this repo:

```sh
git clone https://github.com/dpanangian/sentiment-analysis-challenge
cd sentiment-analysis-challenge
```

Install the dependencies:

```sh
pip install -r requirements.txt
```

Download the pre-trained models:

```sh
python scripts\download_models.py
```
## Configuration
_____

To use SVM replace **config.json** with :
```sh
{
    "MODEL": "BERT",
    "PARAMS":{
        "BERT_MODEL": "bert-base-uncased",
        "PRE_TRAINED_MODEL": "models/bert_state_dict.bin",
        "MAX_SEQUENCE_LEN": 64
    },
    "CLASS_NAMES": [
        "Irrelevant", 
        "Negative", 
        "Neutral", 
        "Positive"
    ]
}
```
To use BERT replace **config.json** with :
```sh
{
    "MODEL": "SVM",
    "PARAMS":{
        "PRE_TRAINED_MODEL": "models/svm.pkl"
    },
    "CLASS_NAMES": [
        "Irrelevant", 
        "Negative", 
        "Neutral", 
        "Positive"
    ]
}
```
## Test the setup
_____

Run the application:

```sh
uvicorn api.app:app 
```
the API will run on http://127.0.0.1:8000/

Available endpoints can be seen here http://127.0.0.1:8000/docs







