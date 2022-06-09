import requests

url = 'http://127.0.0.1:8000/predict'
payload ={"text": "This app is a total waste of time!"}
resp = requests.post(url=url, json=payload)
print(resp.json())
