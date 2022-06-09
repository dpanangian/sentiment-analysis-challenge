import requests

url = 'http://127.0.0.1:8000/predict'
payload ={"text": "This app is so good!"}
resp = requests.post(url=url, json=payload)
print(resp.json())
