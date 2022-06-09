import requests

url = 'http://127.0.0.1:8000/predict'
payload = {"text": "This app is so good!"}
headers = {"accept: application/json","Content-Type":"application/json","access_token": "932ea2a0-29b9-44ea-8f08-6c42f761b205"}
resp = requests.request("POST",url=url, json=payload, headers=headers)
print(resp.json())
