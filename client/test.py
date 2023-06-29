import requests

res = requests.get('http://localhost:8484/ping/')
print(res.status_code)