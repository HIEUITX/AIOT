# src/test_api.py

import requests

data = {

    "value": 95,

    "rolling_mean": 30,

    "rolling_std": 1,

    "delta": 50,

    "zscore": 5,

    "stuck": 0
}

response = requests.post(
    "http://127.0.0.1:8000/detect-anomaly",
    json=data
)

print(response.json())