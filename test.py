import requests

"""
Set the host to
    1. 'localhost' if you want to run the server locally.
    2. the URL of the deployed environment if you want to send a request to the cloud server.
"""
host = "localhost:9696"
url = f"http://{host}/predict"

features = {
    "work_year": 2025,
    "experience_level": "mi",
    "job_title": "ml_engineer",
    "employee_residence": "us",
    "remote_ratio": 0.0,
    "company_location": "us",
    "company_size": "l"
}

result = requests.post(url, json=features).json()
print(result)