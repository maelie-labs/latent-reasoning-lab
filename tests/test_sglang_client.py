import requests

url = "http://127.0.0.1:30000/v1/chat/completions"
payload = {
    "model": "Qwen/Qwen3-1.7B",
    "messages": [{"role": "user", "content": "What is 2 + 2?"}],
    "max_tokens": 50,
    "temperature": 0.0
}
resp = requests.post(url, json=payload, timeout=30)
print("Status:", resp.status_code)
print("Response:", resp.json()["choices"][0]["message"])
