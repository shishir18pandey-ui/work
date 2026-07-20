import base64
import time
import requests

IMAGE_PATH = "/Users/shishir.pandey_tho/script/test_img.png"

API_URL = "https://llm-api.iservebetter.idfcfirstbank.com/qwen3-vl-8b-svc/v1/chat/completions"

TOKEN = "c2hpc2hpci5wYW5kZXlfdGhvOlNhdHR5QDY1NDMyMQ=="

with open(IMAGE_PATH, "rb") as f:
    b64 = base64.b64encode(f.read()).decode("utf-8")
    print(b64)

payload = {
    "model": "/app/models/Qwen3-VL-8B-Instruct",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Describe what is in this image. If it contains text, transcribe it exactly."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{b64}"
                    }
                }
            ]
        }
    ],
    "temperature": 0.0,
    "max_tokens": 500
}

headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json"
}

start = time.perf_counter()

response = requests.post(
    API_URL,
    headers=headers,
    json=payload,
    verify=False,   # Only if you're using a self-signed certificate
    timeout=120
)

elapsed = time.perf_counter() - start

print(f"Status Code : {response.status_code}")
print(f"Latency     : {elapsed:.2f} seconds")

try:
    data = response.json()
    print("\nAssistant Response:\n")
    print(data["choices"][0]["message"]["content"])
except Exception:
    print(response.text)
