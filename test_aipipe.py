import os
import httpx
from dotenv import load_dotenv

load_dotenv(override=True)

url = os.getenv("MODEL_BASE_URL", "").rstrip("/") + "/chat/completions"
token = os.getenv("MODEL_API_KEY", "")
model = os.getenv("MODEL_NAME", "")

print("URL:", url)
print("MODEL:", model)
print("TOKEN LENGTH:", len(token))
print("TOKEN STARTS WITH:", token[:6] + "..." if token else "EMPTY")

response = httpx.post(
    url,
    headers={
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    },
    json={
        "model": model,
        "messages": [
            {"role": "user", "content": "Reply with exactly: OK"}
        ],
    },
    timeout=30,
)

print("STATUS:", response.status_code)
print(response.text)
