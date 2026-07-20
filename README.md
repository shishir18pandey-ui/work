Next step — test with a real image

Now swap in an actual screenshot to validate the part that actually matters for your incident bot: can it read text and describe realistic content correctly. Use the script from before:

python
import base64, json, requests

with open('your_screenshot.png', 'rb') as f:
    b64 = base64.b64encode(f.read()).decode('utf-8')

payload = {
    'model': '/app/models/Qwen3-VL-8B-Instruct',
    'messages': [{
        'role': 'user',
        'content': [
            {'type': 'text', 'text': 'Describe what is in this image. If it contains text, transcribe it exactly.'},
            {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{b64}'}}
        ]
    }],
    'temperature': 0.0,
    'max_tokens': 500
}

r = requests.post(
    'https://llm-api.iservebetter.idfcfirstbank.com/qwen3-vl-8b-svc/v1/chat/completions',
    headers={'Authorization': 'Bearer <YOUR_TOKEN>', 'Content-Type': 'application/json'},
    json=payload,
    verify=False
)
print(r.status_code)
print(r.json())

Use a real-world example that matches what customers will actually attach — e.g. a mobile app error screenshot, a debit card image, or an ID document. Check:

Does the transcribed text match exactly what's visible in the image?
Response time — note how long this takes for a normal-sized screenshot (this tells you if the resize logic we added is actually necessary for your typical use case, or if most incoming images are already small enough).
