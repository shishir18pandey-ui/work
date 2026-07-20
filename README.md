curl -k -X POST "https://llm-api.iservebetter.idfcfirstbank.com/qwen3-vl-8b-svc/v1/chat/completions" \
-H "Content-Type: application/json" \
-H "Authorization: Bearer <YOUR_TOKEN>" \
-d '{
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
          "image_url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
        }
      ]
    }
  ],
  "temperature": 0.0,
  "max_tokens": 500
}'
