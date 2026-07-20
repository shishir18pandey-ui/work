 curl -k -X POST "https://llm-api.iservebetter.idfcfirstbank.com/gemma-4-31b/v1/chat/completions" \
-H "Content-Type: application/json" \
-H "Authorization: Bearer c2hpc2hpci5wYW5kZXlfdGhvOlNhdHR5QDY1NDMyMQ==" \
-d '{
  "model": "gemma-4-31B-it",
  "messages": [
    {
      "role": "user",
      "content": "Why is the sky blue?"
    }
  ],
  "temperature": 0.2,
  "max_tokens": 1024
}'

{"id":"chatcmpl-8f752c992478ab51","object":"chat.completion","created":1784575790,"model":"gemma-4-31B-it","choices":[{"index":0,"message":{"role":"assistant","content":"The short answer is a phenomenon called **Rayleigh scattering**.\n\nHere is the step-by-step breakdown of how it works:\n\n### 1. Sunlight is a spectrum of colors\nAlthough sunlight looks white to us, it is actually made up of all the colors of the rainbow (red, orange, yellow, green, blue, indigo, and violet). Light travels as **waves**, and each color has a different wavelength:\n*   **Red light** has long, lazy wavelengths.\n*   **Blue and violet light** have much shorter, choppier wavelengths.\n\n### 2. The Atmosphere acts as a filter\nThe Earth's atmosphere is filled with gases (mostly nitrogen and oxygen) and tiny particles. When sunlight hits the atmosphere, it strikes the molecules of these gases.\n\n### 3. Scattering the light\nBecause **blue light** travels in shorter, smaller waves, it crashes into the gas molecules more easily than the other colors do. When it hits these molecules, the blue light is scattered in every direction. \n\nMeanwhile, the longer wavelengths (like red and yellow) pass straight through the atmosphere without bumping into much, continuing on their path to the ground.\n\n### 4. Why we see blue\nWhen you look up, your eyes are catching that scattered blue light coming from every part of the sky.\n\n***\n\n### Common Follow-up Questions:\n\n**Why isn't the sky violet?**\nViolet light has an even shorter wavelength than blue light, so it actually scatters *more* than blue does. However, the sky doesn't look violet because:\n1.  The Sun emits much more blue light than violet light.\n2.  **Human eyes** are much more sensitive to blue than to violet. Our brains interpret the mix of scattered violet and blue light simply as pale blue.\n\n**Why is the sunset red?**\nWhen the sun is setting, the light has to travel through much more of the Earth's atmosphere to reach your eyes. By the time the light gets to you, the blue light has been scattered away completely, leaving only the **long-wavelength reds and oranges** to pass through.","refusal":null,"annotations":null,"audio":null,"function_call":null,"tool_calls":[],"reasoning":null},"logprobs":null,"finish_reason":"stop","stop_reason":106,"token_ids":null}],"service_tier":null,"system_fingerprint":null,"usage":{"prompt_tokens":19,"total_tokens":469,"completion_tokens":450,"prompt_tokens_details":null},"prompt_logprobs":null,"prompt_token_ids":null,"kv_transfer_params":null}%                                                                                                                                                                                                                shishir.pandey_tho@0325LTPB0124444 ~ % 



this is vl model im using :
vl_base_url: "https://llm-api.iservebetter.idfcfirstbank.com/qwen3-vl-8b-svc/v1"
vl_model_name: "/app/models/Qwen3-VL-8B-Instruct"
