
import os
import re
import requests
import json
from typing import List, Dict, Optional, Any
import asyncio
import logging
from observability import get_tracer


logger = logging.getLogger(__name__)


OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
OPENAI_API_BASE = os.environ.get('OPENAI_VL_API_BASE')
OPENAI_MODEL_NAME = os.environ.get('OPENAI_VL_MODEL_NAME')

class LLMConfig:
    def __init__(self):
        self.env = os.getenv('ENV', 'non-local')
        self.token = None
        self.url = OPENAI_API_BASE
        
    def _get_config(self) -> str:
        def _get_token():
            response = requests.post(
                url=os.getenv('ENT_AUTH_APPLICATION_TOKEN_URL'),
                headers={},
                data={
                    'client_id': os.getenv('ENT_AUTH_APPLICATION_CLIENT_ID'),
                    'client_secret': os.getenv('ENT_AUTH_APPLICATION_SECRET'),
                    'grant_type': 'client_credentials'
                }
            )
            response_data = response.json()
            return response_data['access_token']
        
        if self.env == 'local':
            url = self.url
            token = os.getenv('OPENAI_API_KEY')
        else: 
            try:
                if "-entauth" not in self.url:
                    logger.info("Converting OpenAI URL to Ent-Auth URL")
                    pattern = r"(https?://[^/]+)(/[^/]+)(/.*)"
                    match = re.match(pattern, self.url)
                    base_domain, model_path, version_path = match.groups()
                    url = f"{base_domain}{model_path}-entauth{version_path}"
                    logger.info("Ent-Auth URL generated: %s", url)
                else:
                    url = self.url
                token = _get_token()
                return url, token
            except Exception as e:
                logger.error(e)

            # retry
            pattern = r'(https?://[^/]+)(/[^/]+)(/.*)'
            match = re.match(pattern, self.url)
            base_domain, model_path, version_path = match.groups()
            url = f"{base_domain}{model_path}-entauth{version_path}"
            logger.info("Testing 2 Ent-Auth URL generated: %s", url)

            token = _get_token()
            return url, token

    def set_llm_config(self) -> dict:
        try:
            url, token = self._get_config()
            self.url = url
            self.token = token
            os.environ['OPENAI_API_KEY'] = token
            os.environ['OPENAI_API_BASE'] = url
            logger.info("OPENAI_API_KEY refreshed")
        except Exception as e:
            logger.error(f"OPENAI_API_KEY refresh failed: {e}")


    async def refresh_loop(self):
        while True:
            self.set_llm_config()
            await asyncio.sleep(600)

llm_config = LLMConfig()


def run_crew_with_retry(crew_factory, *args, max_retries=3, base_delay=1, **kwargs):
    import time
    from litellm.exceptions import AuthenticationError
    
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            return crew_factory(*args, **kwargs)
        except (AuthenticationError, Exception) as e:
            error_str = str(e).lower()
            if ('401' in error_str or 'invalid_token' in error_str or 
                'authentication' in error_str or 'access token' in error_str):
                last_error = e
                if attempt < max_retries:
                    # Exponential backoff: 1s, 2s, 4s, etc.
                    delay = base_delay * (10 ** attempt)
                    logger.info(f"Token expired (attempt {attempt + 1}/{max_retries}), "
                               f"refreshing token and retrying in {delay}s...")
                    llm_config.set_llm_config()
                    time.sleep(delay)
                else:
                    logger.error(f"Max retries ({max_retries}) reached for crew execution")
            else:
                raise
    
    raise last_error


def call_llm(
    messages: List[Dict[str, str]],
    tools: Optional[List[Dict[str, Any]]] = None,
    model: str = OPENAI_MODEL_NAME,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    first_attempt = True
) -> Dict[str, Any]:

    tracer = get_tracer(__name__)
    
    with tracer.start_as_current_span("call_llm") as span:
        span.set_attribute("model", model)
        span.set_attribute("temperature", temperature)
        if tools:
            span.set_attribute("has_tools", True)
        else:
            span.set_attribute("has_tools", False)

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }

        if max_tokens:
            payload["max_tokens"] = max_tokens

        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        # Set headers
        headers = {
            "Authorization": f"Bearer {llm_config.token}",
            "Content-Type": "application/json"
        }

        # Make API call
        try:
            response = requests.post(
                f'{llm_config.url}/chat/completions',
                headers=headers,
                json=payload,
                timeout=240,
                verify='./IDFCBANKCA.pem'
            )

            response.raise_for_status()
            data = response.json()

            message = data["choices"][0]["message"]

            # Extract response and tool calls
            response_text = message.get("content")
            tool_calls = None

            if message.get("tool_calls"):
                tool_calls = [
                    {
                        "id": tc["id"],
                        "type": tc["type"],
                        "function": {
                            "name": tc["function"]["name"],
                            "arguments": tc["function"]["arguments"]
                        }
                    }
                    for tc in message["tool_calls"]
                ]

            return {
                "response": response_text,
                "tool_calls": tool_calls,
                "finish_reason": data["choices"][0]["finish_reason"],
                "usage": {
                    "prompt_tokens": data["usage"]["prompt_tokens"],
                    "completion_tokens": data["usage"]["completion_tokens"],
                    "total_tokens": data["usage"]["total_tokens"]
                },
                "raw_message": message
            }

        except requests.exceptions.RequestException as e:
            # Return error in consistent format
            span.record_exception(e)
            return {
                "response": None,
                "tool_calls": None,
                "finish_reason": "error",
                "error": str(e),
                "usage": None,
                "raw_message": None
            }
        except (KeyError, json.JSONDecodeError) as e:
            span.record_exception(e)
            return {
                "response": None,
                "tool_calls": None,
                "finish_reason": "error",
                "error": f"Failed to parse API response: {str(e)}",
                "usage": None,
                "raw_message": None
            }
        except Exception as e:
            logger.error(e)
            span.record_exception(e)
            if first_attempt:
                llm_config.set_llm_config()
                return call_llm(messages, tools, model, temperature, max_tokens, False)


def call_llm_streaming(
    messages: List[Dict[str, str]],
    tools: Optional[List[Dict[str, Any]]] = None,
    model: str = OPENAI_MODEL_NAME,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    first_attempt = True
):
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": True
    }

    if max_tokens:
        payload["max_tokens"] = max_tokens

    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    headers = {
        "Authorization": f"Bearer {llm_config.token}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(
            llm_config.url,
            headers=headers,
            json=payload,
            stream=True,
            timeout=30
        )

        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    line = line[6:]  # Remove 'data: ' prefix

                if line == '[DONE]':
                    break

                try:
                    chunk_data = json.loads(line)
                    delta = chunk_data["choices"][0]["delta"]

                    yield {
                        "delta": delta.get("content"),
                        "tool_calls": delta.get("tool_calls"),
                        "finish_reason": chunk_data["choices"][0].get("finish_reason")
                    }
                except json.JSONDecodeError:
                    continue

    except requests.exceptions.RequestException as e:
        yield {
            "delta": None,
            "tool_calls": None,
            "finish_reason": "error",
            "error": str(e)
        }

    except Exception as e:
        logger.error(e)
        if first_attempt:
            llm_config.set_llm_config()
            return call_llm_streaming(messages, tools, model, temperature, max_tokens, False)









import os
import io
import base64
import logging
import requests
from typing import List, Dict, Optional
from PIL import Image
from llm import llm_config

logger = logging.getLogger(__name__)

VISION_MODEL_NAME = os.getenv("OPENAI_VL_MODEL_NAME", "/app/models/Qwen3-VL-8B-Instruct")
VISION_API_BASE = os.getenv("OPENAI_VL_API_BASE")

IMAGE_MIME_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/gif", "image/webp"}
PDF_MIME_TYPES = {"application/pdf"}

print(VISION_MODEL_NAME,"testing1.5")
print(VISION_API_BASE,"testing2")


MAX_IMAGE_DIMENSION = 1280
JPEG_QUALITY = 85


def _resize_image_if_needed(base64_data: str, mime_type: str, file_name: str) -> tuple:
    """
    Downscale image so its longest side <= MAX_IMAGE_DIMENSION px.
    Controls Qwen-VL visual-token count. Falls back to original data on
    any failure — never blocks the flow.
    """
    try:
        base64_data = base64_data.replace("\n", "").replace("\r", "").strip()
        raw_bytes = base64.b64decode(base64_data)
        img = Image.open(io.BytesIO(raw_bytes))

        width, height = img.size
        longest_side = max(width, height)

        if longest_side <= MAX_IMAGE_DIMENSION:
            return base64_data, mime_type

        scale = MAX_IMAGE_DIMENSION / longest_side
        new_size = (int(width * scale), int(height * scale))

        if img.mode != "RGB":
            img = img.convert("RGB")

        resized = img.resize(new_size, Image.LANCZOS)

        buffer = io.BytesIO()
        resized.save(buffer, format="JPEG", quality=JPEG_QUALITY)
        new_bytes = buffer.getvalue()

        logger.info(
            f"Resized image | file={file_name} "
            f"original={width}x{height} -> {new_size[0]}x{new_size[1]} "
            f"original_bytes={len(raw_bytes)} new_bytes={len(new_bytes)}"
        )

        return base64.b64encode(new_bytes).decode("utf-8"), "image/jpeg"

    except Exception as e:
        logger.error(f"Image resize failed | file={file_name} err={e} — using original")
        return base64_data, mime_type


def _describe_image(base64_data: str, mime_type: str, file_name: str) -> str:
    base64_data, mime_type = _resize_image_if_needed(base64_data, mime_type, file_name)

    data_uri = f"data:{mime_type};base64,{base64_data}"

    payload = {
        "model": VISION_MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Describe what is in this image. If it contains text "
                            "(screenshot, error message, document, ID card), transcribe "
                            "the visible text exactly. Be concise. No speculation."
                        )
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": data_uri}
                    }
                ]
            }
        ],
        "temperature": 0.0,
        "max_tokens": 800
    }
    
    headers = {
        "Authorization": f"Bearer {llm_config.token}",
        "Content-Type": "application/json"
    }

    base_url =  llm_config.url
    print(base_url,"testing1")

    try:
        response = requests.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=60,
            verify='./IDFCBANKCA.pem'
        )
        response.raise_for_status()
        data = response.json()
        description = data["choices"][0]["message"]["content"]
        logger.info(f"Image described | file={file_name}")
        return description.strip()
    except Exception as e:
        logger.error(f"Image description failed | file={file_name} err={e}")
        return "IMAGE UNREADABLE"


def _extract_pdf(base64_data: str, file_name: str) -> str:
    try:
        from pypdf import PdfReader

        pdf_bytes = base64.b64decode(base64_data)
        reader = PdfReader(io.BytesIO(pdf_bytes))

        text_parts = []
        for i, page in enumerate(reader.pages):
            page_text = (page.extract_text() or "").strip()
            if page_text:
                text_parts.append(f"[Page {i + 1}]\n{page_text}")

        if text_parts:
            combined = "\n\n".join(text_parts)
            if len(combined) > 5000:
                combined = combined[:5000] + "\n...(truncated)"
            logger.info(f"PDF extracted | file={file_name}")
            return combined
        return "PDF appears to be scanned; no extractable text."
    except Exception as e:
        logger.error(f"PDF extraction failed | file={file_name} err={e}")
        return "PDF UNREADABLE"


def _process_one(file: Dict) -> Optional[str]:
    file_name = file.get("fileName", "unknown")
    file_type = (file.get("fileType") or "").lower()
    encoding = (file.get("contentEncoding") or "").lower()
    content = file.get("fileContent")

    if not content or encoding != "base64":
        return None

    if file_type in IMAGE_MIME_TYPES:
        desc = _describe_image(content, file_type, file_name)
        return f"[Attached image: {file_name}]\n{desc}"

    if file_type in PDF_MIME_TYPES:
        text = _extract_pdf(content, file_name)
        return f"[Attached PDF: {file_name}]\n{text}"

    logger.warning(f"Skipping unsupported file type | file={file_name} type={file_type}")
    return None


def process_attachments(files: Optional[List[Dict]]) -> str:
    """
    Sync file processor. Returns "" if no files or any failure.
    Never raises — incident creation must succeed regardless.
    """
    if not files:
        return ""

    try:
        descriptions = []
        for f in files:
            result = _process_one(f)
            if result:
                descriptions.append(result)
        if not descriptions:
            return ""
        return "\n\n".join(descriptions)
    except Exception as e:
        logger.error(f"process_attachments failed: {e}")
        return ""










        shishir.pandey_tho@0325LTPB0124444 ~ % kubectl logs incident-agent-7576f69dd-gp4q8 
2026-07-21 13:08:28,130 - crew_main - INFO - OpenTelemetry tracer initialization started...
2026-07-21 13:08:28,167 - observability - INFO - HTTPX auto-instrumented
2026-07-21 13:08:28,169 - observability - INFO - Requests auto-instrumented
2026-07-21 13:08:28,169 - observability - INFO - OpenTelemetry initialized successfully
2026-07-21 13:08:28,169 - observability - INFO -   Service Name : genai-de-incident-agent
2026-07-21 13:08:28,169 - observability - INFO -   Environment  : uat
2026-07-21 13:08:28,169 - observability - INFO -   Endpoint     : http://ot-collector.tracing.svc.cluster.local:4318/v1/traces
2026-07-21 13:08:28,169 - crew_main - INFO - OpenTelemetry tracer initialized
2026-07-21 13:08:28,295 - crew_main - INFO - crew_main.py loaded successfully
2026-07-21 13:08:28,295 - crew_main - INFO - INCIDENT_BOT_ENABLED = true
2026-07-21 13:08:28,295 - crew_main - INFO - KAFKA_BROKER_URL = b-1.dcawscentraluatkafka0.fil3x5.c4.kafka.ap-south-1.amazonaws.com:9096,b-2.dcawscentraluatkafka0.fil3x5.c4.kafka.ap-south-1.amazonaws.com:9096,b-3.dcawscentraluatkafka0.fil3x5.c4.kafka.ap-south-1.amazonaws.com:9096
2026-07-21 13:08:28,295 - crew_main - INFO - KAFKA_TOPIC = GEN-AI-DE-INCIDENT-EVENTS
2026-07-21 13:08:28,300 - crew_main - INFO - FastAPI app initialized
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
2026-07-21 13:09:55,649 - crew_main - INFO - → Incoming request | incident=343579 has_comments=False
2026-07-21 13:09:55,649 - crew_main - INFO -   INCIDENT_BOT_ENABLED=true
2026-07-21 13:09:55,651 - crew_main - INFO -   DB lookup | incident=343579 exists=False
2026-07-21 13:09:55,677 - file_processor - ERROR - Image description failed | file=apple.jpeg err=401 Client Error: Unauthorized for url: https://llm-api.iservebetter.idfcfirstbank.com/qwen3-vl-8b-svc/v1/chat/completions
2026-07-21 13:09:55,677 - crew_main - INFO -   Files processed | incident=343579 count=1 has_description=True
2026-07-21 13:09:55,791 - crew_main - INFO -   DB saved | incident=343579
2026-07-21 13:09:55,909 - crew_main - INFO - ✓ Kafka published | incident=343579 event=new_incident
2026-07-21 13:09:55,910 - crew_main - INFO - ✓ New incident done | incident=343579
/app/models/Qwen3-VL-8B-Instruct testing1.5
https://llm-api.iservebetter.idfcfirstbank.com/qwen3-vl-8b-svc/v1 testing2
https://llm-api.iservebetter.idfcfirstbank.com/qwen3-vl-8b-svc/v1 testing1
INFO:     100.64.35.111:45800 - "POST /incident_agent/incident/create HTTP/1.1" 200 OK
shishir.pandey_tho@0325LTPB0124444 ~ % 






now tell me how we can fix this 
why it is nottaking entauth 




            
