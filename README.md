
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
OPENAI_API_BASE = os.environ.get('OPENAI_API_BASE')
OPENAI_MODEL_NAME = os.environ.get('OPENAI_MODEL_NAME')

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
                if '-entauth' not in self.url:
                    pattern = r'(https?://[^/]+)(/[^/]+)(/.*)'
                    match = re.match(pattern, self.url)
                    base_domain, model_path, version_path = match.groups()
                    url = f"{base_domain}{model_path}-entauth{version_path}"
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








            INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
2026-07-20 21:05:07,212 - crew_main - INFO - → Incoming request | incident=12345 has_comments=False
2026-07-20 21:05:07,212 - crew_main - INFO -   INCIDENT_BOT_ENABLED=true
2026-07-20 21:05:07,214 - crew_main - INFO -   DB lookup | incident=12345 exists=False
2026-07-20 21:05:07,237 - file_processor - ERROR - Image description failed | file=pan_card.jpeg err=401 Client Error: Unauthorized for url: https://llm-api.iservebetter.idfcfirstbank.com/minimax-m2/v1/chat/completions
2026-07-20 21:05:07,237 - crew_main - INFO -   Files processed | incident=12345 count=1 has_description=True
2026-07-20 21:05:07,355 - crew_main - INFO -   DB saved | incident=12345
2026-07-20 21:05:07,459 - crew_main - INFO - ✓ Kafka published | incident=12345 event=new_incident
2026-07-20 21:05:07,460 - crew_main - INFO - ✓ New incident done | incident=12345
INFO:     100.64.35.111:50972 - "POST /incident_agent/incident/create HTTP/1.1" 200 OK
shishir.pandey_tho@032



     - name: OPENAI_API_KEY
              value: "{{ .Values.openai.api_key }}"
            - name: OPENAI_API_BASE
              value: "{{ .Values.openai.base_url }}"
            - name: OPENAI_MODEL_NAME
              value: "{{ .Values.openai.model_name }}"
            - name: VISION_API_BASE
              value: "{{ .Values.openai.vl_base_url }}"
            - name : VISION_MODEL_NAME
              value: "{{ .Values.openai.vl_model_name }}"
