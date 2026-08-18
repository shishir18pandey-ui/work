import os
import re
import requests
import httpx
import json
import traceback
from typing import List, Dict, Optional, Any
import asyncio
import logging
from new_flow.utils.observability import get_tracer

logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
OPENAI_API_BASE = os.environ.get('OPENAI_API_BASE')
OPENAI_MODEL_NAME = os.environ.get('OPENAI_MODEL_NAME')


class LLMConfig:
    def __init__(self):
        self.env = os.getenv('ENV', 'non-local')
        self.token = None
        self.url = OPENAI_API_BASE
        self.model_name = os.getenv('OPENAI_MODEL_NAME', '/app/models/MiniMax-M2.5')

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


async def run_crew_with_retry_async(crew_factory, max_retries=3, base_delay=1):
    from litellm.exceptions import AuthenticationError

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            crew_coro = crew_factory()
            result = await crew_coro
            return result
        except AuthenticationError as e:
            error_str = str(e).lower()
            if ('401' in error_str or 'invalid_token' in error_str or
                'authentication' in error_str or 'access token' in error_str):
                last_error = e
                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt)
                    logger.info(f"Token expired (attempt {attempt + 1}/{max_retries}), "
                               f"refreshing token and retrying in {delay}s...")
                    llm_config.set_llm_config()
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Max retries ({max_retries}) reached for crew execution")
            else:
                raise
        except Exception as e:
            error_str = str(e).lower()
            logger.error(f"LLM call failed: {type(e).__name__}: {e}")
            logger.error(traceback.format_exc())
            if ('connection' in error_str or 'timeout' in error_str or
                'rate limit' in error_str or 'temporarily unavailable' in error_str or
                'service unavailable' in error_str or '500' in error_str or
                '502' in error_str or '503' in error_str or '504' in error_str or
                '429' in error_str or 'unexpected error' in error_str or
                'internal server error' in error_str):
                last_error = e
                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt)
                    logger.info(f"Server error (attempt {attempt + 1}/{max_retries}), "
                               f"retrying in {delay}s... Error: {e}")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Max retries ({max_retries}) reached for crew execution")
            else:
                raise

    raise last_error


async def call_llm(
    messages: List[Dict[str, str]],
    tools: Optional[List[Dict[str, Any]]] = None,
    model: str = OPENAI_MODEL_NAME,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    first_attempt: bool = True
) -> Dict[str, Any]:
    tracer = get_tracer(__name__)
    with tracer.start_as_current_span("call_llm_async") as span:
        span.set_attribute("model", model)
        span.set_attribute("temperature", temperature)
        span.set_attribute("has_tools", bool(tools))

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

        headers = {
            "Authorization": f"Bearer {llm_config.token}",
            "Content-Type": "application/json"
        }

        try:
            async with httpx.AsyncClient(timeout=240.0, verify='./IDFCBANKCA.pem') as client:
                response = await client.post(
                    f'{llm_config.url}/chat/completions',
                    headers=headers,
                    json=payload
                )
            response.raise_for_status()
            data = response.json()
            message = data["choices"][0]["message"]
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

        except httpx.RequestError as e:
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
                return await call_llm(messages, tools, model, temperature, max_tokens, False)


async def call_llm_streaming(
    messages: List[Dict[str, str]],
    tools: Optional[List[Dict[str, Any]]] = None,
    model: str = OPENAI_MODEL_NAME,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    first_attempt: bool = True
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
        async with httpx.AsyncClient(timeout=30.0, verify='./IDFCBANKCA.pem') as client:
            async with client.stream('POST', llm_config.url, headers=headers, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line:
                        if line.startswith('data: '):
                            line = line[6:]
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
    except httpx.RequestError as e:
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
            async for chunk in call_llm_streaming(messages, tools, model, temperature, max_tokens, False):
                yield chunk
