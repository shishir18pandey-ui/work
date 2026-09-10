
import os
import re
import requests
import httpx
import json
from typing import List, Dict, Optional, Any
import asyncio
import logging
from new_flow.utils.observability import get_tracer

# Also import enhanced observability for better span handling (backward compatible)
# Can be disabled via ENHANCED_OTEL_ENABLED=false environment variable
_ENHANCED_OTEL_ENABLED = os.getenv("ENHANCED_OTEL_ENABLED", "true").lower() == "true"

if _ENHANCED_OTEL_ENABLED:
    try:
        from new_flow.utils.observability_enhanced import (
            get_tracer as get_tracer_enhanced,
            set_span_attribute,
            record_span_event,
            record_span_exception,
        )
        _ENHANCED_OTEL_AVAILABLE = True
    except ImportError:
        _ENHANCED_OTEL_AVAILABLE = False
        set_span_attribute = None
        record_span_event = None
        record_span_exception = None
else:
    _ENHANCED_OTEL_AVAILABLE = False
    set_span_attribute = None
    record_span_event = None
    record_span_exception = None


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
                'authentication' in error_str or 'access token' in error_str or 
                'invalid_token' in error_str):
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
            import traceback
            error_str = str(e).lower()
            logger.error(f"=== FULL EXCEPTION TRACEBACK ===")
            logger.error(traceback.format_exc())
            logger.error(f"=== EXCEPTION TYPE: {type(e).__name__} ===")
            logger.error(f"=== EXCEPTION MESSAGE: {e} ===")
            if ('connection' in error_str or 'timeout' in error_str or 
                'rate limit' in error_str or 'temporarily unavailable' in error_str or
                'service unavailable' in error_str or '502' in error_str or 
                '503' in error_str or '504' in error_str or '429' in error_str):
                last_error = e
                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt)
                    logger.info(f"Connection error (attempt {attempt + 1}/{max_retries}), "
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
    import time
    tracer = get_tracer(__name__)
    
    with tracer.start_as_current_span("call_llm_async") as span:
        # === INPUT ATTRIBUTES ===
        span.set_attribute("model", model)
        span.set_attribute("temperature", temperature)
        span.set_attribute("has_tools", bool(tools))
        
        # Estimate input tokens (rough estimate: ~4 chars per token)
        input_estimate = sum(len(str(m)) for m in messages) // 4
        span.set_attribute("input_tokens_estimate", input_estimate)

        # === ENHANCED OTEL: Use enhanced span attribute helper if available ===
        if _ENHANCED_OTEL_AVAILABLE and set_span_attribute:
            set_span_attribute(span, "input.messages_preview", str(messages[:2])[:200] if messages else "")
            set_span_attribute(span, "input.tools_count", len(tools) if tools else 0)
        
        # Tool info
        if tools:
            tool_names = [t.get("function", {}).get("name", "unknown") for t in tools]
            span.set_attribute("tool_count", len(tools))
            span.set_attribute("tool_names", ",".join(tool_names))
        
        # Start timing
        start_time = time.time()

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

            # === RESPONSE ATTRIBUTES ===
            duration_ms = (time.time() - start_time) * 1000
            span.set_attribute("request_duration_ms", round(duration_ms, 2))
            span.set_attribute("http_status_code", response.status_code)
            span.set_attribute("response_success", True)
            
            response.raise_for_status()
            data = response.json()

            message = data["choices"][0]["message"]
            finish_reason = data["choices"][0].get("finish_reason", "unknown")
            span.set_attribute("finish_reason", finish_reason)

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
                # Track tool calls made
                span.set_attribute("tool_calls_count", len(tool_calls))
                span.set_attribute("tool_calls_made", ",".join([tc.get("function", {}).get("name", "") for tc in tool_calls]))

            # Token usage
            if "usage" in data:
                usage = data["usage"]
                span.set_attribute("prompt_tokens", usage.get("prompt_tokens", 0))
                span.set_attribute("completion_tokens", usage.get("completion_tokens", 0))
                span.set_attribute("total_tokens", usage.get("total_tokens", 0))

            return {
                "response": response_text,
                "tool_calls": tool_calls,
                "finish_reason": finish_reason,
                "usage": {
                    "prompt_tokens": data["usage"]["prompt_tokens"],
                    "completion_tokens": data["usage"]["completion_tokens"],
                    "total_tokens": data["usage"]["total_tokens"]
                },
                "raw_message": message
            }

        except httpx.RequestError as e:
            span.set_attribute("response_success", False)
            span.set_attribute("error_message", str(e))
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
            span.set_attribute("response_success", False)
            span.set_attribute("error_message", f"Failed to parse API response: {str(e)}")
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
            span.set_attribute("response_success", False)
            span.set_attribute("error_message", str(e))
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
    """Async version of call_llm_streaming"""
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
