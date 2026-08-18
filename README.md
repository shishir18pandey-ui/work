import utils.minimax_tool_call_patch
import os
import json
import logging
import importlib

from utils.oracle_connection import close_oracle_async_pool
from utils.observability import init_telemetry, get_tracer
from utils.llm import llm_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

telemetry_endpoint = os.getenv("TELEMETRY_ENDPOINT")

if telemetry_endpoint:
    try:

        init_telemetry()

        logger.info("OpenTelemetry initialized for incident-manager worker")
    except Exception as e:
        logger.warning(f"Failed to initialize OpenTelemetry: {e}")
else:
    logger.warning("TELEMETRY_ENDPOINT not set — running without tracing")


import asyncio
import signal
import threading
import time
from confluent_kafka import Consumer, KafkaError, KafkaException
from dotenv import load_dotenv

load_dotenv()

KAFKA_BROKER   = os.getenv("KAFKA_BROKER_URL")
KAFKA_TOPIC    = os.getenv("KAFKA_TOPIC", "GEN-AI-DE-INCIDENT-EVENTS")
KAFKA_GROUP_ID = os.getenv("KAFKA_GROUP_ID", "gen-ai-de-incident-managers")
KAFKA_USERNAME = os.getenv("KAFKA_USERNAME")
KAFKA_PASSWORD = os.getenv("KAFKA_PASSWORD")


USE_NEW_FLOW = os.getenv("USE_NEW_FLOW", "false").lower() == "true"
FLOW_MODULE = "new_flow.flow" if USE_NEW_FLOW else "flow"

running = True
semaphore = None
active_tasks = {}  # {task: (incident_id, payload, start_time)}
TIMEOUT_SECONDS = 3600  # 1 hour
print(telemetry_endpoint,"tetsing1")
logger.info(f"Worker config loaded")
logger.info(f"  KAFKA_BROKER    : {KAFKA_BROKER}")
logger.info(f"  KAFKA_TOPIC     : {KAFKA_TOPIC}")
logger.info(f"  KAFKA_GROUP_ID  : {KAFKA_GROUP_ID}")
logger.info(f"  USE_NEW_FLOW    : {USE_NEW_FLOW} (module: {FLOW_MODULE})")


def get_kafka_consumer():
    config = {
        'bootstrap.servers': KAFKA_BROKER,
        'group.id': KAFKA_GROUP_ID,
        'auto.offset.reset': 'earliest',
        'enable.auto.commit': False,
        'security.protocol': 'SASL_SSL',
        'sasl.mechanism': 'SCRAM-SHA-512',
        'sasl.username': KAFKA_USERNAME,
        'sasl.password': KAFKA_PASSWORD,
        'session.timeout.ms': 60000,
        'max.poll.interval.ms': 7200000,
    }
    return Consumer(config)


async def process_message(incident_id: str, event_type: str, payload: dict):
    from utils.incident_db_async import get_incident_status_async, upsert_incident_payload_async

    tracer = get_tracer(__name__)

    async with semaphore:
        with tracer.start_as_current_span("process_message") as span:
            span.set_attribute("incident_id", incident_id)
            span.set_attribute("event_type", event_type)
            span.set_attribute("flow_module", FLOW_MODULE)

            logger.info(f"→ Processing | incident={incident_id} event={event_type} module={FLOW_MODULE}")

            send_rejection_to_servicenow_async = None  # populated once the module import succeeds

            try:
                current_status = await get_incident_status_async(incident_id)
                logger.info(f"  DB status | incident={incident_id} status={current_status}")
                span.set_attribute("current_status", current_status or "unknown")

                if event_type == "new_incident" and current_status in ['resolved', 'rejected']:
                    logger.info(f"  Skipping | incident={incident_id} reason=already_{current_status}")
                    span.set_attribute("skipped", True)
                    return

                if event_type == "additional_comments" and current_status not in ['in_progress']:
                    logger.info(f"  Skipping | incident={incident_id} reason=status_{current_status}_not_in_progress")
                    span.set_attribute("skipped", True)
                    return

                # Load whichever flow is active — everything from here on (success AND
                # fallback rejection) stays inside this one module. Nothing from the
                # other flow is ever touched.
                flow_module = importlib.import_module(FLOW_MODULE)
                IncidentManagementFlow = flow_module.IncidentManagementFlow
                send_rejection_to_servicenow_async = flow_module.send_rejection_to_servicenow_async

                flow = IncidentManagementFlow()
                flow.state.incident_id = incident_id
                flow.state.payload = payload

                if event_type == "additional_comments":
                    flow.state.current_comment = payload.get("additionalComments", "")
                    logger.info(f"  Flow type | incident={incident_id} type=additional_comments")
                    span.set_attribute("flow_type", "additional_comments")
                else:
                    logger.info(f"  Flow type | incident={incident_id} type=new_incident")
                    span.set_attribute("flow_type", "new_incident")

                await flow.akickoff()
                logger.info(f"✓ Flow completed | incident={incident_id}")
                span.set_attribute("flow_status", "completed")

            except asyncio.CancelledError:
                # timeout_checker_thread already sent rejection — just log + update DB
                logger.error(f"Task cancelled (timeout) | incident={incident_id}")
                span.set_attribute("flow_status", "cancelled")
                try:
                    await upsert_incident_payload_async(
                        incident_id,
                        json.dumps(payload),
                        'Failed - Timeout'
                    )
                except Exception as db_err:
                    logger.error(f"DB update failed after cancel | incident={incident_id} error={db_err}")
                raise  # re-raise so asyncio knows the task was cancelled

            except Exception as e:
                # Catches EVERYTHING else — including the module import itself
                # failing (e.g. flow_v2 missing/broken in this pod).
                logger.error(f"✗ Flow FAILED | incident={incident_id} module={FLOW_MODULE} error={e}", exc_info=e)
                span.set_attribute("flow_status", "failed")
                span.set_attribute("error", str(e))
                span.record_exception(e)

                if send_rejection_to_servicenow_async is not None:
                    try:
                        (status, info), _ = await send_rejection_to_servicenow_async(payload)
                        payload.setdefault('__agent_data', {}).setdefault('snow_logs', []).append({
                            "type": "rejection", "status": status, "response": info
                        })
                        logger.info(f"  Fallback rejection sent | incident={incident_id}")
                    except Exception as reject_err:
                        logger.error(f"  Rejection send also failed | incident={incident_id} error={reject_err}")
                else:
                    logger.error(f"  Skipped rejection send | incident={incident_id} reason=flow_module_import_failed")

                try:
                    await upsert_incident_payload_async(
                        incident_id,
                        json.dumps(payload),
                        'on_hold'
                    )
                except Exception as db_err:
                    logger.error(f"  DB update failed after flow failure | incident={incident_id} error={db_err}")


def timeout_checker_thread():
    global active_tasks, running
    while running:
        time.sleep(10)
        now = time.time()
        for task, (incident_id, payload, start_time) in list(active_tasks.items()):
            if task not in active_tasks:
                continue
            if task.done():
                continue
            if now - start_time > TIMEOUT_SECONDS:
                logger.error(f"Timeout | incident={incident_id} - sending rejection")
                if task in active_tasks:
                    del active_tasks[task]
                task.cancel()
                logger.info(f"Task cancelled for timeout | incident={incident_id}")


async def run_consumer():
    global semaphore, active_tasks

    asyncio.create_task(llm_config.refresh_loop())
    logger.info("LLM config initialized")

    semaphore = asyncio.Semaphore(10)
    active_tasks = {}

    tracer = get_tracer(__name__)
    consumer = get_kafka_consumer()

    timeout_thread = threading.Thread(target=timeout_checker_thread, daemon=True)
    timeout_thread.start()
    logger.info("Timeout checker thread started")

    try:
        consumer.subscribe([KAFKA_TOPIC])
        logger.info(f"✓ Worker started")
        logger.info(f"  Topic    : {KAFKA_TOPIC}")
        logger.info(f"  Group    : {KAFKA_GROUP_ID}")
        logger.info(f"  Broker   : {KAFKA_BROKER}")

        while running:
            msg = await asyncio.get_event_loop().run_in_executor(
                None, lambda: consumer.poll(1.0)
            )

            if msg is None:
                continue

            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    logger.debug(f"End of partition {msg.partition()}")
                    continue
                else:
                    logger.error(f"Kafka error: {msg.error()}")
                    raise KafkaException(msg.error())

            incident_id = None
            try:
                raw = msg.value().decode('utf-8')
                data = json.loads(raw)

                incident_id = data.get("incident_id")
                event_type  = data.get("event_type", "new_incident")
                payload     = data.get("payload", {})

                logger.info(f"→ Message received | incident={incident_id} event={event_type} partition={msg.partition()} offset={msg.offset()}")

                if not incident_id:
                    logger.warning("Message missing incident_id, skipping")
                    consumer.commit(message=msg)
                    continue

                with tracer.start_as_current_span("kafka_message_received") as span:
                    span.set_attribute("incident_id", incident_id)
                    span.set_attribute("event_type", event_type)
                    span.set_attribute("partition", msg.partition())
                    span.set_attribute("offset", msg.offset())

                task = asyncio.create_task(process_message(incident_id, event_type, payload))
                active_tasks[task] = (incident_id, payload, time.time())

                # Auto-remove from event loop when done, and surface any exception
                # that would otherwise be silently dropped by fire-and-forget tasks.
                def cleanup_done(t, _incident_id=incident_id):
                    if t in active_tasks:
                        del active_tasks[t]
                    if t.cancelled():
                        return
                    exc = t.exception()
                    if exc:
                        logger.error(
                            f"✗ Unhandled task exception | incident={_incident_id} error={exc}",
                            exc_info=exc
                        )
                task.add_done_callback(cleanup_done)

                consumer.commit(message=msg)
                logger.info(f"✓ Offset committed | incident={incident_id}")

            except Exception as e:
                logger.error(f"✗ Message handling FAILED | incident={incident_id} error={e}")
                await asyncio.sleep(5)

    except KeyboardInterrupt:
        logger.info("Worker interrupted")
    finally:
        logger.info("Closing consumer...")
        consumer.close()
        logger.info("Consumer closed cleanly")


def handle_shutdown(signum, frame):
    global running
    logger.info(f"Shutdown signal {signum} received...")
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.call_soon(lambda: asyncio.create_task(close_oracle_async_pool()))
        else:
            asyncio.run(close_oracle_async_pool())
    except RuntimeError:
        asyncio.run(close_oracle_async_pool())
    running = False


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, handle_shutdown)
    signal.signal(signal.SIGINT, handle_shutdown)

    logger.info("Starting incident-manager worker...")
    asyncio.run(run_consumer())


    this is util.llm.py
    
import os
import re
import requests
import httpx
import json
from typing import List, Dict, Optional, Any
import asyncio
import logging
from utils.observability import get_tracer


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



this si new_flow.utils.llm.py

import os
import re
import requests
import httpx
import json
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

            print("========== LLM CONFIG ==========")
            print("URL:", url)
            print("TOKEN:", token)
            print("================================")

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
    tracer = get_tracer(__name__)
    
    with tracer.start_as_current_span("call_llm_async") as span:
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

