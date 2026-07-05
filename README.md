from glob import glob
import json
import asyncio
from utils.logs import execute_elk_query
from utils.oracle_connection import execute_oracle_query_async
from crewai.tools import BaseTool
import os

from utils.http_calls import http_client_post_async, http_client_get_async

os.environ["OTEL_SDK_DISABLED"] = "true"

from glob import glob

import traceback
import logging
import threading

logger = logging.getLogger(__name__)


import json as _json
import time as _time
import base64 as _base64
import datetime as _datetime
import httpx
from collections import defaultdict

JAEGER_API_BASE = os.getenv("JAEGER_API_BASE", "https://tracing.uat-opt.idfcfirstbank.com/api")
JAEGER_AUTH_TOKEN = os.getenv("JAEGER_AUTH_TOKEN")

_JAEGER_EXCLUDED_OPS = {
    'get', 'set', 'sql:prepare', 'sql:query', 'sql-conn-query', 'sql-rows-next',
    'sql-tx-begin', 'sql-tx-commit', 'sql:exec', 'sql-prepare', 'sql-conn-exec',
    'sql-stmt-close',
    'sql.conn.prepare', 'sql.stmt.query', 'sql.stmt.exec', 'sql.stmt.close',
    'sql.tx.begin', 'sql.tx.commit', 'sql.conn.exec', 'sql.rows.next',
    'persistence.sql.findSessionBySignature', 'persistence.sql.toRequest',
    'persistence.sql.GetConcreteClient',
}
_JAEGER_EXCLUDED_SERVICES = {
    'RISK-CONTROL-REQUEST-METADATA-PROCESSOR',
    'RISK-CONTROL-USER-METADATA-PROCESSOR',
    'OAUTH-SERVER',
}
_JAEGER_MAX_PAYLOAD_CHARS = 400


def _jaeger_auth_headers():
    if JAEGER_AUTH_TOKEN:
        return {"Authorization": f"Basic {JAEGER_AUTH_TOKEN}"}
    return {}


def _jaeger_us_to_ist(microseconds_ts):
    from datetime import timezone, timedelta
    seconds = microseconds_ts / 1_000_000
    dt_utc = _datetime.datetime.fromtimestamp(seconds, tz=timezone.utc)
    ist_tz = timezone(timedelta(hours=5, minutes=30))
    return dt_utc.astimezone(ist_tz).strftime('%Y-%m-%d %H:%M:%S IST')


def _jaeger_trace_has_error(trace) -> bool:
    for span in trace.get("spans", []):
        for tag in span.get("tags", []):
            if tag.get("key") == "http.status_code":
                code = str(tag.get("value", ""))
                if code and not code.startswith("2"):
                    return True
            if tag.get("key") == "error" and tag.get("value") in (True, "true", "True"):
                return True
        for log in span.get("logs", []):
            for f in log.get("fields", []):
                if f.get("key") == "level" and str(f.get("value", "")).lower() in ("error", "fatal", "critical"):
                    return True
    return False


def _jaeger_process_trace(trace):
    spans = trace.get("spans", [])
    processes = trace.get("processes", {})

    id_to_span = {s["spanID"]: s for s in spans}
    pid_to_service = {pid: p["serviceName"] for pid, p in processes.items()}
    children_map = defaultdict(list)

    all_ids = set(id_to_span.keys())
    child_ids = set()

    for span in spans:
        for ref in span.get("references", []):
            parent_id = ref.get("spanID")
            if parent_id in id_to_span:
                children_map[parent_id].append(span["spanID"])
                child_ids.add(span["spanID"])

    roots = all_ids - child_ids
    if not roots:
        return []

    result = []

    def process_span(span_id, depth=0):
        span = id_to_span[span_id]
        indent = "    " * depth
        op_name = span.get("operationName", "")
        service_name = pid_to_service.get(span.get("processID"), "")

        if depth == 0:
            result.append("Time: " + _jaeger_us_to_ist(span["startTime"]))

        is_noise = op_name in _JAEGER_EXCLUDED_OPS or service_name in _JAEGER_EXCLUDED_SERVICES

        if not is_noise:
            result.append(f"{indent}{span_id} - {service_name} - {op_name}")
            for tag in span.get("tags", []):
                if tag.get("key") == "http.status_code":
                    result.append(f"{indent}HTTP Status: {tag.get('value')}")

            for log in span.get("logs", []):
                fields = {f.get("key"): f.get("value") for f in log.get("fields", [])}

                level = str(fields.get("level", "")).lower()
                event = fields.get("event", "")
                if level in ("error", "fatal", "critical") and event:
                    cls = fields.get("Class") or fields.get("class") or ""
                    method = fields.get("Method") or fields.get("method") or ""
                    ctx = f" [{cls}.{method}]" if (cls or method) else ""
                    result.append(f"{indent}  ERROR{ctx}: {event}")

                for payload_key in ("request", "response"):
                    if fields.get(payload_key):
                        result.append(f"{indent}  {payload_key}: {str(fields[payload_key])[:_JAEGER_MAX_PAYLOAD_CHARS]}")

        for child_id in children_map.get(span_id, []):
            process_span(child_id, depth + 1)

    for root_id in roots:
        process_span(root_id)

    return result


async def _jaeger_fetch(service: str, tag_name: str, tag_value: str):
    now_us = int(_time.time() * 1_000_000)
    start_us = now_us - 120 * 3600 * 1_000_000
    end_us = now_us

    if tag_name == "mobile_number" and not tag_value.startswith("+"):
        if len(tag_value) == 10:
            tag_value = f"+91{tag_value}"

    tags = _json.dumps({tag_name: tag_value})
    headers = _jaeger_auth_headers()
    params = {"service": service, "start": start_us, "end": end_us, "limit": 100, "tags": tags}

    logger.info(f"[JAEGER] service={service} {tag_name}={tag_value}")

    try:
        async with httpx.AsyncClient(timeout=45) as client:
            response = await client.get(f"{JAEGER_API_BASE}/traces", params=params, headers=headers)
            if response.status_code != 200:
                logger.error(f"[JAEGER] API returned {response.status_code}: {response.text}")
                return {"service": service, "tag_name": tag_name, "tag_value": tag_value,
                        "total_traces_scanned": 0, "failed_traces": [],
                        "error": f"Jaeger API returned {response.status_code}"}
            data = response.json().get("data", [])
    except Exception as e:
        logger.error(f"[JAEGER] Request failed: {e}")
        return {"service": service, "tag_name": tag_name, "tag_value": tag_value,
                "total_traces_scanned": 0, "failed_traces": [], "error": str(e)}

    if not data:
        return {"service": service, "tag_name": tag_name, "tag_value": tag_value,
                "total_traces_scanned": 0, "failed_traces": [],
                "message": f"No traces found for {tag_name}={tag_value} in {service}"}

    failed_traces_raw = [t for t in data if _jaeger_trace_has_error(t)]

    failed_traces_text = []
    for trace in failed_traces_raw:
        lines = _jaeger_process_trace(trace)
        if lines:
            failed_traces_text.append("\n".join(lines))

    logger.info(f"[JAEGER] Done: scanned={len(data)} failed={len(failed_traces_text)}")

    return {
        "service": service,
        "tag_name": tag_name,
        "tag_value": tag_value,
        "total_traces_scanned": len(data),
        "total_failed": len(failed_traces_text),
        "failed_traces": failed_traces_text
    }


# ─────────────────────────────────────────
# TOKEN MANAGER — CBS only, unchanged
# ─────────────────────────────────────────

class GiftWrappingTokenManager:

    def __init__(self):
        self.token = None
        self.token_expiry = None
        self._token_lock = threading.Lock()

    async def _get_token(self):
        client_id = os.getenv('GIFTWRAP_ENT_AUTH_CLIENT_ID')
        client_secret = os.getenv('GIFTWRAP_ENT_AUTH_CLIENT_SECRET')
        oauth_url = os.getenv('GIFTWRAP_ENT_AUTH_TOKEN_URL')

        if not client_id or not client_secret:
            raise ValueError("GIFTWRAP_ENT_AUTH_CLIENT_ID and GIFTWRAP_ENT_AUTH_CLIENT_SECRET must be set")

        form_data = {
            'client_id': client_id,
            'client_secret': client_secret,
            'grant_type': 'client_credentials'
        }

        headers = {
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        response = await http_client_post_async(url=oauth_url, headers=headers, data=None, form_data=form_data)
        json_resp = response.json
        return json_resp['access_token']

    async def get_valid_token(self):
        import time
        with self._token_lock:
            if self.token is None or self.token_expiry is None or time.time() >= self.token_expiry:
                self.token = await self._get_token()
                self.token_expiry = time.time() + (30 * 60) - 60
            return self.token

giftwrap_token_manager = GiftWrappingTokenManager()

async def make_giftwrap_api_call_async(endpoint: str, method: str, parameters: dict, headers: dict = None, query_params: dict = None):
    import uuid

    base_url = os.getenv('GIFTWRAP_API_BASE_URL')

    token = await giftwrap_token_manager.get_valid_token()

    request_headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json',
        'User-Agent': 'incident-agent/1.0',
        'X-B3-Sampled': '1',
        'X-B3-SpanId': f'{uuid.uuid4()}',
        'X-B3-TraceId': f'{uuid.uuid4()}'
    }

    if headers:
        request_headers.update(headers)

    url = f"{base_url}{endpoint}"
    if query_params:
        query_string = '&'.join([f"{k}={v}" for k, v in query_params.items() if v is not None])
        if query_string:
            url = f"{url}?{query_string}"

    logger.info(f"[GiftWrapping API Async] Calling {method} {url}")
    logger.info(f"[GiftWrapping API Async] Parameters: {parameters}")
    logger.info(f"[GiftWrapping API Async] Headers: {request_headers}")

    if method == 'GET':
        response = await http_client_get_async(url=url, headers=request_headers)
    elif method == 'POST':
        response = await http_client_post_async(url, headers=request_headers, json=parameters)
    else:
        raise ValueError(f"Unsupported HTTP method: {method}")

    return response.json


# ─────────────────────────────────────────
# ENTRY POINT — app-based routing
# ─────────────────────────────────────────

def get_tool_list(app: str = "cbs"):
    app_key = app.lower().strip()
    logger.info(f"Loading tools for app: {app_key}")

    if app_key == "cbs":
        return _load_tools_from_folder("./tools/cbs", app="cbs", include_normalize=True)
    elif app_key == "optimus":
        return _load_tools_from_folder("./tools/optimus", app="optimus", include_normalize=False)
    else:
        logger.warning(f"Unknown app '{app}' — defaulting to CBS tools")
        return _load_tools_from_folder("./tools/cbs", app="cbs", include_normalize=True)


# ─────────────────────────────────────────
# SHARED TOOL LOADER
# ─────────────────────────────────────────

def _load_tools_from_folder(folder: str, app: str, include_normalize: bool = False):
    json_tool_list = glob(f'{folder}/*.json')

    tool_parameter_list = []
    for path in json_tool_list:
        with open(path, 'r') as f:
            d = json.loads(f.read())
            tool_parameter_list += d

    tool_list = []
    for tool in tool_parameter_list:
        logger.info(f"Adding tool: {tool['name']} (type={tool.get('type')}, db_instance={tool.get('db_instance', 'main')})")
        tool_list.append(_make_tool(tool, app))

    if include_normalize:
        tool_list.append(NormalizeAccountTool())

    return tool_list


def _make_tool(tool, app: str):
    tool_name = tool['name']
    tool_type = tool.get('type', 'UNKNOWN')
    tool_query = tool.get('query', '')

    # Capture app and db_instance in closure for SQL tools
    _app = app
    _db_instance = tool.get('db_instance', 'main')

    if not tool.get('parameters'):
        tool['parameters'] = [{'name': 'key', 'description': 'Input value'}]

    param_name = tool['parameters'][0]['name']
    param_description = tool['parameters'][0]['description']

    # ── SQL ──
    if tool_type == 'SQL':
        class SQLTool(BaseTool):
            name: str = tool_name
            description: str = (
                tool['purpose'] + '\nEnter the ' + param_name + ' in key, it should contain ' + param_description
            )

            def regularize_id(self, incoming_id: str) -> str:
                raw = str(incoming_id).strip()
                raw = raw.removeprefix('003')
                return raw.lstrip('0')

            def _run(self, key: str):
                return asyncio.run(self._arun(key))

            async def _arun(self, key: str):
                clean_value = self.regularize_id(str(key).strip())

                # CBS-specific account number formatting
                if self.name in ["verify_sms_required_flag", "get_account_level_mobile",
                                 "get_email_preference_flag", "check_posting_restriction_mask",
                                 "find_ucic_from_account"]:
                    processed_value = f"003{clean_value.zfill(16)}"
                elif self.name in ["get_account_level_email", "get_ucic_email",
                                   "check_transaction_freeze", "check_npa_debit_freeze"]:
                    processed_value = f"{clean_value.zfill(16)}"
                elif self.name in ["find_account_from_ucic"]:
                    clean_value = clean_value[:-1]
                    processed_value = f"{clean_value.zfill(16)}"
                else:
                    # Optimus SQL tools and any future apps — no special formatting
                    processed_value = clean_value

                final_query = tool_query.replace(f":{param_name}", f"'{processed_value}'")
                logger.info(f"[{self.name}] app={_app} db={_db_instance} query={final_query}")
                try:
                    return await execute_oracle_query_async(
                        final_query,
                        app=_app,
                        db_instance=_db_instance
                    )
                except Exception as e:
                    return f"Query Error: {str(e)}"

        return SQLTool()

    # ── ELK ──
    elif tool_type == 'ELK':
        class ELKTool(BaseTool):
            name: str = tool['name']
            description: str = (
                tool['purpose'] + '\nEnter the ' + tool['parameters'][0]['name'] +
                ' in key, it should contain ' + tool['parameters'][0]['description']
            )

            def _run(self, key: str):
                return asyncio.run(self._arun(key))

            async def _arun(self, key: str):
                try:
                    pname = tool['parameters'][0]['name']
                    query_json = tool['query']
                    result = await execute_elk_query(query_json, parameter_name=pname, parameter_value=key)
                    return result
                except Exception as e:
                    return f"ELK query failed: {str(e)}"

        return ELKTool()

    # ── JAEGER ──
    elif tool_type == 'JAEGER':
        from pydantic import BaseModel

        jaeger_service = tool.get('service', '')

        class JaegerToolSchema(BaseModel):
            tag_name: str
            tag_value: str

        class JaegerTool(BaseTool):
            name: str = tool_name
            description: str = (
                tool['purpose']
                + f"\nThis queries the '{jaeger_service}' service."
                + "\ntag_name: identifier type (ucic, customer_id, mobile_number, username, user_id)"
                + "\ntag_value: the actual identifier value"
                + "\nNote: if ucic/customer_id is provided, the tool automatically tries "
                  "both tag keys internally — you do not need to retry manually."
                + "\nOnly failed/error traces are returned (non-2xx HTTP status, error tags, "
                  "or error-level log events). If no failures are found, the issue may not be "
                  "reproducible in logs, or login is working fine for this identifier."
            )

            args_schema: type = JaegerToolSchema

            def _run(self, tag_name: str, tag_value: str):
                return asyncio.run(self._arun(tag_name, tag_value))

            async def _arun(self, tag_name: str, tag_value: str):
                tag_name = tag_name.strip().lower()
                tag_value = str(tag_value).strip()

                tried = []
                attempt_order = [tag_name]

                if tag_name in ("ucic", "customer_id", "individualucic", "entityucic"):
                    for candidate in ("ucic", "customer_id"):
                        if candidate not in attempt_order:
                            attempt_order.append(candidate)

                last_result = None

                for candidate_tag in attempt_order:
                    tried.append(candidate_tag)
                    try:
                        result = await _jaeger_fetch(
                            service=jaeger_service,
                            tag_name=candidate_tag,
                            tag_value=tag_value
                        )
                    except Exception as e:
                        logger.error(f"[JaegerTool {self.name}] failed on tag_name={candidate_tag}: {e}")
                        result = {"total_traces_scanned": 0, "failed_traces": [], "error": str(e)}

                    last_result = result

                    scanned = result.get("total_traces_scanned", 0)
                    if scanned and scanned > 0:
                        result["tag_name_used"] = candidate_tag
                        result["tag_names_tried"] = tried
                        logger.info(f"[JaegerTool {self.name}] found {scanned} traces using tag_name={candidate_tag}")
                        return result

                logger.info(f"[JaegerTool {self.name}] no traces found after trying tag_names={tried}")
                if last_result is not None:
                    last_result["tag_names_tried"] = tried
                    return last_result
                return {"total_traces_scanned": 0, "failed_traces": [], "tag_names_tried": tried}

        return JaegerTool()

    # ── UNKNOWN ──
    else:
        class TempTool(BaseTool):
            name: str = tool_name
            description: str = tool['purpose']

            def _run(self, *args, **kwargs):
                return "NOT AVAILABLE"

            async def _arun(self, key: str):
                return "NOT AVAILABLE"

        return TempTool()


# ─────────────────────────────────────────
# SHARED TOOLS
# ─────────────────────────────────────────

class NormalizeAccountTool(BaseTool):
    name: str = "normalize_account_number"
    description: str = (
        "Normalizes account numbers to a standard format for comparison. "
        "Input: Account number in any format. Output: Normalized account number. "
        "Use this to compare account numbers from different sources."
    )

    def _run(self, account_number: str):
        return asyncio.run(self._arun(account_number))

    async def _arun(self, account_number: str):
        raw = str(account_number).strip()
        raw = raw.removeprefix('003')
        return raw.lstrip('0')




        this is tool.py

        now this is llm.py
        
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


flow.py
from pydantic import BaseModel
from typing import List, Optional, Dict
import httpx
import os
import json
import re
from dotenv import load_dotenv
from crewai.flow.flow import Flow, start, listen, router
from crewai.flow.persistence import persist
from utils.incident_db_async import upsert_incident_payload_async
from agents.debugger import run_backend_resolver_crew_async
from agents.context_builder import run_incident_context_crew_async
from agents.intent_classifier import run_intent_classifier_crew_async
from utils.llm import run_crew_with_retry_async

load_dotenv()

CA_CERT_FILE = os.getenv("CA_CERT_FILE", "./IDFCBANKCA.pem")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "/app/models/gpt-oss-120b")
CHAT_COMPLETIONS_URL = f"{OPENAI_API_BASE}/chat/completions"

import logging
from utils.observability import get_tracer

logger = logging.getLogger(__name__)


def extract_json_from_output(output: str) -> dict:
    if not output or not output.strip():
        logger.warning("Empty output received, returning fallback response")
        return {"diagnosis": "Unable to process incident", "solution": "Please try again later", "questions": [], "resolved": "no"}
    
    # Try direct JSON parsing
    try:
        return json.loads(output.strip())
    except json.JSONDecodeError:
        pass
    
    # Try to extract JSON from markdown code blocks
    json_patterns = [
        r'```json\s*([\s\S]*?)\s*```',  # ```json ... ```
        r'```\s*([\s\S]*?)\s*```',       # ``` ... ```
    ]
    
    for pattern in json_patterns:
        match = re.search(pattern, output)
        if match:
            json_str = match.group(1).strip()
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
    
    # Try to find JSON-like object in the text
    # Look for {...} pattern
    json_like_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    match = re.search(json_like_pattern, output)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    
    logger.error(f"Failed to parse JSON from output: {output}...")
    return {
        "diagnosis": "Unable to process incident response",
        "solution": "Please try again later",
        "questions": ["System encountered an issue processing the incident"],
        "resolved": "no"
    }


class IncidentState(BaseModel):
    incident_id: str = ""
    payload: dict = {}
    incident_description: str = ""
    incident_context: str = ""
    user_qa_pairs: List[dict] = []
    intent: str = ""
    final_output_json: dict = {}
    snow_status: str = ""
    ucic: str = ""
    current_comment: Optional[str] = None

async def send_update_to_servicenow_async(payload: Dict, question: str, resolution: str):
    tracer = get_tracer(__name__)
    with tracer.start_as_current_span("send_update_to_servicenow_async") as span:
        span.set_attribute("incident_id", payload.get("incidentId"))
        span.set_attribute("question_length", len(question) if question else 0)
        span.set_attribute("resolution_length", len(resolution) if resolution else 0)
        
        url = os.environ['SNOW_ENDPOINT']

        incident_id = payload.get("incidentId")
        headers = payload.get("headers", {})
        request_payload = {
            "callerId": payload.get("callerId"),
            "incidentType": payload.get("incidentType"),
            "businessService": payload.get("businessService"),
            "tier1": payload.get("tier1"),
            "tier2": payload.get("tier2"),
            "tier3": payload.get("tier3"),
            "impact": payload.get("impact"),
            "urgency": payload.get("urgency"),
            "shortDescription": payload.get("shortDescription"),
            "description": payload.get("description"),
            "contactType": payload.get("contactType"),
            "sourceIncidentNum": payload.get("sourceIncidentNum"),
            "sourceIncidentId": payload.get("sourceIncidentId"),
            "assignmentGroup": payload.get("assignmentGroup"),
            "incidentId": payload.get("incidentId"),
            "state": payload.get("state"),
            "causedByPatch": payload.get("causedByPatch"),
            "resolutionCode": payload.get("resolutionCode"),
            "solutionType": payload.get("solutionType"),
            "outageType": payload.get("outageType"),
            "additionalComments": question,
            "resolutionNotes": resolution,
            "cause": payload.get("cause"),
            "onHoldReason": payload.get("onHoldReason"),
            "correlationDisplay": payload.get("correlationDisplay"),
            "vendorGroup": payload.get("vendorGroup")
        }

        # Remove None values
        request_payload = {k: v for k, v in request_payload.items() if v is not None}

        headers.update({
            "Authorization": f"Basic {os.environ['SNOW_TOKEN']}",
        })

        interaction_counter = payload.get("interaction_counter")
        print(f"interaction_counter: {interaction_counter}")
        if interaction_counter <= 3:
            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    print(f"Request - url:{url} json:{request_payload} headers:{headers}")
                    response = await client.post(url, json=request_payload, headers=headers)

                    if response.status_code == 200:
                        print(f"Successfully updated incident {incident_id} in ServiceNow")
                        print(f"Response: {response.json()}")
                        return True,{"status_code":response.status_code,"response":response.json()}
                    else:
                        logger.warning(
                            f"Failed to update incident {incident_id} in ServiceNow. "
                            f"Status code: {response.status_code}, Response: {response.text}"
                        )
                        try:
                            return False,{"status_code":response.status_code,"response":response.json()}
                        except:
                            return False,{"status_code":response.status_code,"response_text":response.text}

            except Exception as e:
                logger.error(f"Error calling ServiceNow API for incident {incident_id}: {str(e)}")
                return False,{"status_code": 0 ,"response_text":"Exception : "+str(e)}


async def send_rejection_to_servicenow_async(payload, additonal_comment: str = 'BOT is unable to resolve, assign to an Engineer'):
    tracer = get_tracer(__name__)
    with tracer.start_as_current_span("send_rejection_to_servicenow_async") as span:
        span.set_attribute("incident_id", payload.get("incidentId"))
        payload.update({"state": "On Hold","cause": "Assign to an Engineer."})
        result = await send_update_to_servicenow_async(payload, additonal_comment, '')
        return result, 'rejected'        


async def send_question_to_servicenow_async(payload, question):
    tracer = get_tracer(__name__)
    with tracer.start_as_current_span("send_question_to_servicenow_async") as span:
        span.set_attribute("incident_id", payload.get("incidentId"))
        payload.update({"state": "On Hold", "onHoldReason": "User Action Required"})
        result = await send_update_to_servicenow_async(payload, question, '')
        return result, 'on_hold'         


async def send_resolution_to_servicenow_async(payload, resolution):
    tracer = get_tracer(__name__)
    with tracer.start_as_current_span("send_resolution_to_servicenow_async") as span:
        span.set_attribute("incident_id", payload.get("incidentId"))
        span.set_attribute("resolution_length", len(resolution) if resolution else 0)
        payload.update({
           #"cause": "Resolved by Bot.",
           #"state": "Resolved",
           # "resolutionCode": "Solved (Permanently)",
            #"solutionType": "Other",
            #"outageType": "No Outage"
            "state":"On Hold",
            "onHoldReason": "User Action Required"
        })
        result = await send_update_to_servicenow_async(payload, None, resolution)
        return result, 'on_hold'    


def payload_to_incident_description(payload):
    tracer = get_tracer(__name__)
    with tracer.start_as_current_span("payload_to_incident_description") as span:
        span.set_attribute("short_description_length", len(payload.get('shortDescription','')))
        span.set_attribute("description_length", len(payload.get('description','')))
        span.set_attribute("individualUCIC", payload.get('individualUCIC','i'))
        
        short_description = payload.get('shortDescription','')
        description = payload.get('description','')
        individualUCIC = payload.get('individualUCIC','i')
        # individualUCIC = individualUCIC[:-1]
        result = f"Short Description: {short_description}\nDescription: {description}"
        print(f"Generated incident description for UCIC {individualUCIC}")
        return result, individualUCIC



# @persist(key="incident_id")
class IncidentManagementFlow(Flow[IncidentState]):

    @start()
    async def initialize_and_classify(self):
        tracer = get_tracer(__name__)
        with tracer.start_as_current_span("initialize_and_classify") as span:
            span.set_attribute("incident_id", self.state.incident_id)

            self.state.incident_description, self.state.ucic = payload_to_incident_description(self.state.payload)

            if '__agent_data' not in self.state.payload:
                self.state.payload['__agent_data'] = {
                    'snow_logs': [], 'qa_pairs': [], 'comments': []
                }

            snow_logs = self.state.payload.get('__agent_data', {}).get('snow_logs', [])
            if snow_logs and snow_logs[-1]['type'] == 'question' and self.state.current_comment:
                 self.state.payload['__agent_data']['qa_pairs'].append({
                     "question": snow_logs[-1]["question"],
                     "answer": self.state.current_comment
                 })

            self.state.user_qa_pairs = self.state.payload['__agent_data'].get('qa_pairs', [])

            comment = self.state.current_comment if self.state.current_comment else "NA"

            self.state.intent = await run_crew_with_retry_async(
                lambda: run_intent_classifier_crew_async(
                    self.state.incident_description, 
                    self.state.user_qa_pairs,
                    comment
                )
            )
            print(f"Initialized and classified incident {self.state.incident_id} with intent: {self.state.intent}")
            return self.state.intent

    @router(initialize_and_classify)
    async def start_process(self):
        tracer = get_tracer(__name__)
        with tracer.start_as_current_span("logic_router") as span:
            span.set_attribute("intent", self.state.intent)
            span.set_attribute("sop_exists", self.state.payload['__agent_data'].get('sop') is not None)
            span.set_attribute("sop_value", self.state.payload['__agent_data'].get('sop'))

            counter = self.state.payload.get("interaction_counter", 0)
            self.state.payload["interaction_counter"] = counter + 1
            if counter >= 3:
                print(f"Interaction limit exceeded for incident {self.state.incident_id}")
                return "limit_exceeded"

            if self.state.intent == "closure": 
                print(f"Closure intent for incident {self.state.incident_id}")
                return "handle_closure"
            if self.state.intent == "rebuttal": 
                print(f"Rebuttal intent for incident {self.state.incident_id}")
                return "handle_rebuttal"
            print(f"Fresh incident {self.state.incident_id}, gather context")
            return "gather_context"


    @listen('gather_context')
    async def semantic_search(self):
        tracer = get_tracer(__name__)
        with tracer.start_as_current_span("gather_context") as span:
            span.set_attribute("incident_id", self.state.incident_id)
            span.set_attribute("incident_description_length", len(self.state.incident_description))

            incident_description, ucic = payload_to_incident_description(self.state.payload)
            desc = f"{incident_description}\nUCIC: {ucic}"
            app = self.state.payload.get("tier1", "CBS")
            incident_context = await run_crew_with_retry_async(
                lambda: run_incident_context_crew_async(desc,application=app)
            )
            self.state.incident_context = incident_context if incident_context else "No context found"
                
            return 'run_resolver'

    
    @listen(semantic_search)
    async def run_resolver_crew(self):
        tracer = get_tracer(__name__)
        with tracer.start_as_current_span("run_resolver") as span:
            span.set_attribute("incident_id", self.state.incident_id)
            span.set_attribute("qa_pairs_count", len(self.state.user_qa_pairs))

            # ── read app from payload ──
            app = self.state.payload.get("tier1", "cbs").lower().strip()
            span.set_attribute("app", app)
            logger.info(f"Resolver | incident={self.state.incident_id} app={app}")

            raw_resolution = await run_crew_with_retry_async(
                lambda: run_backend_resolver_crew_async(
                    self.state.incident_description,
                    self.state.incident_context,
                    self.state.user_qa_pairs,
                    self.state.ucic,
                    self.state.current_comment,
                    app=app    
                )
            )

            self.state.final_output_json = extract_json_from_output(raw_resolution)
            print(f"Resolver task completed for incident {self.state.incident_id}")
            return "update_servicenow"

    @listen(run_resolver_crew)
    async def update_servicenow(self):
        tracer = get_tracer(__name__)
        with tracer.start_as_current_span("update_servicenow") as span:
            span.set_attribute("incident_id", self.state.incident_id)
            span.set_attribute("resolution_result", self.state.final_output_json.get("resolved", "unknown"))

            res = self.state.final_output_json
            incident_status = 'in_progress'

            if res.get("resolved") == 'yes':
                msg = f"Diagnosis:\n{res['diagnosis']}\n\nSolution:\n{res['solution']}"
                (status, info), incident_status = await send_resolution_to_servicenow_async(self.state.payload, msg)
                self.state.payload['__agent_data']['snow_logs'].append({
                    "type": "resolution", "resolution": msg, "status": status, "response": info
                })
                print(f"Resolution sent for incident {self.state.incident_id}")

            elif res.get("questions"):
                msg = "\n".join(res["questions"])
                (status, info), incident_status = await send_question_to_servicenow_async(self.state.payload, msg)
                self.state.payload['__agent_data']['snow_logs'].append({
                    "type": "question", "question": msg, "status": status, "response": info
                })
                print(f"Question sent for incident {self.state.incident_id}")

            else:
                msg = f"Diagnosis:\n{res.get('diagnosis')}\n\nSolution:\n{res['solution']}"
                (status, info), incident_status = await send_question_to_servicenow_async(self.state.payload, msg)
                self.state.payload['__agent_data']['snow_logs'].append({
                    "type": "question", "question": msg, "status": status, "response": info
                })
                print(f"Question sent for incident {self.state.incident_id}")

            state = self.state.model_dump()
            payload_copy = state['payload']

            await upsert_incident_payload_async(
                self.state.incident_id,
                json.dumps(payload_copy),
                incident_status
            )
            print(f"DB updated for incident {self.state.incident_id} status={incident_status}")

    @listen('handle_rebuttal')
    async def run_rebuttal_crew(self):
        tracer = get_tracer(__name__)
        with tracer.start_as_current_span("handle_rebuttal") as span:
            span.set_attribute("incident_id", self.state.incident_id)
            msg = 'BOT is unable to resolve, assign to an Engineer'
            (status, info), incident_status = await send_rejection_to_servicenow_async(self.state.payload, msg)
            self.state.payload['__agent_data']['snow_logs'].append({
                "type": "rejection", "status": status, "response": info
            })
            state = self.state.model_dump()
            payload_copy = state['payload']
            await upsert_incident_payload_async(
                self.state.incident_id,
                json.dumps(payload_copy),
                incident_status
            )
            print(f"Rebuttal handled for incident {self.state.incident_id} status={incident_status}")

    @listen('limit_exceeded')
    async def handle_limit_exceeded(self):
        tracer = get_tracer(__name__)
        with tracer.start_as_current_span("handle_limit_exceeded") as span:
            span.set_attribute("incident_id", self.state.incident_id)
            (status, info), incident_status = await send_rejection_to_servicenow_async(self.state.payload)
            self.state.payload['__agent_data']['snow_logs'].append({
                "type": "rejection", "status": status, "response": info
            })
            state = self.state.model_dump()
            payload_copy = state['payload']
            await upsert_incident_payload_async(
                self.state.incident_id,
                json.dumps(payload_copy),
                incident_status
            )
            print(f"Limit exceeded rejected incident {self.state.incident_id} status={incident_status}")
            return

    @listen('closure')
    async def handle_incident_closure(self):
        tracer = get_tracer(__name__)
        with tracer.start_as_current_span("handle_closure") as span:
            span.set_attribute("incident_id", self.state.incident_id)
            print("No action for bot to take")
            return

    @listen("reject_incident")
    async def handle_rejection(self):
        tracer = get_tracer(__name__)
        with tracer.start_as_current_span("handle_rejection") as span:
            span.set_attribute("incident_id", self.state.incident_id)
            (status, info), incident_status = await send_rejection_to_servicenow_async(self.state.payload)
            self.state.payload['__agent_data']['snow_logs'].append({
                "type": "rejection", "status": status, "response": info
            })
            state = self.state.model_dump()
            payload_copy = state['payload']
            await upsert_incident_payload_async(
                self.state.incident_id,
                json.dumps(payload_copy),
                incident_status
            )
            print(f"Rejected incident {self.state.incident_id} status={incident_status}")
            return






            now i want to imple,ment the iamge processing as well :

            new payload would be :



            curl --request POST \
  --url 'https://ent-nonprod.api.idk.com/api-ext/v1/incident-agent/create-incident?=' \
  --header 'Authorization: Bearer ' \
  --header 'Content-Type: application/json' \
  --header 'source: KongTest' \
  --data '{
  "message": "Incident has been created successfully.",
  "incidentNumber": "INC00701",
  "incidentId": "b50370443b90f290b6986f34c3e45a4d",
  "incidentType": "Application",
  "businessService": "SFDC Asset Org 2",
  "tier1": "SFDC Asset Org 2",
  "tier2": "Infra",
  "tier3": "Self Monitoring Patrol Agent",
  "impact": "Low",
  "urgency": "Medium",
  "priority": "Low",
  "shortDescription": "27 OCT 2023 Short Description Test",
  "description": "27 OCT 2023 Description Test",
  "contactType": "Event",
  "state": "New",
  "assignmentGroup": "",
  "sourceIncidentNum": "T000123",
  "sourceIncidentId": "26SEP2023-0120-111",
  "businessImpact": null,
  "cause": null,
  "businessCorrectiveAction": null,
  "techCorrectiveAction": null,
  "dataSource": null,
  "descriptionOfOutage": null,
  "emailID": null,
  "entityUCIC": null,
  "hashValues": null,
  "ipDetails": null,
  "ldwNotifyInformation": null,
  "loanAccountNumber": null,
  "loginId": null,
  "mobileNumber": null,
  "businessPreventiveAction": null,
  "techPreventiveAction": null,
  "resoultionTeam": null,
  "rootCause": null,
  "systemName": null,
  "urlOrDomain": null,
  "userDetail": null,
  "individualUCIC": "123",
  "sourceIncCreateddttime": "24-Feb-2023 09:10:04",
  "incidentURL": "https://.service-now.com/isupport?sys_idb50370443b90f290b6986f34c3e45a4d&viewsp&idticket&tableincident%22,
  "userLocation": "",
  "files": [
    {
      "fileId": "FILE001",
      "fileName": "pan_card.pdf",
      "fileType": "application/pdf",
      "fileSize": 245678,
      "contentEncoding": "base64",
      "fileContent": "4AAQSkZJRgABAQAAAQ"
    },
    {
      "fileId": "FILE002",
      "fileName": "aadhaar_front.jpg",
      "fileType": "image/jpeg",
      "fileSize": 125678,
      "contentEncoding": "base64",
      "fileContent": "/9j/4AAQSkZJRgABAQAAAQ..."
    }
  ]
}'



this is one way from different repo  for image processing can you use this :
 if doc_item.label == "picture" and doc_item.image:
                uri = str(doc_item.image.uri)
                img_id = str(uuid.uuid4())
                img_refs[img_id] = uri
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this image"},
                            {"type": "image_url", "image_url": {"url": uri}},
                        ],
                    }
                ]
                try:
                    image_description = (
                        await llm.chat_completion(http_client=http_client, messages=messages, config_type='vision')
                    )["choices"][0]["message"].get("content")
                except Exception as e:
                    print(e)
                    image_description = "IMAGE UNREADABLE NO DESCRIPTION"
