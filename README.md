from glob import glob
import json
import asyncio
from utils.logs import execute_elk_query
from utils.oracle_connection import execute_oracle_query_async
from crewai.tools import BaseTool
import os

from utils.http_calls import  http_client_post_async, http_client_get_async
from utils.oracle_connection import execute_oracle_query_async

os.environ["OTEL_SDK_DISABLED"] = "true"

from glob import glob

import traceback

import logging
import threading

logger = logging.getLogger(__name__)


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
        response= await http_client_post_async(url=oauth_url, headers=headers, data=None, form_data=form_data)
        json_resp = response.json
        return json_resp['access_token']

    async def get_valid_token(self):
        import time
        
        with self._token_lock:
            if self.token is None or self.token_expiry is None or time.time() >= self.token_expiry:
                self.token = await self._get_token()
                # Token expires in 30 minutes, refresh 1 minute before expiry
                self.token_expiry = time.time() + (30 * 60) - 60
            return self.token

giftwrap_token_manager = GiftWrappingTokenManager()

async def make_giftwrap_api_call_async(endpoint: str, method: str, parameters: dict, headers: dict = None, query_params: dict = None):
    import uuid
    
    base_url = os.getenv('GIFTWRAP_API_BASE_URL')
    
    token = await giftwrap_token_manager.get_valid_token()
    
    # Build base headers
    request_headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json',
        'User-Agent': 'incident-agent/1.0',
        'X-B3-Sampled': '1',
        'X-B3-SpanId': f'{uuid.uuid4()}',
        'X-B3-TraceId': f'{uuid.uuid4()}'
    }

    # Add custom headers from the tool definition
    if headers:
        request_headers.update(headers)

    # Build URL with query parameters
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
    """
    Returns tools based on app type.
    To add a new app: add elif block + create tools/<app>/ folder.

    CBS     → Oracle SQL + ELK + CBS API tools (existing)
    Optimus → Jaeger log tool
    """
    app_key = app.lower().strip()
    logger.info(f"Loading tools for app: {app_key}")

    if app_key == "cbs":
        return _load_cbs_tools()
    elif app_key == "optimus":
        return _load_optimus_tools()
    else:
        logger.warning(f"Unknown app '{app}' — defaulting to CBS tools")
        return _load_cbs_tools()


# ─────────────────────────────────────────
# CBS TOOLS — existing logic, zero changes
# ─────────────────────────────────────────

def _load_cbs_tools():
    """Loads all CBS tools from JSON files. Zero changes from original."""

    json_tool_list = glob(f'./tools/cbs/api_tools.json')
    json_tool_list += glob(f'./tools/cbs/elk_tools.json')
    json_tool_list += glob(f'./tools/cbs/sql_tools.json')

    tool_parameter_list = []

    for path in json_tool_list:
        with open(path, 'r') as f:
            d = json.loads(f.read())
            tool_parameter_list += d

    tool_list = []

    def make_tool(tool):

        tool_name = tool['name']
        tool_type = tool.get('type', 'UNKNOWN')
        tool_query = tool.get('query', '')

        if not tool.get('parameters'):
            tool['parameters'] = [{'name': 'key', 'description': 'Input value'}]

        param_name = tool['parameters'][0]['name']
        param_description = tool['parameters'][0]['description']

        if tool_type == 'SQL':
            class SQLTool(BaseTool):
                name: str = tool_name
                description: str = (
                    tool['purpose']
                    + '\nEnter the '
                    + param_name
                    + ' in key, it should contain '
                    + param_description
                )

                def regularize_id(self, incoming_id: str) -> str:
                    raw = str(incoming_id).strip()
                    raw = raw.removeprefix('003')
                    candidate = raw.lstrip('0')
                    return candidate

                def _run(self, key: str):
                    return asyncio.run(self._arun(key))

                async def _arun(self, key: str):
                    clean_value = self.regularize_id(str(key).strip())
                    if self.name in ["verify_sms_required_flag", "get_account_level_mobile", "get_email_preference_flag", "check_posting_restriction_mask", "find_ucic_from_account"]:
                        processed_value = f"003{clean_value.zfill(16)}"
                    elif self.name in ["get_account_level_email", "get_ucic_email", "check_transaction_freeze", "check_npa_debit_freeze", "check_transaction_freeze"]:
                        processed_value = f"{clean_value.zfill(16)}"
                    elif self.name in ["find_account_from_ucic"]:
                        clean_value = clean_value[:-1]
                        processed_value = f"{clean_value.zfill(16)}"
                    else:
                        processed_value = clean_value

                    final_query = tool_query.replace(f":{param_name}", processed_value)

                    logger.info(f"[{self.name}] Executing async: {final_query}")
                    logger.info(f"[SQL] Clean Key: {clean_value}")
                    logger.info(f"[SQL] Processed Key: {processed_value}")

                    try:
                        return await execute_oracle_query_async(final_query)
                    except Exception as e:
                        return f"Query Error: {str(e)}"

            return SQLTool()

        elif tool['type'] == 'ELK':
            class ELKTool(BaseTool):
                name: str = tool['name']
                description: str = (
                    tool['purpose']
                    + '\nEnter the '
                    + tool['parameters'][0]['name']
                    + ' in key, it should contain '
                    + tool['parameters'][0]['description']
                )

                def _run(self, key: str):
                    return asyncio.run(self._arun(key))

                async def _arun(self, key: str):
                    try:
                        param_name = tool['parameters'][0]['name']
                        query_json = tool['query']
                        result = await execute_elk_query(
                            query_json,
                            parameter_name=param_name,
                            parameter_value=key
                        )
                        logger.info(f"[ELK] Query executed successfully")
                        return result
                    except Exception as e:
                        return f"ELK query failed: {str(e)}"

            return ELKTool()

        elif tool_type == 'API':
            endpoint = tool.get('endpoint', '')
            method = tool.get('method', 'POST')
            parameters = tool.get('parameters', [])
            headers_def = tool.get('headers', [])
            query_params_def = tool.get('query_params', [])

            param_descriptions = []
            required_params = []
            param_types = {}

            for param in parameters:
                param_name = param['name']
                param_type = param.get('type', 'string')
                required_str = " (required)" if param.get('required', False) else " (optional)"
                param_descriptions.append(f"{param_name}: {param['description']}{required_str}")

                if param.get('required', False):
                    required_params.append(param_name)

                if param_type == 'integer':
                    param_types[param_name] = (int, None)
                elif param_type == 'array':
                    param_types[param_name] = (list, None)
                elif param_type == 'boolean':
                    param_types[param_name] = (bool, None)
                else:
                    param_types[param_name] = (str, None)

            param_desc_str = ', '.join(param_descriptions)

            from pydantic import BaseModel, Field
            from typing import Optional, Union

            field_definitions = {}
            field_defaults = {}
            for param in parameters:
                param_name = param['name']
                param_type = param.get('type', 'string')
                is_required = param.get('required', False)

                if param_type == 'integer':
                    py_type = int if is_required else Optional[int]
                elif param_type == 'array':
                    py_type = list if is_required else Optional[list]
                elif param_type == 'boolean':
                    py_type = bool if is_required else Optional[bool]
                else:
                    py_type = str if is_required else Optional[str]

                field_definitions[param_name] = py_type
                if not is_required:
                    field_defaults[param_name] = None

            APIToolSchema = type('APIToolSchema', (BaseModel,), {
                '__annotations__': field_definitions,
                **field_defaults,
                'model_config': {'extra': 'forbid', 'validate_default': True}
            })

            class APITool(BaseTool):
                name: str = tool_name
                description: str = (
                    tool['purpose']
                    + '\nParameters: '
                    + param_desc_str
                )
                args_schema: type = APIToolSchema

                def _run(self, **kwargs):
                    return asyncio.run(self._arun(**kwargs))

                async def _arun(self, **kwargs):
                    logger.info(f"[API Async] Tool: {self.name}")
                    logger.info(f"[API Async] Parameters: {kwargs}")

                    try:
                        missing_required = []
                        for param in parameters:
                            if param.get('required', False):
                                param_name = param['name']
                                if param_name not in kwargs or kwargs[param_name] is None:
                                    missing_required.append(param_name)

                        if missing_required:
                            return f"Error: Missing required parameters: {', '.join(missing_required)}"

                        api_params = {}
                        path_params = {}

                        for param in parameters:
                            param_name = param['name']
                            if param_name in kwargs and kwargs[param_name] is not None:
                                if f"{{{param_name}}}" in endpoint:
                                    path_params[param_name] = kwargs[param_name]
                                else:
                                    api_params[param_name] = kwargs[param_name]
                            else:
                                if parameters and param_name not in api_params:
                                    first_param = parameters[0]['name']
                                    if first_param in kwargs and kwargs[first_param] is not None:
                                        if f"{{{param_name}}}" in endpoint:
                                            path_params[param_name] = kwargs[first_param]
                                        else:
                                            api_params[param_name] = kwargs[first_param]

                        final_endpoint = endpoint
                        for param_name, param_value in path_params.items():
                            final_endpoint = final_endpoint.replace(f"{{{param_name}}}", str(param_value))

                        request_headers = {}
                        for header_def in headers_def:
                            header_name = header_def['name']
                            if header_name.startswith('!'):
                                continue
                            if header_name in kwargs and kwargs[header_name] is not None:
                                request_headers[header_name] = kwargs[header_name]
                            elif header_def.get('enum'):
                                request_headers[header_name] = header_def['enum'][0]
                            elif header_def.get('required', False) and header_def.get('default'):
                                request_headers[header_name] = header_def['default']

                        request_query_params = {}
                        for qp_def in query_params_def:
                            qp_name = qp_def['name']
                            if qp_name in kwargs and kwargs[qp_name] is not None:
                                request_query_params[qp_name] = kwargs[qp_name]

                        result = await make_giftwrap_api_call_async(
                            endpoint=final_endpoint,
                            method=method,
                            parameters=api_params,
                            headers=request_headers if request_headers else None,
                            query_params=request_query_params if request_query_params else None
                        )
                        logger.info(f"[API Async] Response: {result}")
                        return result
                    except Exception as e:
                        traceback.print_exc()
                        error_msg = f"API call failed (async): {str(e)}"
                        logger.error(f"[API Async] Error: {error_msg}")
                        return error_msg

            return APITool()

        else:
            class TempTool(BaseTool):
                name: str = tool_name
                description: str = (
                    tool['purpose']
                    + '\nEnter the '
                    + tool['parameters'][0]['name']
                    + ' in key, it should contain '
                    + tool['parameters'][0]['description']
                )

                def _run(self, *args, **kwargs):
                    return "NOT AVAILABLE"

                async def _arun(self, key: str):
                    return "NOT AVAILABLE"

            return TempTool()

    for tool in tool_parameter_list:
        logger.info(f"Adding CBS tool: {tool['name']}")
        tool_list.append(make_tool(tool))

    tool_list.append(NormalizeAccountTool())
    return tool_list


# ─────────────────────────────────────────
# OPTIMUS TOOLS — Jaeger log fetching
# ─────────────────────────────────────────

def _load_optimus_tools():
    """Loads Optimus tools — Jaeger log tool."""
    from tools.optimus.jaeger_tool import get_optimus_jaeger_tool

    tool_list = []
    tool_list.append(get_optimus_jaeger_tool())

    logger.info(f"Optimus tools loaded: {len(tool_list)} tools")
    return tool_list


# ─────────────────────────────────────────
# SHARED TOOLS
# ─────────────────────────────────────────

class NormalizeAccountTool(BaseTool):
    name: str = "normalize_account_number"
    description: str = (
        "Normalizes account numbers to a standard format for comparison. "
        "Input: Account number in any format (e.g., '200020557' or '0030000000200020557'). "
        "Output: Normalized account number (e.g., '200020557'). "
        "Use this when you need to compare account numbers from different sources to find matches."
    )

    def _run(self, account_number: str):
        return asyncio.run(self._arun(account_number))

    async def _arun(self, account_number: str):
        raw = str(account_number).strip()
        raw = raw.removeprefix('003')
        return raw.lstrip('0')







        this is new 



        from glob import glob
import json
import asyncio
from utils.logs import execute_elk_query
from utils.oracle_connection import execute_oracle_query_async
from crewai.tools import BaseTool
import os

from utils.http_calls import http_client_post_async, http_client_get_async
from utils.oracle_connection import execute_oracle_query_async

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
}
_JAEGER_EXCLUDED_SERVICES = {
    'RISK-CONTROL-REQUEST-METADATA-PROCESSOR',
    'RISK-CONTROL-USER-METADATA-PROCESSOR',
}
_JAEGER_MAX_ERROR_TRACES = 10
_JAEGER_MAX_LINES_PER_TRACE = 50


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
        return [], None

    result = []
    session_id = None

    def process_span(span_id, depth=0):
        nonlocal session_id
        span = id_to_span[span_id]
        indent = "    " * depth
        if depth == 0:
            result.append("Time: " + _jaeger_us_to_ist(span["startTime"]))
        op_name = span.get("operationName", "")
        service_name = pid_to_service.get(span.get("processID"), "")
        if op_name not in _JAEGER_EXCLUDED_OPS and service_name not in _JAEGER_EXCLUDED_SERVICES:
            result.append(f"{indent}{span_id} - {service_name} - {op_name}")
        for tag in span.get("tags", []):
            if tag.get("key") == "http.status_code":
                result.append(f"{indent}HTTP Status: {tag.get('value')}")
            elif tag.get("key") == "session_tracing_id" and not session_id:
                session_id = tag.get("value")
        for child_id in children_map.get(span_id, []):
            process_span(child_id, depth + 1)

    for root_id in roots:
        process_span(root_id)

    return result, session_id


async def _jaeger_fetch(service: str, tag_name: str, tag_value: str):
    now_us = int(_time.time() * 1_000_000)
    start_us = now_us - 24 * 3600 * 1_000_000   # always last 24 hours
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
                return {"error": f"Jaeger API returned {response.status_code}", "error_traces": [], "trace_links": [], "sessions": [], "success_count": 0}
            data = response.json().get("data", [])
    except Exception as e:
        logger.error(f"[JAEGER] Request failed: {e}")
        return {"error": str(e), "error_traces": [], "trace_links": [], "sessions": [], "success_count": 0}

    if not data:
        return {"error_traces": [], "trace_links": [], "sessions": [], "success_count": 0,
                "message": f"No traces found for {tag_name}={tag_value} in {service}"}

    jaeger_ui_base = JAEGER_API_BASE.replace("/api", "")
    error_traces = []
    trace_links = []
    sessions = set()
    success_count = 0

    for trace in data:
        trace_id = trace.get("traceID")
        lines, session_id = _jaeger_process_trace(trace)
        if session_id:
            sessions.add(session_id)
        is_success = lines and any("HTTP Status: 2" in l for l in lines)
        if is_success:
            success_count += 1
            continue
        error_codes = []
        for span in trace.get("spans", []):
            for tag in span.get("tags", []):
                if tag.get("key") == "http.status_code":
                    code = str(tag.get("value", ""))
                    if code.startswith(("4", "5")):
                        error_codes.append(code)
        if error_codes:
            status_summary = ", ".join(sorted(set(error_codes)))
            trace_links.append(f"{jaeger_ui_base}/trace/{trace_id} — Status: {status_summary}")
            truncated = lines[:_JAEGER_MAX_LINES_PER_TRACE]
            error_traces.append({
                "trace_id": trace_id,
                "status_codes": list(set(error_codes)),
                "trace_url": f"{jaeger_ui_base}/trace/{trace_id}",
                "details": "\n".join(truncated)
            })
            if len(error_traces) >= _JAEGER_MAX_ERROR_TRACES:
                break

    logger.info(f"[JAEGER] Done: {len(error_traces)} errors, {success_count} successes")
    return {
        "service": service, "tag_name": tag_name, "tag_value": tag_value,
        "error_traces": error_traces, "trace_links": trace_links,
        "sessions": list(sessions), "success_count": success_count,
        "total_traces_scanned": len(data)
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
        response= await http_client_post_async(url=oauth_url, headers=headers, data=None, form_data=form_data)
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
    
    # Build base headers
    request_headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json',
        'User-Agent': 'incident-agent/1.0',
        'X-B3-Sampled': '1',
        'X-B3-SpanId': f'{uuid.uuid4()}',
        'X-B3-TraceId': f'{uuid.uuid4()}'
    }

    # Add custom headers from the tool definition
    if headers:
        request_headers.update(headers)

    # Build URL with query parameters
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
    """
    Returns tools based on app type.
    To add a new app: add elif block + create tools/<app>/ folder.

    CBS     → Oracle SQL + ELK + CBS API tools (existing)
    Optimus → Jaeger log tool
    """
    app_key = app.lower().strip()
    logger.info(f"Loading tools for app: {app_key}")

    if app_key == "cbs":
        return _load_tools_from_folder("./tools/cbs", include_normalize=True)
    elif app_key == "optimus":
        return _load_tools_from_folder("./tools/optimus", include_normalize=False)
    else:
        logger.warning(f"Unknown app '{app}' — defaulting to CBS tools")
        return _load_tools_from_folder("./tools/cbs", include_normalize=True)


# ─────────────────────────────────────────
# SHARED TOOL LOADER — reads all JSON in folder, builds tools by type
# ─────────────────────────────────────────

def _load_tools_from_folder(folder: str, include_normalize: bool = False):
    """Loads all *.json tool files from a folder and builds tools based on type."""

    json_tool_list = glob(f'{folder}/*.json')

    tool_parameter_list = []
    for path in json_tool_list:
        with open(path, 'r') as f:
            d = json.loads(f.read())
            tool_parameter_list += d

    tool_list = []
    for tool in tool_parameter_list:
        logger.info(f"Adding tool: {tool['name']} (type={tool.get('type')})")
        tool_list.append(_make_tool(tool))

    if include_normalize:
        tool_list.append(NormalizeAccountTool())

    return tool_list


def _make_tool(tool):
    tool_name = tool['name']
    tool_type = tool.get('type', 'UNKNOWN')
    tool_query = tool.get('query', '')

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
                if self.name in ["verify_sms_required_flag", "get_account_level_mobile", "get_email_preference_flag", "check_posting_restriction_mask", "find_ucic_from_account"]:
                    processed_value = f"003{clean_value.zfill(16)}"
                elif self.name in ["get_account_level_email", "get_ucic_email", "check_transaction_freeze", "check_npa_debit_freeze"]:
                    processed_value = f"{clean_value.zfill(16)}"
                elif self.name in ["find_account_from_ucic"]:
                    clean_value = clean_value[:-1]
                    processed_value = f"{clean_value.zfill(16)}"
                else:
                    processed_value = clean_value

                final_query = tool_query.replace(f":{param_name}", processed_value)
                logger.info(f"[{self.name}] Executing: {final_query}")
                try:
                    return await execute_oracle_query_async(final_query)
                except Exception as e:
                    return f"Query Error: {str(e)}"

        return SQLTool()

    elif tool['type'] == 'ELK':
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

    # ── JAEGER ──  (NEW — Optimus log traces)
    elif tool['type'] == 'JAEGER':
        from pydantic import BaseModel
        from typing import Optional

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
                + "\ntag_value: the actual identifier value")
                
            args_schema: type = JaegerToolSchema

            def _run(self, tag_name: str, tag_value: str):
                return asyncio.run(self._arun(tag_name, tag_value))

            async def _arun(self, tag_name: str, tag_value: str):
                try:
                    return await _jaeger_fetch(
                        service=jaeger_service,
                        tag_name=tag_name,
                        tag_value=tag_value,
                       
                    )
                except Exception as e:
                    logger.error(f"[JaegerTool {self.name}] failed: {e}")
                    return {"error": str(e), "error_traces": [], "trace_links": [], "sessions": []}

        return JaegerTool()


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




        this is latest :


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

JAEGER_API_BASE = os.getenv("JAEGER_API_BASE")
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

# ── Chunk/summarize config ──
_JAEGER_CHAR_BUDGET = 40000     # if combined output exceeds this, summarize
_JAEGER_CHUNK_SIZE = 40000      # size of each chunk sent to summarizer
_JAEGER_MAX_CHUNKS = 8          # hard cap so a giant trace can't spin forever


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
    """Keep a trace ONLY if it has a real 4xx or 5xx HTTP status."""
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
                    val = fields.get(payload_key)
                    if val:
                        s = str(val)
                        # skip auth/token noise
                        if 'token' in s.lower() or s.startswith('--') or 'introspect' in s.lower():
                            continue
                        result.append(f"{indent}  {payload_key}: {s[:_JAEGER_MAX_PAYLOAD_CHARS]}")

        for child_id in children_map.get(span_id, []):
            process_span(child_id, depth + 1)

    for root_id in roots:
        process_span(root_id)

    return result


# ─────────────────────────────────────────
# CHUNK + SUMMARIZE (Jaeger only) — parallel map-reduce
# ─────────────────────────────────────────

async def _summarize_one_chunk(chunk: str, idx: int, total: int) -> str:
    """Summarize a single chunk. Isolated — if it fails, returns a note, never raises."""
    prompt = (
        "You are analyzing Jaeger trace logs to find the ACTUAL error a customer hit.\n"
        "IMPORTANT: HTTP status 200 does NOT always mean success here — the real error is "
        "often inside the response BODY (error_code, error_message, status:fail, decline "
        "reason, downstream failure). Read response payloads carefully, not just status codes.\n"
        "If any value looks base64/encoded, decode it and report the meaning.\n\n"
        f"Log chunk ({idx}/{total}):\n{chunk}\n\n"
        "Extract ONLY: error codes, error messages, failed endpoints, decline/failure reasons, "
        "downstream service errors. Ignore tokens, OAuth spans, and clean successful noise. "
        "Be concise."
        "Never remove the error code and reponse from the summary, always pass the error code and response  as it is and donot summarise those" 
    )
    try:
        from utils.llm import llm_config, OPENAI_MODEL_NAME
        payload = {
            "model": OPENAI_MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 1200
        }
        headers = {
            "Authorization": f"Bearer {llm_config.token}",
            "Content-Type": "application/json"
        }
        async with httpx.AsyncClient(timeout=120.0, verify='./IDFCBANKCA.pem') as client:
            r = await client.post(f"{llm_config.url}/chat/completions",
                                  headers=headers, json=payload)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.error(f"[JAEGER] chunk {idx} summarize failed: {e}")
        return f"[chunk {idx} could not be summarized]"


async def _merge_summaries(summaries: list) -> str:
    """Merge per-chunk summaries into one final diagnosis. One LLM call."""
    joined = "\n\n".join(summaries)
    prompt = (
        "Below are error summaries extracted from separate chunks of a customer's Jaeger "
        "trace logs. Merge them into one clear picture of what actually failed.\n"
        "Report the concrete error(s): error code, error message, which endpoint/service "
        "Never remove the error code and reponse from the summary, always pass the error code and response "
        "failed, and the likely root cause. Ignore anything that looks successful or is noise.\n\n"
        f"{joined}\n\nFinal consolidated error summary:"
    )
    try:
        from utils.llm import llm_config, OPENAI_MODEL_NAME
        payload = {
            "model": OPENAI_MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 1500
        }
        headers = {
            "Authorization": f"Bearer {llm_config.token}",
            "Content-Type": "application/json"
        }
        async with httpx.AsyncClient(timeout=120.0, verify='./IDFCBANKCA.pem') as client:
            r = await client.post(f"{llm_config.url}/chat/completions",
                                  headers=headers, json=payload)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.error(f"[JAEGER] merge failed: {e}")
        return joined  # fall back to raw joined summaries


async def _chunk_and_summarize(full_text: str) -> str:
    chunks = [full_text[i:i + _JAEGER_CHUNK_SIZE]
              for i in range(0, len(full_text), _JAEGER_CHUNK_SIZE)]

    if len(chunks) > _JAEGER_MAX_CHUNKS:
        chunks = chunks[:_JAEGER_MAX_CHUNKS]

    logger.info(f"[JAEGER] Summarizing {len(chunks)} chunks in parallel (total {len(full_text)} chars)")

    # summarize all chunks in parallel — one failure doesn't kill the rest
    summaries = await asyncio.gather(
        *[_summarize_one_chunk(c, i + 1, len(chunks)) for i, c in enumerate(chunks)]
    )

    if len(summaries) == 1:
        return summaries[0]

    return await _merge_summaries(summaries)


async def _jaeger_fetch(service: str, tag_name: str, tag_value: str):
    now_us = int(_time.time() * 1_000_000)
    start_us = now_us - 72 * 3600 * 1_000_000
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

    # keep ONLY 4xx/5xx traces
    failed_traces_raw = [t for t in data if _jaeger_trace_has_error(t)]

    failed_traces_text = []
    for trace in failed_traces_raw:
        lines = _jaeger_process_trace(trace)
        if lines:
            failed_traces_text.append("\n".join(lines))

    logger.info(f"[JAEGER] Done: scanned={len(data)} failed={len(failed_traces_text)}")

    combined = "\n\n".join(failed_traces_text)

    # ── chunk + summarize ONLY if too big ──
    if len(combined) > _JAEGER_CHAR_BUDGET:
        logger.info(f"[JAEGER] Output too large ({len(combined)} chars) — chunking & summarizing")
        combined = await _chunk_and_summarize(combined)
        failed_traces_text = [combined]

    return {
        "service": service,
        "tag_name": tag_name,
        "tag_value": tag_value,
        "total_traces_scanned": len(data),
        "total_failed": len(failed_traces_raw),
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
                + "\nOnly failed/error traces are returned. If no failures are found, the issue "
                  "may not be reproducible in logs."
                  +"Always return all error code and there message in reponse"
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





        
