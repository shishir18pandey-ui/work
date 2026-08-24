from glob import glob
import json
import asyncio
from new_flow.utils.logs import execute_elk_query
from new_flow.utils.oracle_connection import execute_oracle_query_async
from crewai.tools import BaseTool
import os

from new_flow.utils.http_calls import http_client_post_async, http_client_get_async

os.environ["OTEL_SDK_DISABLED"] = "true"

from glob import glob

import traceback
import logging
import threading
from app_config import get_jager_auth_token
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
_JAEGER_MAX_PAYLOAD_CHARS = 450


def _jaeger_auth_headers(app:str):
    token=get_jager_auth_token(app)
    return {"Authorization":f"Basic {token}"} if token else {}


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

async def _check_jaeger_connection():
    try:
        headers = _jaeger_auth_headers()

        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(
                f"{JAEGER_API_BASE}/services",
                headers=headers,
            )

        logger.info(
            "[JAEGER-CONNECTION] status=%s url=%s",
            response.status_code,
            f"{JAEGER_API_BASE}/services",
        )

        if response.status_code == 200:
            logger.info("[JAEGER-CONNECTION] SUCCESS - Jaeger connection/auth working")
            return True

        logger.error(
            "[JAEGER-CONNECTION] FAILED - status=%s body=%s",
            response.status_code,
            response.text[:500],
        )
        return False

    except Exception as e:
        logger.exception("[JAEGER-CONNECTION] ERROR: %s", e)
        return False
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
asyncio.run(_check_jaeger_connection())
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
