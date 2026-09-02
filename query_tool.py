"""
Universal Query Tools for Incident Manager.

These tools allow the LLM to dynamically generate and execute queries
for SQL, ELK, Jaeger, and schema discovery without predefined JSON definitions.
"""

import os
import json
import base64
import httpx
import asyncio
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

# Disable OpenTelemetry for internal tools
os.environ["OTEL_SDK_DISABLED"] = "true"

from crewai.tools import BaseTool
from new_flow.utils.logs import execute_elk_query
from new_flow.utils.oracle_connection import execute_oracle_query_async
from new_flow.utils.http_calls import http_client_post_async, http_client_get_async

# Import service metadata for validation
from new_flow.tools.service_metadata import (
    get_services,
    get_service_tags,
    validate_tag_with_warning,
    get_elk_indexes,
)

# Import jaeger_endpoint from app_config (single source of truth)
from new_flow.tools.app_config import get_jaeger_endpoint, get_jager_auth_token

logger = __import__('logging').getLogger(__name__)


class SQLQueryInput(BaseModel):
    """Input for SQL Query Tool."""
    query: str = Field(description="The SQL query to execute")
    db_instance: str = Field(default="main", description="Database instance (main, idp, etc.)")


class ELKQueryInput(BaseModel):
    """Input for ELK Query Tool."""
    app: str = Field(description="Application name: optimus, cbs, idp (use lowercase). Determines which ELK index to search.")
    query_json: str = Field(description="ELK query in JSON format. Common fields: customer_id, txn_id, message, logtype, @timestamp")
    parameter_name: Optional[str] = Field(default=None, description="Parameter name to substitute in query")
    parameter_value: Optional[str] = Field(default=None, description="Parameter value to substitute")


class JaegerQueryInput(BaseModel):
    """Input for Jaeger Trace Tool.
    
    For backward compatibility: if 'app' is not provided, defaults to 'optimus'.
    If 'service' is not provided, queries all services.
    
    IMPORTANT: Use time_range_index to specify which 24h window to search:
    - 0 = 0-24h ago (most recent)
    - 1 = 24-48h ago
    - 2 = 48-72h ago
    - 3 = 72-96h ago
    - 4 = 96-120h ago
    - 5 = 120-144h ago
    - 6 = 144-168h ago (oldest)
    """
    app: Optional[str] = Field(default="optimus", description="Application name: optimus, cbs, idp (use lowercase). Default: optimus")
    service: str = Field(description="Service name (e.g., upi-api, idp-api, cbs-backend). Use GetServicesTool to discover valid services. If not provided, searches all services.")
    tag_name: str = Field(description="Identifier type: Use GetServiceTagsTool to get valid tags for the service. Common: customer_id, txn_id, mobile_number, ucic, user_id, session_tracking_id")
    tag_value: str = Field(description="The identifier value (e.g., customer number, transaction ID)")
    time_range_index: int = Field(default=0, description="Which 24h time window to search: 0=0-24h, 1=24-48h, 2=48-72h, 3=72-96h, 4=96-120h, 5=120-144h, 6=144-168h. Default 0 (most recent 24h).")
    return_raw_json: bool = Field(default=False, description="If true, returns full trace JSON for LLM to analyze. Use when you need complete trace data including all spans and tags.")


class TableSchemaInput(BaseModel):
    """Input for Table Schema Tool."""
    table_name: str = Field(description="Table name to get schema for")
    app: str = Field(default="cbs", description="Application name (cbs, optimus, idp)")


class ListTablesInput(BaseModel):
    """Input for List Tables Tool."""
    pattern: str = Field(default="%", description="Table name pattern (e.g., '%GLD%', '%INF%')")
    app: str = Field(default="cbs", description="Application name (cbs, optimus, idp)")


class SimilaritySearchInput(BaseModel):
    """Input for Similarity Search Tool."""
    query: str = Field(description="Natural language description of the incident")
    app: str = Field(default="cbs", description="Application name")
    top_k: int = Field(default=5, description="Number of similar incidents to return")


# For local testing, use PROD endpoint
# In K8s, these will be overridden by environment variables
JAEGER_API_BASE = os.getenv("JAEGER_API_BASE", "https://tracing.uat-opt.idfcfirstbank.com/api")
JAEGER_AUTH_TOKEN = os.getenv("JAEGER_AUTH_TOKEN")

# SSL certificate verification - must match the CA cert used for ELK/LLM calls
CA_CERT_FILE = os.getenv("CA_CERT_FILE", "./IDFCBANKCA.pem")
CA_CERT_PATH = CA_CERT_FILE  # Alias for consistency with logs.py

# Count-based cap only — keep failed trace COUNT bounded, but never slice trace CONTENT.
MAX_ERROR_TRACES = 20

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
# Generous guard on a single request/response field only — not a per-trace/per-line cap.
_JAEGER_MAX_PAYLOAD_CHARS = 5000


def _jaeger_auth_headers(app: str):
    token = get_jager_auth_token(app)
    return {"Authorization": f"Basic {token}"} if token else {}


def _jaeger_us_to_ist(microseconds_ts):
    from datetime import timezone, timedelta
    import datetime as _datetime
    seconds = microseconds_ts / 1_000_000
    dt_utc = _datetime.datetime.fromtimestamp(seconds, tz=timezone.utc)
    ist_tz = timezone(timedelta(hours=5, minutes=30))
    return dt_utc.astimezone(ist_tz).strftime('%Y-%m-%d %H:%M:%S IST')


def _try_decode_base64(value: str) -> str:
    """If value looks like base64-encoded text, decode it; otherwise return unchanged.
    Best-effort only — falls back to the original string on any failure or ambiguity."""
    if not value or not isinstance(value, str):
        return value
    stripped = value.strip()
    if len(stripped) < 8 or len(stripped) % 4 != 0:
        return value
    try:
        decoded_bytes = base64.b64decode(stripped, validate=True)
        decoded_str = decoded_bytes.decode('utf-8')
        if decoded_str.isprintable() or decoded_str.strip().startswith(("{", "[")):
            return decoded_str
    except Exception:
        pass
    return value


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
    """Process a Jaeger trace and return output lines and status info.

    Root-cause-aware pruning: when a trace contains any error span, only the
    error span(s) plus their FULL descendant subtree get full detail (tags,
    logs, request/response) — because the actual root cause is usually deeper
    in the tree than where the error status was first reported. Ancestor spans
    up to root are shown as lightweight breadcrumbs only (service - operation
    name, no tag/log dump), so you still see which flow the error occurred in.
    Unrelated branches with no connection to any error are dropped entirely.

    Traces with no error at all are processed in full as before (happy path).

    Returns a tuple: (output_lines, session_id, status_info)
    status_info = {
        "http_codes": set of HTTP status codes,
        "error_logs": list of error log messages,
        "error_flag": bool,
    }
    """
    from collections import defaultdict
    spans = trace.get("spans", [])
    processes = trace.get("processes", {})
    
    id_to_span = {s["spanID"]: s for s in spans}
    pid_to_service = {pid: p["serviceName"] for pid, p in processes.items()}
    children_map = defaultdict(list)
    parent_map = {}
    
    all_ids = set(id_to_span.keys())
    child_ids = set()
    
    for span in spans:
        for ref in span.get("references", []):
            parent_id = ref.get("spanID")
            if parent_id in id_to_span:
                children_map[parent_id].append(span["spanID"])
                child_ids.add(span["spanID"])
                parent_map[span["spanID"]] = parent_id
    
    roots = all_ids - child_ids
    if not roots:
        return [], None, {"http_codes": set(), "error_logs": [], "error_flag": False}

    def has_error_signal(span):
        for tag in span.get("tags", []):
            if tag.get("key") == "http.status_code" and str(tag.get("value", "")).startswith(("4", "5")):
                return True
            if tag.get("key") == "error" and tag.get("value") in (True, "true", "True"):
                return True
        for log in span.get("logs", []):
            for f in log.get("fields", []):
                if f.get("key") == "level" and str(f.get("value", "")).lower() in ("error", "fatal", "critical"):
                    return True
        return False

    # Pass 1: which spans carry an error signal themselves
    error_span_ids = {sid for sid, s in id_to_span.items() if has_error_signal(s)}
    only_prune = bool(error_span_ids)   # if no error anywhere, process everything (happy path)

    # Pass 2: FULL detail = error span + its entire descendant subtree
    keep_full = set()
    def collect_descendants(span_id):
        keep_full.add(span_id)
        for child_id in children_map.get(span_id, []):
            collect_descendants(child_id)

    for eid in error_span_ids:
        collect_descendants(eid)

    # Pass 3: lightweight breadcrumb = ancestor chain up to root
    keep_context = set()
    for eid in error_span_ids:
        cur = parent_map.get(eid)
        while cur:
            if cur not in keep_full:
                keep_context.add(cur)
            cur = parent_map.get(cur)
    
    import datetime as _datetime
    import time as _time
    
    result = []
    session_id = None
    status = {"http_codes": set(), "error_logs": [], "error_flag": False}
    
    def process_span(span_id, depth=0):
        nonlocal session_id
        span = id_to_span[span_id]
        indent = "    " * depth
        op_name = span.get("operationName", "")
        service_name = pid_to_service.get(span.get("processID"), "")
        
        if depth == 0:
            result.append("Time: " + _jaeger_us_to_ist(span["startTime"]))
        
        is_noise = op_name in _JAEGER_EXCLUDED_OPS or service_name in _JAEGER_EXCLUDED_SERVICES

        # Unrelated branch (no connection to any error) — skip entirely, but still
        # recurse through it in case a descendant needs visiting (safety net).
        if only_prune and span_id not in keep_full and span_id not in keep_context:
            for child_id in children_map.get(span_id, []):
                process_span(child_id, depth + 1)
            return
        
        if not is_noise:
            result.append(f"{indent}{span_id} - {service_name} - {op_name}")

            # Breadcrumb-only span: name shown, no tag/log/payload dump.
            if only_prune and span_id in keep_context:
                for child_id in children_map.get(span_id, []):
                    process_span(child_id, depth + 1)
                return

            for tag in span.get("tags", []):
                key = tag.get("key")
                val = tag.get("value")
                if key == "http.status_code":
                    status["http_codes"].add(str(val))
                    result.append(f"{indent}HTTP Status: {val}")
                elif key == "error" and val in (True, "true", "True"):
                    status["error_flag"] = True
                elif key == "session_tracking_id" and not session_id:
                    session_id = val
            
            for log in span.get("logs", []):
                fields = {f.get("key"): f.get("value") for f in log.get("fields", [])}
                
                # Extract session_tracing_id from logs
                sid = fields.get("session_tracing_id")
                if not session_id and sid and sid not in ("no-session-trace-id", ""):
                    session_id = sid
                
                level = str(fields.get("level", "")).lower()
                event = fields.get("event", "")
                if level in ("error", "fatal", "critical") and event:
                    cls = fields.get("Class") or fields.get("class") or ""
                    method = fields.get("Method") or fields.get("method") or ""
                    ctx = f" [{cls}.{method}]" if (cls or method) else ""
                    result.append(f"{indent}  ERROR{ctx}: {event}")
                    status["error_logs"].append(event)
                
                for payload_key in ("request", "response"):
                    if fields.get(payload_key):
                        raw_val = str(fields[payload_key])
                        decoded_val = _try_decode_base64(raw_val)
                        result.append(f"{indent}  {payload_key}: {decoded_val[:_JAEGER_MAX_PAYLOAD_CHARS]}")
        
        for child_id in children_map.get(span_id, []):
            process_span(child_id, depth + 1)
    
    for root_id in roots:
        process_span(root_id)
    
    return result, session_id, status


async def _jaeger_fetch(app: str, service: str, tag_name: str, tag_value: str, start_us: int = None, end_us: int = None):
    import time as _time
    import json as _json
    
    # Get app-specific Jaeger endpoint if available
    jaeger_endpoint = get_jaeger_endpoint(app)
    api_base = jaeger_endpoint if jaeger_endpoint else JAEGER_API_BASE
    
    # Calculate time range
    now_us = int(_time.time() * 1_000_000)
    hours_ago = 24  # Default, will be updated if timestamps are passed
    
    if start_us is None or end_us is None:
        start_us = now_us - hours_ago * 3600 * 1_000_000
        end_us = now_us
        hours_label = f"{hours_ago}h"
    else:
        # Calculate hours from the range for logging
        hours_diff = (end_us - start_us) / (3600 * 1_000_000)
        hours_ago = int(hours_diff)  # Update for error message
        hours_label = f"{hours_diff:.0f}h"
    
    # Handle mobile number format
    if tag_name == "mobile_number" and not tag_value.startswith("+"):
        if len(tag_value) == 10:
            tag_value = f"+91{tag_value}"
    
    # Build query params - use exact service name (Jaeger service names are case-sensitive)
    headers = _jaeger_auth_headers(app)
    params = {"service": service, "start": start_us, "end": end_us, "limit": 100}
    
    # Only add tags filter if both tag_name and tag_value are provided and non-empty
    if tag_name and tag_value and str(tag_name).strip() and str(tag_value).strip():
        tags = _json.dumps({tag_name: tag_value})
        params["tags"] = tags
    
    logger.info(f"[JAEGER] app={app} service={service} {tag_name}={tag_value} range={hours_label} endpoint={api_base} params={params}")
    
    try:
        async with httpx.AsyncClient(timeout=45,verify=CA_CERT_PATH) as client:
            response = await client.get(f"{api_base}/traces", params=params, headers=headers)
            if response.status_code != 200:
                logger.error(f"[JAEGER][FETCH] Non-200 response | status={response.status_code} body={response.text[:500]}")
                return {"total_traces_scanned": 0, "failed_traces": [], "error": f"API returned {response.status_code}"}
            data = response.json()

            traces = data.get("data", [])
            logger.info(f"[JAEGER][FETCH] Raw response | trace_count={len(traces)}")
            
            # Note: Jaeger API may return total=0 but still have data in the array
            if not traces or len(traces) == 0:
                logger.info(f"[JAEGER][FETCH] No traces in raw response for service={service} {tag_name}={tag_value} range={hours_label}")
                return {"total_traces_scanned": 0, "failed_traces": [], "note": f"No traces found in last {hours_ago} hours"}
            
            failed = []
            happy_hits = {}  # Track successful traces by endpoint
            sessions = set()
            success_sessions = set()
            
            for idx, trace in enumerate(traces):
                output_lines, session_id, status_info = _jaeger_process_trace(trace)
                
                # Track session
                if session_id:
                    sessions.add(session_id)
                
                # Determine if trace is an error or success
                error_codes = sorted(c for c in status_info["http_codes"] if c.startswith(("4", "5")))
                has_2xx = any(c.startswith("2") for c in status_info["http_codes"])
                is_error = bool(error_codes) or status_info["error_flag"] or bool(status_info["error_logs"])

                logger.info(
                    f"[JAEGER][TRACE {idx+1}/{len(traces)}] traceID={trace.get('traceID', 'unknown')} "
                    f"http_codes={status_info['http_codes']} error_flag={status_info['error_flag']} "
                    f"error_logs_count={len(status_info['error_logs'])} classified={'ERROR' if is_error else ('SUCCESS' if has_2xx else 'UNKNOWN')}"
                )
                
                if not is_error and has_2xx:
                    # Happy path - track successful endpoint
                    if session_id:
                        success_sessions.add(session_id)
                    if len(output_lines) > 1:
                        # Extract endpoint from second line (service - operation)
                        endpoint_key = output_lines[1] if output_lines else ""
                        happy_hits[endpoint_key] = happy_hits.get(endpoint_key, 0) + 1
                    continue
                
                if is_error and output_lines:
                    # Pruning already happened at the span level in _jaeger_process_trace —
                    # keep the trace's full (already-pruned) text, no further line-count cap.
                    failed.append("\n".join(output_lines))
            
            logger.info(
                f"[JAEGER][FETCH] Classification summary | total={len(traces)} "
                f"failed={len(failed)} happy_endpoints={len(happy_hits)} "
                f"sessions={len(sessions)} success_sessions={len(success_sessions)}"
            )
            
            return {
                "total_traces_scanned": len(traces),
                "failed_traces": failed[:MAX_ERROR_TRACES],  # count cap only, content is full
                "happy_hits": happy_hits,
                "sessions": list(sessions),
                "success_sessions": list(success_sessions),
                "hours_ago": hours_ago,
                "service": service,
                "tag_name": tag_name,
                "tag_value": tag_value
            }
    except Exception as e:
        logger.error(f"[JAEGER][FETCH] Exception during fetch | type={type(e).__name__} msg='{e}'", exc_info=True)
        return {"total_traces_scanned": 0, "failed_traces": [], "error": str(e)}


class SQLQueryTool(BaseTool):
    """
    Execute Oracle SQL queries against CBS/Optimus database.
    
    Use this tool to:
    - Check account status, freezes, balances
    - Query customer data
    - Look up transaction records
    
    The LLM should generate appropriate SQL queries based on the incident.
    """
    name: str = "execute_db_query"
    description: str = (
        "Executes Oracle SQL query against the database. "
        "Use this to check account status, freezes, balances, customer data. "
        "Parameters: query (SQL string), db_instance (main/idp, default main)."
    )
    args_schema: type = SQLQueryInput
    
    def _run(self, query: str, db_instance: str = "main") -> str:
        return asyncio.run(self._arun(query, db_instance))
    
    async def _arun(self, query: str, db_instance: str = "main") -> str:
        try:
            logger.info(f"[SQLQueryTool] Executing: {query[:200]}...")
            result = await execute_oracle_query_async(
                query,
                app="cbs",  # Will be overridden based on context
                db_instance=db_instance
            )
            
            if result.get("success"):
                data = result.get("data", [])
                if not data:
                    return "Query executed successfully. No results returned."
                
                # Format results
                output = [f"Found {len(data)} rows:"]
                for row in data[:20]:  # Limit to 20 rows
                    output.append(str(row))
                
                if len(data) > 20:
                    output.append(f"... and {len(data) - 20} more rows")
                
                return "\n".join(output)
            else:
                return f"Query Error: {result.get('error')}"
        
        except Exception as e:
            logger.error(f"[SQLQueryTool] Error: {e}")
            return f"Query Error: {str(e)}"


class ELKQueryTool(BaseTool):
    name: str = "search_elk_logs"
    description: str = (
        "Search Elasticsearch for application logs. "
        "Use for transaction errors, API failures, detailed debugging. "
        "Parameters: app (optimus/cbs/idp), query_json (ELK query), parameter_name, parameter_value."
    )
    args_schema: type = ELKQueryInput
    
    def _run(self, app: str, query_json: str, parameter_name: str = None, parameter_value: str = None) -> str:
        return asyncio.run(self._arun(app, query_json, parameter_name, parameter_value))
    
    async def _arun(self, app: str, query_json: str, parameter_name: str = None, parameter_value: str = None) -> str:
        try:
            elk_indexes = get_elk_indexes(app)
            elk_index = elk_indexes.get("elk_log_index")  # Default to log index
            
            logger.info(f"[ELKQueryTool] Executing query for app={app}, index={elk_index}...")
            
            # Import execute_elk_query from utils.logs - but we need to pass index
            # For now, we'll use the existing function and it will use default index
            # A future update can pass app-specific index
            
            result = await execute_elk_query(
                query_json,
                parameter_name=parameter_name,
                parameter_value=parameter_value
            )
            
            # Prepend app info to result
            if result:
                result = f"[App: {app} | Index: {elk_index}]\n{result}"
            
            return result
        
        except Exception as e:
            logger.error(f"[ELKQueryTool] Error: {e}")
            return f"ELK Query Error: {str(e)}"



def _parse_known_tags_from_warning(warning: str) -> List[str]:
    import re
    # Match content between "Known tags: [" and "]"
    match = re.search(r"Known tags: \[(.*?)\]", warning)
    if match:
        tags_str = match.group(1)
        # Parse the list - could be quoted strings or plain values
        tags = re.findall(r"['\"]([^'\"]+)['\"]", tags_str)
        if not tags:
            # Try plain comma-separated values
            tags = [t.strip() for t in tags_str.split(',')]
        return [t.strip() for t in tags if t.strip()]
    return []


def _jaeger_has_root_cause(result: dict) -> bool:
    failed = result.get("failed_traces", [])
    if not failed:
        return False
    
    # Check if any failed trace has a clear error message (not just HTTP errors)
    for trace in failed:
        # Look for error patterns that indicate root cause found
        if "ERROR:" in trace or "error" in trace.lower():
            # Check for specific error messages (not just status codes)
            if any(pattern in trace for pattern in ["Exception", "Error:", "Failed:", "DENIED", "FORBIDDEN", "ACCESS_DENIED"]):
                return True
    return False


class JaegerTraceTool(BaseTool):
    name: str = "fetch_jaeger_traces"
    description: str = (
        "Fetch distributed traces from Jaeger for troubleshooting. "
        "Parameters: app (optimus/cbs/idp), service, tag_name, tag_value, hours_ago. "
        "IMPORTANT: Use GetServicesTool and GetServiceTagsTool to discover valid services and tags first. "
        "If no traces in 24h, tool will automatically try 48h and 72h."
    )
    args_schema: type = JaegerQueryInput
    
    def _run(self, app: str = None, service: str = None, tag_name: str = None, tag_value: str = None, 
             hours_ago: int = 24, service_name: str = None, identifier_name: str = None, 
             identifier_value: str = None, return_raw_json: bool = False, **kwargs) -> str:
        # Handle backward compatibility with old parameter names
        if service_name and not service:
            service = service_name
        if identifier_name and not tag_name:
            tag_name = identifier_name
        if identifier_value and not tag_value:
            tag_value = identifier_value
        if not app:
            app = "optimus"  # Default
            
        return asyncio.run(self._arun(app, service, tag_name, tag_value, hours_ago, return_raw_json))
    
    async def _arun(self, app: str = None, service: str = None, tag_name: str = None, 
                   tag_value: str = None, time_range_index: int = 0, return_raw_json: bool = False, **kwargs) -> str:
        """
        Fetch Jaeger traces for a SPECIFIC 24h time range.
        
        time_range_index:
        - 0 = 0-24h ago (most recent)
        - 1 = 24-48h ago
        - 2 = 48-72h ago
        - 3 = 72-96h ago
        - 4 = 96-120h ago
        - 5 = 120-144h ago
        - 6 = 144-168h ago (oldest)
        
        The Execute Agent is responsible for looping through time ranges (0-6)
        and deciding when to stop based on the results.
        """
        
        if kwargs.get('service_name') and not service:
            service = kwargs.get('service_name')
        if kwargs.get('identifier_name') and not tag_name:
            tag_name = kwargs.get('identifier_name')
        if kwargs.get('identifier_value') and not tag_value:
            tag_value = kwargs.get('identifier_value')
        
        # Support legacy hours_ago parameter for backward compatibility
        if kwargs.get('hours_ago') and time_range_index == 0:
            # Convert hours_ago to time_range_index
            hours = kwargs.get('hours_ago')
            if hours <= 24:
                time_range_index = 0
            elif hours <= 48:
                time_range_index = 1
            elif hours <= 72:
                time_range_index = 2
            elif hours <= 96:
                time_range_index = 3
            elif hours <= 120:
                time_range_index = 4
            elif hours <= 144:
                time_range_index = 5
            else:
                time_range_index = 6
        
        if not app:
            app = "optimus"
        if not service:
            service = ""

        validation_warning = validate_tag_with_warning(app, service, tag_name)
        
        # Search ONLY ONE 24h time range based on time_range_index
        import time as _time
        now_us = int(_time.time() * 1_000_000)
        
        # Define time ranges: (start_offset_hours, end_offset_hours)
        time_ranges = [
            (0, 24), (24, 48), (48, 72), (72, 96), 
            (96, 120), (120, 144), (144, 168)
        ]
        
        # Clamp to valid range
        time_range_index = max(0, min(6, time_range_index))
        start_offset, end_offset = time_ranges[time_range_index]
        
        start_us = now_us - end_offset * 3600 * 1_000_000
        end_us = now_us - start_offset * 3600 * 1_000_000
        range_label = f"{start_offset}-{end_offset}h ago"

        logger.info(f"[JaegerTraceTool] Searching {range_label} for service={service}, tag={tag_name}={tag_value}")

        result = await _jaeger_fetch(app, service, tag_name, tag_value, start_us, end_us)

        # ── DIAGNOSTIC: dump raw counts so we can see what was actually found ──
        logger.info(
            f"[JaegerTraceTool][RAW RESULT] service={service} tag={tag_name}={tag_value} "
            f"range={range_label} total_scanned={result.get('total_traces_scanned', 0)} "
            f"failed_traces_count={len(result.get('failed_traces', []))} "
            f"happy_hits={result.get('happy_hits', {})} "
            f"sessions={len(result.get('sessions', []))} "
            f"success_sessions={len(result.get('success_sessions', []))} "
            f"error={result.get('error')}"
        )
        if result.get('failed_traces'):
            for i, ft in enumerate(result['failed_traces'][:3], 1):
                logger.info(f"[JaegerTraceTool][FAILED TRACE #{i}] {ft[:500]}")
        else:
            logger.info(
                f"[JaegerTraceTool][NO ERROR EVIDENCE] All {result.get('total_traces_scanned', 0)} "
                f"scanned traces were successful/non-error"
            )

        scanned = result.get("total_traces_scanned", 0)

        if scanned and scanned > 0:
            # Found traces
            
            # Check if raw JSON is requested
            if return_raw_json:
                import json
                return json.dumps(result, indent=2, default=str)
            
            output = []
            
            # Add validation warning if present
            if validation_warning:
                output.append(validation_warning)
                output.append("\n" + "="*60 + "\n")
            
            output.append(f"Found {scanned} traces in time range {range_label}:")
            
            failed = result.get("failed_traces", [])
            happy_hits = result.get("happy_hits", {})
            sessions = result.get("sessions", [])
            success_sessions = result.get("success_sessions", [])
            
            # Add happy path tracking info
            if happy_hits:
                output.append("\n=== HAPPY PATH (Successful Traces) ===")
                for endpoint, count in happy_hits.items():
                    output.append(f"  {endpoint} → Success: {count}")
            
            # Add session info
            if sessions:
                output.append(f"\n=== SESSIONS ===")
                output.append(f"  Total sessions: {len(sessions)}")
                output.append(f"  Successful sessions: {len(success_sessions)}")
            
            if failed:
                if len(failed) > MAX_ERROR_TRACES:
                    output.append(f"\n[Showing {MAX_ERROR_TRACES} of {len(failed)} error evidence - truncated for context]")
                    for trace in failed[:MAX_ERROR_TRACES]:
                        output.append(f"\n{trace}")
                else:
                    output.append(f"\n=== {len(failed)} ERROR EVIDENCE ===")
                    for trace in failed:
                        output.append(f"\n{trace}")
                
                # If root cause found, note it but let LLM decide next step
                if _jaeger_has_root_cause(result):
                    output.append("\n[ROOT CAUSE IDENTIFIED in this time range]")
            else:
                output.append(f"\n[Showing {min(scanned, 5)} traces for analysis]")
            
            final_output = "\n".join(output)
            logger.info(f"[JaegerTraceTool][OUTPUT PREVIEW] {final_output[:800]}")
            return final_output
        
        # No traces found in this time range
        output = []
        if validation_warning:
            output.append(validation_warning)
            output.append("\n" + "="*60 + "\n")
        
        output.append(
            f"No traces found in time range {range_label} for this identifier.\n\n"
            f"Use time_range_index={time_range_index + 1} to search the next older time range (24h older)."
        )
        return "\n".join(output)


class GetTableSchemaTool(BaseTool):
    """
    Get column information for a database table.
    
    Use this tool before writing SQL queries to understand available columns.
    """
    name: str = "get_table_schema"
    description: str = (
        "Get column names and types for a database table. "
        "Use this before writing SQL queries to understand available columns."
    )
    args_schema: type = TableSchemaInput
    
    def _run(self, table_name: str, app: str = "cbs") -> str:
        """Get table schema synchronously."""
        return asyncio.run(self._arun(table_name, app))
    
    async def _arun(self, table_name: str, app: str = "cbs") -> str:
        """Get table schema from Oracle metadata."""
        # Query Oracle ALL_TAB_COLUMNS
        query = """
            SELECT column_name, data_type, data_length, nullable, column_id
            FROM all_tab_columns 
            WHERE owner = 'FISONLPD' AND table_name = UPPER(:table_name)
            ORDER BY column_id
        """
        
        try:
            result = await execute_oracle_query_async(
                query,
                app=app,
                db_instance="main",
                params={"table_name": table_name}
            )
            
            if result.get("success"):
                data = result.get("data", [])
                if not data:
                    return f"No columns found for table '{table_name}'. Check if table name is correct."
                
                output = [f"Schema for {table_name}:"]
                for col in data:
                    col_name = col[0] if len(col) > 0 else ""
                    col_type = col[1] if len(col) > 1 else ""
                    col_len = col[2] if len(col) > 2 else ""
                    nullable = col[3] if len(col) > 3 else ""
                    output.append(f"  {col_name:30} {col_type:15} nullable={nullable}")
                
                return "\n".join(output)
            else:
                return f"Error: {result.get('error')}"
        
        except Exception as e:
            return f"Error getting schema: {str(e)}"


# ─────────────────────────────────────────────────────────────────────────────
# List Tables Tool
# ─────────────────────────────────────────────────────────────────────────────

class ListTablesTool(BaseTool):
    """
    List database tables matching a pattern.
    
    Use this to discover available tables in the database.
    """
    name: str = "list_tables"
    description: str = (
        "List database tables matching a pattern. "
        "Use this to discover available tables. "
        "Example pattern: '%GLD%' or '%INF%'."
    )
    args_schema: type = ListTablesInput
    
    def _run(self, pattern: str = "%", app: str = "cbs") -> str:
        """List tables synchronously."""
        return asyncio.run(self._arun(pattern, app))
    
    async def _arun(self, pattern: str = "%", app: str = "cbs") -> str:
        """List tables from Oracle."""
        query = """
            SELECT table_name 
            FROM user_tables 
            WHERE table_name LIKE UPPER(:pattern)
            ORDER BY table_name
        """
        
        try:
            result = await execute_oracle_query_async(
                query,
                app=app,
                db_instance="main",
                params={"pattern": pattern}
            )
            
            if result.get("success"):
                data = result.get("data", [])
                if not data:
                    return f"No tables found matching pattern '{pattern}'."
                
                tables = [row[0] for row in data]
                return f"Found {len(tables)} tables:\n" + "\n".join(f"  - {t}" for t in tables)
            else:
                return f"Error: {result.get('error')}"
        
        except Exception as e:
            return f"Error listing tables: {str(e)}"


# ─────────────────────────────────────────────────────────────────────────────
# Similarity Search Tool
# ─────────────────────────────────────────────────────────────────────────────

class SimilaritySearchTool(BaseTool):
    """
    Search for similar historic incidents.
    
    Use this to find resolved incidents with similar problems.
    The LLM can use these to understand how similar issues were resolved.
    """
    name: str = "search_historic_incidents"
    description: str = (
        "Search for similar historic incidents to understand how similar "
        "problems were resolved. Returns top similar incidents with their resolutions."
    )
    args_schema: type = SimilaritySearchInput
    
    def _run(self, query: str, app: str = "cbs", top_k: int = 5) -> str:
        """Search historic incidents synchronously."""
        return asyncio.run(self._arun(query, app, top_k))
    
    async def _arun(self, query: str, app: str = "cbs", top_k: int = 5) -> str:
        """Search historic incidents using semantic search."""
        try:
            # Import here to avoid circular imports
            from agents.context_builder import run_incident_context_crew_async
            
            context = await run_incident_context_crew_async(
                f"Find similar incidents: {query}",
                application=app
            )
            
            if context and context.strip():
                return context
            else:
                return "No similar historic incidents found."
        
        except Exception as e:
            logger.error(f"[SimilaritySearchTool] Error: {e}")
            return f"Error searching historic incidents: {str(e)}"



def get_tools_for_agent(agent_type: str) -> List[BaseTool]:
    if agent_type == "plan":
        return [
            SimilaritySearchTool(),
            JaegerTraceTool(),
        ]
    elif agent_type == "execute":
        return [
            JaegerTraceTool(),
            ELKQueryTool(),
            SQLQueryTool(),
            GetTableSchemaTool(),
            ListTablesTool(),
        ]
    else:
        return []

def get_all_tools() -> List[BaseTool]:
    """Get all universal tools."""
    return [
        SimilaritySearchTool(),
        JaegerTraceTool(),
        ELKQueryTool(),
        SQLQueryTool(),
        GetTableSchemaTool(),
        ListTablesTool(),
    ]
