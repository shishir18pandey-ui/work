"""
Universal Query Tools for Incident Manager.

These tools allow the LLM to dynamically generate and execute queries
for SQL, ELK, Jaeger, and schema discovery without predefined JSON definitions.
"""

import os
import re
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

# ─────────────────────────────────────────────────────────────────────────────
# Evidence budget: generous, ranked, never cut mid-record. With structured
# extraction real incidents land at 2k-8k chars, so this almost never binds.
# When it does, EVERY distinct error's header (service/operation/status/message)
# is still always included - only payload BODIES of lowest-ranked records are
# shed, and that fact is stated explicitly in the output.
# ─────────────────────────────────────────────────────────────────────────────
_EVIDENCE_CHAR_BUDGET = 120000      # ~30k tokens
_PAYLOAD_SLOT = 8000                # guard against one pathological single payload


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
    Best-effort only - falls back to the original string on any failure or ambiguity."""
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
    """Lightweight pass used ONLY for session-id extraction and happy-path
    endpoint labels. Error content is handled entirely by _extract_error_records
    below. This no longer dumps payloads or subtrees - that was the source of
    multi-million-character outputs.

    Returns a tuple: (output_lines, session_id, status_info)
    """
    from collections import defaultdict
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
        return [], None, {"http_codes": set(), "error_logs": [], "error_flag": False}

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

        if not is_noise:
            result.append(f"{indent}{span_id} - {service_name} - {op_name}")
            for tag in span.get("tags", []):
                key = tag.get("key")
                val = tag.get("value")
                if key == "http.status_code":
                    status["http_codes"].add(str(val))
                elif key == "error" and val in (True, "true", "True"):
                    status["error_flag"] = True
                elif key == "session_tracking_id" and not session_id:
                    session_id = val

            for log in span.get("logs", []):
                fields = {f.get("key"): f.get("value") for f in log.get("fields", [])}
                sid = fields.get("session_tracing_id")
                if not session_id and sid and sid not in ("no-session-trace-id", ""):
                    session_id = sid

                level = str(fields.get("level", "")).lower()
                event = fields.get("event", "")
                if level in ("error", "fatal", "critical") and event:
                    status["error_logs"].append(event)

        for child_id in children_map.get(span_id, []):
            process_span(child_id, depth + 1)

    for root_id in roots:
        process_span(root_id)

    return result, session_id, status


def _extract_error_records(trace) -> List[Dict]:
    """Extract structured error records from one trace.

    Walks EVERY span - no size-based skipping, no truncation, no subtree
    expansion. Returns [] if the trace contains no errors. All diagnostic
    signal lives in logs.fields where level=error, plus request/response
    payloads that actually carry error structure.
    """
    from collections import defaultdict
    spans = trace.get("spans", [])
    processes = trace.get("processes", {})
    trace_id = trace.get("traceID", "")

    id_to_span = {s["spanID"]: s for s in spans}
    pid_to_service = {pid: p["serviceName"] for pid, p in processes.items()}
    children_map = defaultdict(list)
    parent_map = {}

    for span in spans:
        for ref in span.get("references", []):
            pid = ref.get("spanID")
            if pid in id_to_span:
                children_map[pid].append(span["spanID"])
                parent_map[span["spanID"]] = pid

    def span_error_info(span):
        http_status = None
        error_flag = False
        for tag in span.get("tags", []):
            k, v = tag.get("key"), tag.get("value")
            if k == "http.status_code":
                http_status = str(v)
            elif k == "error" and v in (True, "true", "True"):
                error_flag = True

        error_events, error_payloads = [], []
        for log in span.get("logs", []):
            f = {x.get("key"): x.get("value") for x in log.get("fields", [])}
            level = str(f.get("level", "")).lower()
            event = f.get("event", "")
            if level in ("error", "fatal", "critical") and event:
                cls = f.get("Class") or f.get("class") or ""
                mth = f.get("Method") or f.get("method") or ""
                error_events.append({
                    "message": str(event),
                    "context": f"{cls}.{mth}" if (cls or mth) else "",
                })
            # Only keep payloads that actually carry error structure. This is
            # what stops healthy 200-OK response bodies from flooding evidence.
            for pk in ("request", "response"):
                if f.get(pk):
                    val = _try_decode_base64(str(f[pk]))
                    if any(m in val for m in (
                        "error_code", "errorCode", "error_message", "errorMessage",
                        '"error"', "Exception", "DENIED", "FORBIDDEN", "UNAUTHORIZED",
                    )):
                        error_payloads.append({"kind": pk, "body": val})

        is_error = (
            (http_status and http_status.startswith(("4", "5")))
            or error_flag or error_events or error_payloads
        )
        if not is_error:
            return None
        return {
            "http_status": http_status,
            "error_flag": error_flag,
            "error_events": error_events,
            "error_payloads": error_payloads,
        }

    error_ids = {}
    for sid, s in id_to_span.items():
        info = span_error_info(s)
        if info is not None:
            error_ids[sid] = info

    def has_error_descendant(x):
        for c in children_map.get(x, []):
            if c in error_ids or has_error_descendant(c):
                return True
        return False

    records = []
    for sid, info in error_ids.items():
        span = id_to_span[sid]
        depth, cur, chain = 0, sid, []
        while cur in parent_map:
            cur = parent_map[cur]
            depth += 1
            p = id_to_span[cur]
            chain.append(f"{pid_to_service.get(p.get('processID'), '')} - {p.get('operationName', '')}")

        records.append({
            "trace_id": trace_id,
            "span_id": sid,
            "service": pid_to_service.get(span.get("processID"), ""),
            "operation": span.get("operationName", ""),
            "depth": depth,
            "call_chain": list(reversed(chain))[-4:],
            # Leaf error = no failing descendant, i.e. this is where the failure
            # ORIGINATED rather than where it propagated to.
            "is_leaf_error": not has_error_descendant(sid),
            "start_time": span.get("startTime"),
            **info,
        })
    return records


def _score_record(r: Dict) -> int:
    """Higher = more likely the true root cause."""
    s = 0
    if r["is_leaf_error"]:
        s += 1000
    if r["error_payloads"]:
        s += 400
    if r["error_events"]:
        s += 300
    st = r.get("http_status") or ""
    if st.startswith("5"):
        s += 200
    elif st.startswith("4"):
        s += 120
    s += min(r["depth"], 10) * 10
    return s


def _dedupe_records(records: List[Dict]) -> List[Dict]:
    """Collapse identical repeated failures into one record + occurrence count.
    Nothing is silently dropped - the count is shown in the rendered output."""
    seen = {}
    for r in records:
        if r["error_events"]:
            msg = r["error_events"][0]["message"]
        elif r["error_payloads"]:
            msg = r["error_payloads"][0]["body"]
        else:
            msg = ""
        norm = re.sub(r'\d{6,}|[0-9a-f]{16,}', "*", msg)[:200]
        key = (r["service"], r["operation"], r.get("http_status"), norm)
        if key in seen:
            seen[key]["occurrences"] += 1
        else:
            r["occurrences"] = 1
            seen[key] = r
    return list(seen.values())


def _record_header_block(r: Dict, index: int) -> str:
    """Header-only rendering: service, operation, call chain, status, error
    messages - no payload bodies. Always included for every record."""
    parts = [
        f"--- ERROR #{index} "
        f"[{'ROOT-LEVEL FAILURE' if r['is_leaf_error'] else 'propagated'}] "
        f"{r['service']} - {r['operation']}"
        + (f" (x{r['occurrences']} identical occurrences)" if r["occurrences"] > 1 else "")
    ]
    if r["call_chain"]:
        parts.append("  via: " + " -> ".join(r["call_chain"]))
    if r.get("http_status"):
        parts.append(f"  HTTP: {r['http_status']}")
    for e in r["error_events"]:
        ctx = f" [{e['context']}]" if e["context"] else ""
        parts.append(f"  ERROR{ctx}: {e['message']}")
    if r["error_payloads"] and not r["error_events"]:
        parts.append(f"  [has {len(r['error_payloads'])} error payload(s)]")
    return "\n".join(parts)


def _render_evidence(all_records: List[Dict], traces_scanned: int) -> Dict:
    """Render ranked evidence within budget.

    Guarantee: every distinct error's header (service/operation/status/message)
    is ALWAYS included regardless of budget - no error code is ever invisible
    to the LLM. Only payload BODIES of the lowest-ranked records get dropped if
    the budget is exceeded, and that is stated explicitly per-record.
    """
    ranked = sorted(_dedupe_records(all_records), key=_score_record, reverse=True)

    if not ranked:
        return {
            "text": "",
            "total_errors": 0,
            "shown_full": 0,
            "shown_header_only": 0,
            "has_root_level": False,
        }

    # Pass 1: headers for every record - always included, always whole
    headers = [_record_header_block(r, i + 1) for i, r in enumerate(ranked)]
    headers_total = sum(len(h) for h in headers) + len(headers) * 2

    # Pass 2: add full payload bodies, highest-ranked first, until budget runs out
    remaining = max(_EVIDENCE_CHAR_BUDGET - headers_total, 0)
    full_blocks = {}
    payloads_included, payloads_omitted, truncated_payloads = 0, 0, 0

    for i, r in enumerate(ranked):
        if not r["error_payloads"]:
            continue
        payload_lines = []
        for p in r["error_payloads"]:
            b = p["body"]
            if len(b) > _PAYLOAD_SLOT:
                b = b[:_PAYLOAD_SLOT] + f" ...[PAYLOAD TRUNCATED - {len(p['body'])} chars total]"
                truncated_payloads += 1
            payload_lines.append(f"  {p['kind']}: {b}")
        block = "\n".join(payload_lines)

        if len(block) <= remaining:
            full_blocks[i] = block
            remaining -= len(block)
            payloads_included += 1
        else:
            payloads_omitted += 1

    body = []
    for i, r in enumerate(ranked):
        block = headers[i]
        if i in full_blocks:
            block += "\n" + full_blocks[i]
        elif r["error_payloads"]:
            block += (
                f"\n  [payload body omitted for size - {len(r['error_payloads'])} "
                f"payload(s) exist for this error. The error message/status above "
                f"is still accurate.]"
            )
        body.append(block)

    has_root_level = any(r["is_leaf_error"] for r in ranked)

    header_lines = [
        "=== ERROR EVIDENCE (ranked - root-level failures first) ===",
        f"Traces scanned: {traces_scanned} | Distinct errors: {len(ranked)} | "
        f"Full detail shown: {payloads_included} | Header-only (payload omitted): {payloads_omitted}",
    ]
    if truncated_payloads:
        header_lines.append(f"NOTE: {truncated_payloads} oversized payload(s) truncated (marked inline).")
    if payloads_omitted:
        header_lines.append(
            f"NOTE: Every distinct error above is listed with its message/status - "
            f"only the raw payload BODY was omitted for {payloads_omitted} lower-ranked "
            f"error(s) due to size. If the shown error messages don't clearly explain "
            f"the reported issue, say so explicitly rather than inferring a cause."
        )

    return {
        "text": "\n".join(header_lines) + "\n\n" + "\n\n".join(body),
        "total_errors": len(ranked),
        "shown_full": payloads_included,
        "shown_header_only": payloads_omitted,
        "has_root_level": has_root_level,
    }


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
        hours_diff = (end_us - start_us) / (3600 * 1_000_000)
        hours_ago = int(hours_diff)
        hours_label = f"{hours_diff:.0f}h"
    
    # Handle mobile number format
    if tag_name == "mobile_number" and not tag_value.startswith("+"):
        if len(tag_value) == 10:
            tag_value = f"+91{tag_value}"
    
    # Build query params - use exact service name (Jaeger service names are case-sensitive)
    headers = _jaeger_auth_headers(app)
    params = {"service": service, "start": start_us, "end": end_us, "limit": 100}
    
    if tag_name and tag_value and str(tag_name).strip() and str(tag_value).strip():
        tags = _json.dumps({tag_name: tag_value})
        params["tags"] = tags
    
    logger.info(f"[JAEGER] app={app} service={service} {tag_name}={tag_value} range={hours_label} endpoint={api_base} params={params}")
    
    try:
        async with httpx.AsyncClient(timeout=45, verify=CA_CERT_PATH) as client:
            response = await client.get(f"{api_base}/traces", params=params, headers=headers)
            if response.status_code != 200:
                logger.error(f"[JAEGER][FETCH] Non-200 response | status={response.status_code} body={response.text[:500]}")
                return {"total_traces_scanned": 0, "failed_traces": [], "total_errors": 0, "error": f"API returned {response.status_code}"}
            data = response.json()

            traces = data.get("data", [])
            logger.info(f"[JAEGER][FETCH] Raw response | trace_count={len(traces)}")
            
            if not traces or len(traces) == 0:
                logger.info(f"[JAEGER][FETCH] No traces in raw response for service={service} {tag_name}={tag_value} range={hours_label}")
                return {"total_traces_scanned": 0, "failed_traces": [], "total_errors": 0, "note": f"No traces found in last {hours_ago} hours"}
            
            all_error_records = []
            happy_hits = {}
            sessions = set()
            success_sessions = set()
            error_trace_count = 0
            
            for idx, trace in enumerate(traces):
                output_lines, session_id, status_info = _jaeger_process_trace(trace)
                if session_id:
                    sessions.add(session_id)

                records = _extract_error_records(trace)  # walks EVERY span, no skipping

                has_2xx = any(c.startswith("2") for c in status_info["http_codes"])
                is_error = bool(records)

                logger.info(
                    f"[JAEGER][TRACE {idx+1}/{len(traces)}] traceID={trace.get('traceID', 'unknown')} "
                    f"error_records={len(records)} "
                    f"root_level={sum(1 for r in records if r['is_leaf_error'])} "
                    f"classified={'ERROR' if is_error else ('SUCCESS' if has_2xx else 'UNKNOWN')}"
                )
                
                if not is_error and has_2xx:
                    if session_id:
                        success_sessions.add(session_id)
                    if len(output_lines) > 1:
                        endpoint_key = output_lines[1] if output_lines else ""
                        happy_hits[endpoint_key] = happy_hits.get(endpoint_key, 0) + 1
                    continue
                
                if records:
                    error_trace_count += 1
                    all_error_records.extend(records)
            
            evidence = _render_evidence(all_error_records, len(traces))

            logger.info(
                f"[JAEGER][FETCH] Evidence built | traces={len(traces)} error_traces={error_trace_count} "
                f"distinct_errors={evidence['total_errors']} full_detail={evidence['shown_full']} "
                f"header_only={evidence['shown_header_only']} root_level={evidence['has_root_level']} "
                f"chars={len(evidence['text'])}"
            )
            
            return {
                "total_traces_scanned": len(traces),
                "evidence_text": evidence["text"],
                "total_errors": evidence["total_errors"],
                "errors_shown_full": evidence["shown_full"],
                "errors_header_only": evidence["shown_header_only"],
                "has_root_level_error": evidence["has_root_level"],
                "failed_traces": [evidence["text"]] if all_error_records else [],  # back-compat only
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
        return {"total_traces_scanned": 0, "failed_traces": [], "total_errors": 0, "error": str(e)}


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
                
                output = [f"Found {len(data)} rows:"]
                for row in data[:20]:
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
            elk_index = elk_indexes.get("elk_log_index")
            
            logger.info(f"[ELKQueryTool] Executing query for app={app}, index={elk_index}...")
            
            result = await execute_elk_query(
                query_json,
                parameter_name=parameter_name,
                parameter_value=parameter_value
            )
            
            if result:
                result = f"[App: {app} | Index: {elk_index}]\n{result}"
            
            return result
        
        except Exception as e:
            logger.error(f"[ELKQueryTool] Error: {e}")
            return f"ELK Query Error: {str(e)}"



def _parse_known_tags_from_warning(warning: str) -> List[str]:
    match = re.search(r"Known tags: \[(.*?)\]", warning)
    if match:
        tags_str = match.group(1)
        tags = re.findall(r"['\"]([^'\"]+)['\"]", tags_str)
        if not tags:
            tags = [t.strip() for t in tags_str.split(',')]
        return [t.strip() for t in tags if t.strip()]
    return []


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
        if service_name and not service:
            service = service_name
        if identifier_name and not tag_name:
            tag_name = identifier_name
        if identifier_value and not tag_value:
            tag_value = identifier_value
        if not app:
            app = "optimus"
            
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
        
        if kwargs.get('hours_ago') and time_range_index == 0:
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
        
        import time as _time
        now_us = int(_time.time() * 1_000_000)
        
        time_ranges = [
            (0, 24), (24, 48), (48, 72), (72, 96), 
            (96, 120), (120, 144), (144, 168)
        ]
        
        time_range_index = max(0, min(6, time_range_index))
        start_offset, end_offset = time_ranges[time_range_index]
        
        start_us = now_us - end_offset * 3600 * 1_000_000
        end_us = now_us - start_offset * 3600 * 1_000_000
        range_label = f"{start_offset}-{end_offset}h ago"

        logger.info(f"[JaegerTraceTool] Searching {range_label} for service={service}, tag={tag_name}={tag_value}")

        result = await _jaeger_fetch(app, service, tag_name, tag_value, start_us, end_us)

        logger.info(
            f"[JaegerTraceTool][RAW RESULT] service={service} tag={tag_name}={tag_value} "
            f"range={range_label} total_scanned={result.get('total_traces_scanned', 0)} "
            f"total_errors={result.get('total_errors', 0)} "
            f"errors_shown_full={result.get('errors_shown_full', 0)} "
            f"errors_header_only={result.get('errors_header_only', 0)} "
            f"has_root_level_error={result.get('has_root_level_error', False)} "
            f"sessions={len(result.get('sessions', []))} "
            f"success_sessions={len(result.get('success_sessions', []))} "
            f"error={result.get('error')}"
        )

        scanned = result.get("total_traces_scanned", 0)

        if scanned and scanned > 0:
            if return_raw_json:
                return json.dumps(result, indent=2, default=str)
            
            output = []
            
            if validation_warning:
                output.append(validation_warning)
                output.append("\n" + "="*60 + "\n")
            
            output.append(f"Found {scanned} traces in time range {range_label}:")
            
            happy_hits = result.get("happy_hits", {})
            sessions = result.get("sessions", [])
            success_sessions = result.get("success_sessions", [])
            
            if happy_hits:
                output.append("\n=== HAPPY PATH (Successful Traces) ===")
                for endpoint, count in happy_hits.items():
                    output.append(f"  {endpoint} → Success: {count}")
            
            if sessions:
                output.append(f"\n=== SESSIONS ===")
                output.append(f"  Total sessions: {len(sessions)}")
                output.append(f"  Successful sessions: {len(success_sessions)}")
            
            if result.get("total_errors", 0) > 0:
                output.append("\n" + result["evidence_text"])
                if result.get("has_root_level_error"):
                    output.append("\n[ROOT-LEVEL FAILURE IDENTIFIED in this time range]")
            else:
                output.append(f"\n[No errors found in {scanned} traces - all successful]")
            
            final_output = "\n".join(output)
            logger.info(
                f"[JaegerTraceTool][OUTPUT] chars={len(final_output)} "
                f"errors={result.get('total_errors', 0)} full={result.get('errors_shown_full', 0)}"
            )
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
        return asyncio.run(self._arun(table_name, app))
    
    async def _arun(self, table_name: str, app: str = "cbs") -> str:
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
        return asyncio.run(self._arun(pattern, app))
    
    async def _arun(self, pattern: str = "%", app: str = "cbs") -> str:
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
        return asyncio.run(self._arun(query, app, top_k))
    
    async def _arun(self, query: str, app: str = "cbs", top_k: int = 5) -> str:
        try:
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
