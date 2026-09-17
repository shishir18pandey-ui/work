"""Keep the *successful* request/response payloads of the planned service.

`new_flow.tools.query_tools._extract_error_records` deliberately throws these
away: it only retains a `request`/`response` log field when the body carries
error structure (`error_code`, `Exception`, `DENIED`, ...), which is what stops
healthy 200-OK bodies from flooding the evidence for an outage incident.

That filter is correct for "something is failing" and exactly wrong for the
other large class of incident: "the app shows me the wrong data". There the
service returned HTTP 200 with an empty list, a stale balance or a false flag -
no error anywhere, and the response body IS the evidence. Such a trace produces
zero error records, so DEEPEN currently reports "traces exist but no error
spans" and the actual answer is discarded unread.

This module is a *separate* extractor rather than a change to the shared
function: `query_tools` is imported by the old flow, `new_flow` and this one, and
loosening its payload filter would push healthy bodies into every outage
diagnosis those flows produce. Rendered separately from error evidence for the
same reason - the summary agent must not read a healthy response as a fault.

Scoped to the planned service on purpose. A trace can span a dozen services and
their bodies are large; the one the plan agent named is the one whose output the
customer is complaining about.
"""

import json
import logging
import re
from typing import Dict, List, Optional, Sequence

from new_flow.tools.query_tools import _try_decode_base64

logger = logging.getLogger(__name__)

# Rendering caps. A single banking response body runs to several KB and the
# summary agent's context is shared with historic context + the plan; a truncated
# body still shows the shape (empty array, null field, count) which is the part
# that matters.
MAX_SPANS_RENDERED = 5
MAX_BODY_CHARS = 1200
MAX_BODIES_PER_SPAN = 2

# Fields that most often hold the answer for a "wrong data shown" incident.
# Surfaced above the raw body so the useful bit survives truncation.
_INTERESTING_KEYS = (
    "totalRecords", "total_records", "totalCount", "total_count", "count",
    "recordCount", "record_count", "status", "statusCode", "status_code",
    "responseCode", "response_code", "message", "responseMessage",
    "flag", "eligible", "isEligible", "balance", "availableBalance",
)


def _service_of(span: Dict, processes: Dict) -> str:
    pid = span.get("processID", "")
    proc = processes.get(pid) or {}
    return proc.get("serviceName", "") or ""


def _http_status(span: Dict) -> Optional[str]:
    for tag in span.get("tags", []) or []:
        if tag.get("key") in ("http.status_code", "http.response.status_code"):
            return str(tag.get("value"))
    return None


def _is_error_span(span: Dict) -> bool:
    """Mirror of query_tools' notion of an error span, so a span never appears in
    both the error evidence and the healthy-payload block."""
    status = _http_status(span)
    if status and status.startswith(("4", "5")):
        return True
    for tag in span.get("tags", []) or []:
        if tag.get("key") == "error" and tag.get("value") in (True, "true", "True"):
            return True
    for log in span.get("logs", []) or []:
        for field in log.get("fields", []) or []:
            if field.get("key") == "level" and str(field.get("value", "")).lower() in (
                "error", "fatal", "critical"
            ):
                return True
    return False


def _highlights(body: str) -> List[str]:
    """Pull the few fields that usually explain a wrong-data complaint.

    Regex over the raw text rather than a parse: these bodies are frequently
    truncated by the emitting service, double-encoded, or not JSON at all, and a
    failed `json.loads` would drop the whole thing. Where it *is* valid JSON the
    parse is tried first, since it catches nesting the regex misses.
    """
    found: List[str] = []
    try:
        parsed = json.loads(body)
    except Exception:
        parsed = None

    if isinstance(parsed, (dict, list)):
        def walk(node, path=""):
            if len(found) >= 8:
                return
            if isinstance(node, dict):
                for k, v in node.items():
                    here = f"{path}.{k}" if path else k
                    if isinstance(v, (dict, list)):
                        # An empty collection is the single most common cause of
                        # "my data is missing", and carries no scalar to match.
                        if not v:
                            found.append(f"{here}: EMPTY")
                        walk(v, here)
                    elif k in _INTERESTING_KEYS:
                        found.append(f"{here}: {v}")
            elif isinstance(node, list):
                if not node:
                    found.append(f"{path or 'body'}: EMPTY")
                else:
                    walk(node[0], f"{path}[0]")
        walk(parsed)
        return found[:8]

    for key in _INTERESTING_KEYS:
        m = re.search(rf'"{re.escape(key)}"\s*:\s*("[^"]*"|[^,}}\s]+)', body)
        if m:
            found.append(f"{key}: {m.group(1)}")
        if len(found) >= 8:
            break
    return found


def extract_success_payloads(trace: Dict, services: Sequence[str]) -> List[Dict]:
    """Return request/response bodies from non-error spans of `services`.

    Returns [] when `services` is empty - an unscoped sweep would pull every
    body in the trace, which is the flooding this module exists to avoid.
    """
    if not services:
        return []
    wanted = {s.lower().strip() for s in services if s and s.strip()}
    if not wanted:
        return []

    processes = trace.get("processes", {}) or {}
    trace_id = trace.get("traceID", "")
    records: List[Dict] = []

    for span in trace.get("spans", []) or []:
        service = _service_of(span, processes)
        if service.lower() not in wanted:
            continue
        if _is_error_span(span):
            continue  # already covered, better, by the error evidence

        bodies: List[Dict] = []
        for log in span.get("logs", []) or []:
            fields = {x.get("key"): x.get("value") for x in log.get("fields", []) or []}
            for kind in ("request", "response"):
                raw = fields.get(kind)
                if not raw:
                    continue
                body = _try_decode_base64(str(raw))
                bodies.append({
                    "kind": kind,
                    "body": body,
                    "highlights": _highlights(body),
                })
        if not bodies:
            continue

        records.append({
            "trace_id": trace_id,
            "service": service,
            "operation": span.get("operationName", ""),
            "http_status": _http_status(span) or "200",
            "duration_us": span.get("duration", 0),
            "bodies": bodies,
        })

    return records


def render_success_payloads(records: List[Dict], services: Sequence[str]) -> str:
    """Render the healthy-payload block.

    The header states plainly that these calls SUCCEEDED. The summary agent is
    prompted to quote error codes verbatim, so an unlabelled body full of status
    fields is a standing invitation to report a healthy response as the fault.
    """
    if not records:
        return ""

    named = " / ".join(s for s in services if s) or "the planned service"
    lines = [
        f"SUCCESSFUL RESPONSES FROM {named.upper()} "
        f"({len(records)} span{'s' if len(records) != 1 else ''})",
        "These calls completed without error. The fault, if any, is in WHAT was "
        "returned - an empty list, a stale value, a flag set the wrong way - not "
        "in the call failing.",
        "",
    ]

    # Prefer spans whose body already looks explanatory, then the slowest, since
    # both beat trace order when only a handful can be shown.
    ordered = sorted(
        records,
        key=lambda r: (
            -sum(len(b["highlights"]) for b in r["bodies"]),
            -int(r.get("duration_us") or 0),
        ),
    )

    for rec in ordered[:MAX_SPANS_RENDERED]:
        lines.append(
            f"- {rec['service']} {rec['operation']} "
            f"[HTTP {rec['http_status']}] trace={rec['trace_id'][:16]}"
        )
        for body in rec["bodies"][:MAX_BODIES_PER_SPAN]:
            if body["highlights"]:
                lines.append(f"    {body['kind']} fields: " + ", ".join(body["highlights"]))
            text = body["body"]
            if len(text) > MAX_BODY_CHARS:
                text = text[:MAX_BODY_CHARS] + f" ...[truncated, {len(body['body'])} chars]"
            lines.append(f"    {body['kind']}: {text}")
        lines.append("")

    hidden = len(ordered) - MAX_SPANS_RENDERED
    if hidden > 0:
        lines.append(f"({hidden} further successful span(s) not shown)")

    return "\n".join(lines).strip()
