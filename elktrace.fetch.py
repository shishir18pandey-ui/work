"""Fetch specific Jaeger traces by ID and rank their errors.

This is the deepening half of the flow. Keyword search over the span index
locates *candidate traces* cheaply, but a flat keyword hit only says "this span
matched some text" - it cannot say "this span failed *because* its child timed
out". The parent/child span tree is what distinguishes a root cause from a
symptom that propagated upward, and it is the reason Jaeger is still worth a
call once ELK has narrowed the search.

`GET /traces/{id}` is used here. The existing `new_flow` Jaeger agent never
fetches by ID (its prompts even tell the LLM that isn't possible) because it
only ever *searches* by tag; having trace IDs from ELK is what unlocks it.

Span-tree parsing and error ranking are reused from `new_flow.tools.query_tools`
via read-only imports - no behaviour there is modified.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Sequence

import httpx

from new_flow.tools.app_config import get_jaeger_endpoint, get_jager_auth_token
from new_flow.tools.query_tools import (
    CA_CERT_PATH,
    JAEGER_API_BASE,
    _extract_error_records,
    _jaeger_process_trace,
    _render_evidence,
)
from elk_search_flow.tools.payload_extract import (
    extract_success_payloads,
    render_success_payloads,
)

logger = logging.getLogger(__name__)


def _auth_headers(app: str) -> Dict[str, str]:
    token = get_jager_auth_token(app)
    return {"Authorization": f"Basic {token}"} if token else {}


async def fetch_traces_by_id(
    app: str,
    trace_ids: List[str],
    max_traces: int = 10,
    payload_services: Optional[Sequence[str]] = None,
) -> Dict:
    """Fetch traces by ID and return ranked error evidence.

    Returns the same keys the Jaeger search path produces (`evidence_text`,
    `total_errors`, `has_root_level_error`, ...) so downstream consumers do not
    need to care which path found the data.

    `payload_services` additionally collects the *successful* request/response
    bodies of those services into `payload_text` - the evidence for a "wrong data
    shown" incident, which the error extractor discards by design. Kept in a
    separate key, never merged into `evidence_text`, so a healthy body cannot be
    read as a fault. Omit it to get error evidence only.
    """
    endpoint = get_jaeger_endpoint(app) or JAEGER_API_BASE
    headers = _auth_headers(app)

    ids = list(dict.fromkeys(t.strip() for t in trace_ids if t and t.strip()))[:max_traces]
    if not ids:
        return {
            "total_traces_scanned": 0,
            "evidence_text": "",
            "total_errors": 0,
            "has_root_level_error": False,
            "payload_text": "",
            "payload_spans": 0,
            "note": "no trace IDs supplied",
        }

    logger.info(f"[TraceFetch] app={app} endpoint={endpoint} ids={len(ids)}")

    all_records: List[Dict] = []
    payload_records: List[Dict] = []
    sessions = set()
    fetched = 0
    error_traces = 0

    try:
        async with httpx.AsyncClient(timeout=45, verify=CA_CERT_PATH) as client:

            async def one(tid: str):
                try:
                    resp = await client.get(f"{endpoint}/traces/{tid}", headers=headers)
                    if resp.status_code != 200:
                        logger.warning(f"[TraceFetch] id={tid} status={resp.status_code}")
                        return []
                    return resp.json().get("data", []) or []
                except Exception as exc:
                    logger.warning(f"[TraceFetch] id={tid} failed: {exc}")
                    return []

            results = await asyncio.gather(*[one(t) for t in ids], return_exceptions=True)

        for tid, traces in zip(ids, results):
            if isinstance(traces, BaseException):
                continue
            for trace in traces:
                fetched += 1
                try:
                    _, session_id, _ = _jaeger_process_trace(trace)
                    if session_id:
                        sessions.add(session_id)
                    records = _extract_error_records(trace)
                except Exception as exc:
                    logger.warning(f"[TraceFetch] parse failed id={tid}: {exc}")
                    continue
                if records:
                    error_traces += 1
                    all_records.extend(records)
                if payload_services:
                    # Independent of `records` on purpose: a trace can hold both a
                    # failure elsewhere and a healthy response from the planned
                    # service, and for a wrong-data incident we want both.
                    try:
                        payload_records.extend(
                            extract_success_payloads(trace, payload_services)
                        )
                    except Exception as exc:
                        logger.warning(f"[TraceFetch] payload parse failed id={tid}: {exc}")
                logger.info(
                    f"[TraceFetch] id={tid} errors={len(records)} "
                    f"root_level={sum(1 for r in records if r.get('is_leaf_error'))}"
                )

        evidence = _render_evidence(all_records, fetched)
        payload_text = render_success_payloads(payload_records, payload_services or [])
        logger.info(
            f"[TraceFetch] traces={fetched} error_traces={error_traces} "
            f"distinct_errors={evidence['total_errors']} root_level={evidence['has_root_level']} "
            f"success_payload_spans={len(payload_records)}"
        )

        return {
            "total_traces_scanned": fetched,
            "evidence_text": evidence["text"],
            "payload_text": payload_text,
            "payload_spans": len(payload_records),
            "total_errors": evidence["total_errors"],
            "errors_shown_full": evidence["shown_full"],
            "errors_header_only": evidence["shown_header_only"],
            "has_root_level_error": evidence["has_root_level"],
            "sessions": list(sessions),
            "trace_ids_requested": ids,
        }

    except Exception as exc:
        logger.error(f"[TraceFetch] failed: {exc}", exc_info=True)
        return {
            "total_traces_scanned": 0,
            "evidence_text": "",
            "total_errors": 0,
            "has_root_level_error": False,
            "payload_text": "",
            "payload_spans": 0,
            "error": str(exc),
        }
