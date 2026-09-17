"""ELK-first investigation: locate by keyword, deepen by trace, fall back to scan.

Contrast with the existing `new_flow` executor
---------------------------------------------
`new_flow/agents/execute_agent_jaeger.py` walks seven fixed 24h windows calling
`GET /traces?service=&tags=` until something matches. Every call needs a correct
service name *and* a correct identifier, because Jaeger's API only filters by
exact tag equality - so an incident whose only distinguishing feature is its
wording is effectively unfindable.

Here the wide, cheap step is a keyword query against the span index in ES:

  1. LOCATE   keyword-search spans -> candidate trace IDs (ES relevance order)
  2. DEEPEN   fetch those traces by ID -> parent/child tree -> ranked evidence
  3. FALLBACK only if LOCATE is dry and we hold an identifier, run the old
              window scan, which remains the better tool in exactly that case

Nothing in `new_flow` is modified; its Jaeger agent is imported read-only for
stage 3 so the two flows cannot drift apart on that behaviour.
"""

import logging
import os
import re
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from new_flow.tools.app_config import AppConfig, get_app_config_safe
from elk_search_flow.tools.service_discovery import format_discovered, service_names
from elk_search_flow.tools.span_search import (
    SpanSearcher,
    describe_semantic_failures,
    hit_services,
    pinnable_tags,
    render_activity,
    span_index_patterns,
    summarise_hits,
)
from elk_search_flow.tools.trace_fetch import fetch_traces_by_id

logger = logging.getLogger(__name__)

# Confidence tiers. Driven by what the span tree shows, not by counting keywords:
# a confirmed root-level failure is the strongest signal available.
CONF_ROOT_LEVEL = 0.85
CONF_MANY_ERRORS = 0.70
CONF_SOME_ERRORS = 0.60
CONF_TRACES_NO_ERRORS = 0.35
# A span whose *outcome* tag is negative (`login_status=unsuccessful`) on an
# otherwise HTTP-200 request. Above the 0.30 line because it is direct evidence
# of what happened to this customer, not a keyword coincidence - but below the
# error tiers because nothing in the system actually malfunctioned, so there is
# usually no fix to apply, only an explanation to give.
CONF_SEMANTIC_FAILURE = 0.55
# Below the 0.30 escalation line on purpose: a text match proves a span matched
# some words, not that we found the cause.
CONF_KEYWORD_ONLY = 0.25
CONF_NOTHING = 0.10
# The planned service handled this customer's requests without error, or was
# never called at all. Both are real, checkable observations about the right
# service - stronger than a keyword coincidence, and they rule out the thing the
# plan agent expected to be broken. Deliberately just under CONF_SOME_ERRORS
# (0.60), the `resolved` threshold: an L2 engineer should read it, because the
# next step is "what did the response contain", which needs the payload rather
# than the span.
CONF_NO_FAULT_FOUND = 0.45
# Nothing errored, but we hold the planned service's actual HTTP-200 response for
# this customer. That is the evidence for a "wrong data shown" incident, and it is
# specific to this customer's request rather than a keyword coincidence - so above
# the probe alone. Still not `resolved`: reading whether an empty list or a stale
# field is *wrong* needs someone who knows the product.
CONF_SUCCESS_PAYLOAD = 0.50


class InvestigationResult(BaseModel):
    """Field-compatible with new_flow's JaegerExecutionResult, plus provenance."""

    resolved: bool = False
    diagnosis: str = ""
    solution: str = ""
    questions: List[str] = Field(default_factory=list)
    tool_calls: List[Dict] = Field(default_factory=list)
    final_state: str = ""
    confidence: float = 0.0
    iterations_completed: int = 0
    # provenance
    evidence_text: str = ""
    trace_ids: List[str] = Field(default_factory=list)
    search_stage: str = ""
    # Non-empty when the span search never ran, vs ran and found nothing.
    locate_skipped_reason: str = ""
    escalation_reason: str = ""
    # Activity-probe outcome on the planned service: "" (not run) / "healthy" /
    # "never_called". Records that the expected service was actually checked,
    # which is otherwise invisible when the evidence came from somewhere else.
    planned_service_state: str = ""


def build_search_text(incident_description: str, problem_category: str, plan_output=None) -> str:
    """Free text is what ELK adds over Jaeger, so prefer the most technical framing."""
    parts: List[str] = []
    for candidate in (
        getattr(plan_output, "issue_summary", "") if plan_output else "",
        problem_category or "",
        incident_description or "",
    ):
        text = str(candidate).strip()
        if text:
            parts.append(text)
    return " ".join(parts)[:1000]


def _searcher_for(config: AppConfig, app: str) -> Tuple[Optional[SpanSearcher], str]:
    """Returns `(searcher, skip_reason)`; the reason lands on the span, because a
    zero-hit search and a misconfigured one look identical otherwise."""
    endpoints = _span_endpoints(config, app)
    if not endpoints:
        reason = f"no span ES endpoint configured for app={app} (set span_es_endpoint in app_config)"
        logger.warning(f"[Investigate] {reason}")
        return None, reason

    auth_env = _span_auth_env(config)
    auth_token = os.getenv(auth_env) if auth_env else None
    if not auth_token:
        reason = f"no span ES credential in env {auth_env!r} for app={app}"
        logger.warning(f"[Investigate] {reason}")
        return None, reason

    patterns = span_index_patterns(app, _configured_span_index(config, app))
    logger.info(f"[Investigate] app={app} span_endpoints={endpoints} span_indices={patterns}")
    return SpanSearcher(endpoints=endpoints, auth_header=auth_token, index_patterns=patterns), ""


def _span_endpoints(config: AppConfig, app: str) -> List[str]:
    """ES cluster(s) holding the *span* index.

    Deliberately does NOT fall back to `elk_endpoint` - that is the *log* cluster
    (`execute_agent_elk`), and on optimus prod it is a different cluster with a
    different credential: the span index matches 0 indices there and the span
    credential 401s. That fallback silently searched the wrong cluster.
    """
    slug = re.sub(r"[^A-Z0-9]+", "_", (app or "").upper()).strip("_")
    override = os.getenv(f"SPAN_ES_URL_{slug}") or os.getenv("SPAN_ES_URL")
    if override:
        logger.info(f"[Investigate] span ES endpoint overridden by env for app={app}")
    source = override or str(config.span_es_endpoint or "")

    endpoints: List[str] = []
    for raw in source.split(";"):
        url = raw.strip()
        if not url:
            continue
        # Tolerate a pasted search URL (".../prod-jaeger-span-*/_search"): the ES
        # client needs the bare host, and the index is passed per query.
        match = re.match(r"(https?://[^/]+)", url)
        endpoints.append(match.group(1) if match else url)
    return endpoints


def _span_auth_env(config: AppConfig) -> str:
    """Env var naming the span cluster credential. Falls back to `elk_auth_key`
    because most apps do share one across both clusters; optimus does not."""
    return str(config.span_es_auth_key or config.elk_auth_key or "")


def _configured_span_index(config: AppConfig, app: str) -> Optional[str]:
    """Span index pattern: env override, then app_config, then service_metadata."""
    slug = re.sub(r"[^A-Z0-9]+", "_", (app or "").upper()).strip("_")
    override = os.getenv(f"SPAN_ES_INDEX_{slug}") or os.getenv("SPAN_ES_INDEX")
    if override and override.strip():
        return override.strip()
    if config.span_es_index and str(config.span_es_index).strip():
        return str(config.span_es_index).strip()
    try:
        from new_flow.tools.service_metadata import get_elk_indexes

        return (get_elk_indexes(app) or {}).get("elk_trace_index")
    except Exception as exc:
        logger.warning(f"[Investigate] span index lookup failed app={app}: {exc}")
        return None


async def discover_services_for_app(app: str) -> Tuple[str, str]:
    """Services the plan agent may choose from, plus which source answered.

    Ordered by how much each source can be trusted to match what a
    `serviceName` filter will actually hit:

      1. **the span index** - authoritative by construction. A `terms` agg on
         `serviceName` over the same cluster/index/field LOCATE filters on, so
         the names cannot disagree with what a query can match.
      2. **Jaeger's /services** - authoritative in principle, but
         `get_jaeger_endpoint` uses only the first of the semicolon-separated
         endpoints, which for optimus is a UAT host while the spans are on prod.
      3. **`service_metadata.yaml`** - static, hand-written, but service-specific
         (real tags and purposes per service) and for optimus it lists 23
         services that look far more plausible than app_config's four.
      4. **`app_config.<env>.json`** - last resort. For optimus this is
         `optimus-api`/`-login`/`-web`/`-mobile`, which shares no name with (3)
         and looks like a placeholder.

    Returned as a `- <name>` block because `plan_agents._parse_plan_output`
    validates the LLM's pick against exactly those lines - so the LLM can only
    choose a name that came from one of these sources, and LOCATE filters on the
    same list via `service_names`.
    """
    app_key = (app or "").lower().strip()
    config = get_app_config_safe(app_key)

    searcher, _skip = _searcher_for(config, app_key)
    if searcher is not None:
        try:
            discovered = await searcher.aggregate_services()
            if discovered:
                logger.info(
                    f"[Discovery] {len(discovered)} services from span index app={app_key}"
                )
                return format_discovered(discovered, "the span index"), "span_index"
            logger.warning(f"[Discovery] span index returned no services app={app_key}")
        except Exception as exc:
            logger.warning(f"[Discovery] span-index discovery failed app={app_key}: {exc}")

    try:
        from new_flow.tools.discovery_tools import discover_jaeger_services_impl

        text = await discover_jaeger_services_impl(app_key)
        # This helper reports failure as prose ("Error connecting to Jaeger…",
        # "No services found…"), so presence of parsed names is the only
        # reliable success test.
        if service_names(text):
            return text, "jaeger_api"
        logger.warning(f"[Discovery] Jaeger /services yielded nothing app={app_key}: {text[:120]}")
    except Exception as exc:
        logger.warning(f"[Discovery] Jaeger discovery failed app={app_key}: {exc}")

    try:
        from new_flow.tools.service_metadata import get_services

        metadata_services = list(get_services(app_key) or [])
        if metadata_services:
            return (
                format_discovered([(s, 0) for s in metadata_services], "service_metadata.yaml"),
                "service_metadata",
            )
    except Exception as exc:
        logger.warning(f"[Discovery] service_metadata lookup failed app={app_key}: {exc}")

    configured = list(getattr(config, "services", []) or [])
    if configured:
        logger.warning(
            f"[Discovery] falling back to app_config services for app={app_key} - "
            f"these are unverified against the span index"
        )
        return (
            format_discovered([(s, 0) for s in configured], "app_config (UNVERIFIED)"),
            "app_config",
        )

    return "No services discovered", "none"


def discovered_service_names(discovered_text: str) -> List[str]:
    """Re-exported so the flow can count services without importing the tool."""
    return service_names(discovered_text)


def _clean_services(values) -> List[str]:
    names: List[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text.upper() != "NONE" and text not in names:
            names.append(text)
    return names


def candidate_services(
    config: AppConfig, plan_output, discovered_services: str = ""
) -> Tuple[List[str], List[str]]:
    """(primary, fallback) service filters for the span search.

    The span index pattern is cluster-wide (`prod-jaeger-span-*`), so the service
    name is the strongest narrowing available. Two tiers, because they fail
    differently:

      primary  - the service the LLM picked out of the discovered list, already
                 validated against it by `_parse_plan_output`. Right most of the
                 time and by far the cheapest.
      fallback - every *discovered* service. Covers a wrong pick while staying
                 inside names the index is known to hold.

    **Both tiers come from the discovered list only.** `app_config`'s `services`
    are deliberately NOT used: for optimus they are `optimus-api`/`-login`/
    `-web`/`-mobile`, which share no name with the 23 services
    `service_metadata.yaml` lists for the same app, and nothing has ever verified
    them against the span index. Filtering on a name the index has never seen
    returns zero hits *silently*. `discover_services_for_app` already falls back
    to app_config as its last resort, so those names still reach us when they are
    genuinely all we have - but then they arrive labelled and via the LLM's pick,
    rather than being AND-ed in unconditionally.

    `default_jaeger_service` is included only if discovery corroborates it, for
    the same reason.

    `SpanSearcher._attempts` drops the filter entirely after both tiers, so an
    entirely wrong list degrades to an unfiltered search rather than a silent
    zero-hit answer.
    """
    discovered = _clean_services(service_names(discovered_services))

    chosen = _clean_services([
        getattr(plan_output, "suggested_service", "") if plan_output else "",
    ])
    # The app default is a hint, not a fact - only trust it if the index (or
    # whichever source answered discovery) actually reported it.
    default = _clean_services([config.default_jaeger_service])
    corroborated = [s for s in default if not discovered or s in discovered]

    primary = _clean_services(chosen + corroborated)
    return primary, discovered


def _confidence(jaeger_result: Dict) -> float:
    if jaeger_result.get("has_root_level_error"):
        return CONF_ROOT_LEVEL
    total = int(jaeger_result.get("total_errors") or 0)
    if total >= 3:
        return CONF_MANY_ERRORS
    if total >= 1:
        return CONF_SOME_ERRORS
    if int(jaeger_result.get("total_traces_scanned") or 0) > 0:
        return CONF_TRACES_NO_ERRORS
    return CONF_NOTHING


async def run_investigation_async(
    plan_output,
    incident_description: str,
    app: str,
    customer_identifiers: Dict[str, str],
    problem_category: str,
    incident_id: str = "unknown",
    discovered_services: str = "",
    max_iterations: int = 5,
) -> InvestigationResult:
    app_key = (app or "").lower().strip()
    config = get_app_config_safe(app_key)
    tool_calls: List[Dict] = []

    search_text = build_search_text(incident_description, problem_category, plan_output)
    tags = pinnable_tags(customer_identifiers)
    services, fallback_services = candidate_services(config, plan_output, discovered_services)
    logger.info(
        f"[Investigate] start incident={incident_id} app={app_key} "
        f"pinned_tags={list(tags)} services={services} "
        f"fallback_services={len(fallback_services)} text_len={len(search_text)}"
    )

    searcher, skip_reason = _searcher_for(config, app_key)

    # Catch a wrong-cluster setting before issuing ~20 searches that cannot match.
    if searcher is not None:
        present, detail = await searcher.index_present()
        if not present:
            skip_reason = f"span index absent on configured cluster: {detail}"
            logger.error(f"[Investigate] {skip_reason} - skipping LOCATE for app={app_key}")
            searcher = None
        tool_calls.append({
            "stage": "locate_precheck",
            "tool": "span_index_present",
            "present": present,
            "detail": detail,
        })

    # ── Stage 1: LOCATE ───────────────────────────────────────────────────
    located: Dict = {"status": "unconfigured", "hits": [], "trace_ids": []}
    if searcher is not None:
        # Pivot-aware: if the quoted identifier only reaches healthy spans, the
        # search harvests the other identifiers off those spans and re-queries.
        # See SpanSearcher.search_with_pivot - this is what finds the failure when
        # the incident quotes a mobile but the failing service tags customer_id.
        located = await searcher.search_with_pivot(
            search_text, tags, services=services, fallback_services=fallback_services
        )
        tool_calls.append({
            "stage": "locate",
            "tool": "span_keyword_search",
            "status": located.get("status"),
            "hits": len(located.get("hits") or []),
            "trace_ids": located.get("trace_ids", []),
            "time_bucket": located.get("time_bucket", ""),
            "filter_scope": located.get("filter_scope", ""),
            "pivoted": located.get("pivoted", False),
            "pivot_tags": list(located.get("pivot_tags") or {}),
            "pinned_tags": list(tags),
        })

        # A pinned tag that isn't mapped on the span index returns zero hits
        # *silently*. Retry on text alone before concluding there is nothing.
        # The service + error filters stay on, so this stays bounded.
        if not located.get("trace_ids") and tags:
            logger.info("[Investigate] no hits with pinned tags - retrying text-only")
            located = await searcher.search(
                search_text, {}, services=services, fallback_services=fallback_services
            )
            tool_calls.append({
                "stage": "locate_untagged",
                "tool": "span_keyword_search",
                "status": located.get("status"),
                "hits": len(located.get("hits") or []),
                "trace_ids": located.get("trace_ids", []),
                "time_bucket": located.get("time_bucket", ""),
                "filter_scope": located.get("filter_scope", ""),
                # Explicit: this step ran with no identifier at all, so the result
                # rests on text ranking alone.
                "pinned_tags": [],
            })

    # ── Stage 1b: was the PLANNED service actually at fault? ──────────────
    # The error filter sits in `filter` context for every `_attempts` step that
    # still has a service filter, so "no hits on deposits-api" means "no FAILING
    # span there" - it cannot distinguish a service that worked from one that was
    # never called. Those have opposite resolutions, and for the whole
    # "data not showing" class of incident the healthy call IS the evidence.
    #
    # Runs whenever no failure was found ON the planned service - including when
    # widening then found one somewhere else, which is exactly the case that
    # misled INC000008754817 (deposits-api clean, an unrelated offers 404 became
    # "the FD data is not synchronised"). Additive: this never replaces a located
    # failure, it only records what the expected service was doing.
    activity: Dict = {}
    planned_service_state = ""
    if searcher is not None and tags and services:
        located_services = hit_services(located.get("hits") or [])
        planned_hit = {s.lower() for s in located_services} & {s.lower() for s in services}
        if not planned_hit:
            logger.info(
                f"[Investigate] no failure on planned service {services} "
                f"(hits were on {located_services or 'nothing'}) - probing activity"
            )
            activity = await searcher.search_activity(tags, services)
            planned_service_state = {
                "success": "healthy",
                "no_activity": "never_called",
            }.get(activity.get("status", ""), "")
            tool_calls.append({
                "stage": "locate_activity",
                "tool": "span_activity_probe",
                "status": activity.get("status"),
                "services": services,
                "hits": len(activity.get("hits") or []),
                "operations": [
                    f"{r['service']} {r['operation']} x{r['count']}"
                    for r in (activity.get("operations") or [])
                ],
            })

    activity_text = render_activity(activity, services) if planned_service_state else ""

    def _with_activity(text: str) -> str:
        """Probe text goes FIRST - it rules out the expected cause, so the summary
        agent must read it before whatever error was found elsewhere. Without the
        ordering the off-service error reads as the answer again."""
        if not activity_text:
            return text
        return f"{activity_text}\n\n{text}" if text else activity_text

    def _no_fault_result(weaker: str = "") -> InvestigationResult:
        """The probe is the strongest thing we have: no failure anywhere for this
        customer, but a checked observation about the service that should have
        failed. `weaker` is whatever inconclusive material we'd otherwise have
        returned - kept, but ranked below the probe."""
        text = _with_activity(weaker)
        solution = (
            "No service error was recorded for this customer. The planned service "
            + (
                "handled the requests successfully, so check WHAT it returned "
                "(empty list, stale value, wrong flag) rather than whether it failed."
                if planned_service_state == "healthy"
                else "was never called for this customer, so the request did not "
                "reach it - check the caller/upstream routing or whether the "
                "customer performed the action at all."
            )
        )
        return InvestigationResult(
            resolved=False,
            diagnosis=text,
            solution=solution,
            tool_calls=tool_calls,
            final_state=(
                f"No failure found; planned service {planned_service_state}"
            ),
            confidence=CONF_NO_FAULT_FOUND,
            iterations_completed=1,
            evidence_text=text,
            trace_ids=trace_ids,
            search_stage="no_fault_on_planned_service",
            planned_service_state=planned_service_state,
            escalation_reason=(
                f"No failing span for this customer; planned service "
                f"{' / '.join(services)} was {planned_service_state.replace('_', ' ')} "
                f"- needs a payload-level or business-logic check"
            ),
        )

    trace_ids = located.get("trace_ids") or []

    # An ES transport failure returns [] exactly like a genuine miss, so without
    # this a timed-out sweep reads downstream as "this incident never happened".
    if not trace_ids and searcher is not None and searcher.last_error:
        skip_reason = f"span search failed: {searcher.last_error}"
        logger.error(f"[Investigate] {skip_reason}")
        tool_calls.append({"stage": "locate_error", "tool": "span_keyword_search",
                           "error": searcher.last_error})

    # ── Stage 2: DEEPEN ───────────────────────────────────────────────────
    if trace_ids:
        # `payload_services` additionally keeps the SUCCESSFUL bodies of the
        # planned service. For a "wrong data shown" incident nothing errored, so
        # the response body is the only evidence there is - the error extractor
        # drops it by design. Scoped to the planned service to bound the volume.
        traced = await fetch_traces_by_id(
            app_key, trace_ids, payload_services=services or fallback_services[:1]
        )
        tool_calls.append({
            "stage": "deepen",
            "tool": "jaeger_fetch_by_trace_id",
            "traces_scanned": traced.get("total_traces_scanned", 0),
            "total_errors": traced.get("total_errors", 0),
            "has_root_level_error": traced.get("has_root_level_error", False),
            "success_payload_spans": traced.get("payload_spans", 0),
        })

        evidence = traced.get("evidence_text") or ""
        payload_text = traced.get("payload_text") or ""
        if evidence:
            confidence = _confidence(traced)
            logger.info(
                f"[Investigate] resolved from span tree incident={incident_id} "
                f"confidence={confidence} errors={traced.get('total_errors')}"
            )
            return InvestigationResult(
                resolved=confidence >= CONF_SOME_ERRORS,
                diagnosis=_with_activity(evidence),
                solution=(
                    "Action the ranked root-level failure above on the failing "
                    "dependency. Note the failure is NOT on the service expected "
                    "for this symptom - confirm the two are actually related "
                    "before acting."
                    if planned_service_state == "healthy"
                    else "Action the ranked root-level failure above on the failing dependency."
                ),
                tool_calls=tool_calls,
                final_state="Span tree located by keyword search",
                confidence=confidence,
                iterations_completed=1,
                evidence_text=_with_activity(evidence),
                trace_ids=trace_ids,
                search_stage="span_tree",
                planned_service_state=planned_service_state,
            )

        # Traces exist but Jaeger found no *error* spans in them. That is not the
        # same as "nothing happened": the span index may have matched a negative
        # business outcome (a wrong MPIN, a declined payment), which carries no
        # error field at all and so is invisible to the error-tree ranking.
        # Distinguish the two, because one is real evidence and one is a keyword
        # coincidence.
        outcome = describe_semantic_failures(
            located.get("hits") or [], search_text=search_text
        )
        if outcome:
            logger.info(
                f"[Investigate] semantic failure (no technical error) "
                f"incident={incident_id} pivoted={located.get('pivoted', False)}"
            )
            return InvestigationResult(
                resolved=True,
                diagnosis=_with_activity(outcome),
                solution=(
                    "The request completed normally but the outcome was negative for "
                    "the customer - explain the outcome rather than looking for a "
                    "system fault. No service error was recorded."
                ),
                tool_calls=tool_calls,
                final_state="Negative outcome confirmed on the span; no system error",
                confidence=CONF_SEMANTIC_FAILURE,
                iterations_completed=1,
                evidence_text=_with_activity(outcome),
                trace_ids=trace_ids,
                search_stage="semantic_failure",
                planned_service_state=planned_service_state,
            )

        # No error and no negative outcome tag, but we do hold what the planned
        # service actually returned. Ranked above the probe and the keyword
        # summary because it is this customer's real response body.
        if payload_text:
            logger.info(
                f"[Investigate] no error; returning planned-service response bodies "
                f"incident={incident_id} spans={traced.get('payload_spans', 0)}"
            )
            text = _with_activity(payload_text)
            return InvestigationResult(
                resolved=False,
                diagnosis=text,
                solution=(
                    "No service error occurred. Compare the response above against "
                    "what the customer expected to see - an empty collection, a "
                    "stale value or a wrongly-set flag is the likely cause, which "
                    "is a data or business-logic issue rather than an outage."
                ),
                tool_calls=tool_calls,
                final_state="No errors in the located traces; planned service returned HTTP 200",
                confidence=CONF_SUCCESS_PAYLOAD,
                iterations_completed=1,
                evidence_text=text,
                trace_ids=trace_ids,
                search_stage="success_payload",
                planned_service_state=planned_service_state,
                escalation_reason=(
                    "No failure found; the planned service responded successfully - "
                    "needs a check of the returned data against expectations"
                ),
            )

        # Genuinely only keyword hits. Surface them rather than discarding, but
        # keep the confidence below the escalation line.
        hit_summary = summarise_hits(located.get("hits") or [])
        # The probe outranks bare keyword hits: "the right service was fine" is a
        # checked fact about the right service, a text match is a coincidence.
        if planned_service_state:
            logger.info(
                f"[Investigate] no fault on planned service ({planned_service_state}) "
                f"incident={incident_id}"
            )
            return _no_fault_result(hit_summary)
        if hit_summary:
            logger.info(f"[Investigate] keyword hits only incident={incident_id}")
            return InvestigationResult(
                resolved=False,
                diagnosis=hit_summary,
                solution=(
                    "Matching log lines found but no failing span - likely a functional "
                    "issue rather than an outage."
                ),
                tool_calls=tool_calls,
                final_state="Keyword hits only; located traces had no error spans",
                confidence=CONF_KEYWORD_ONLY,
                iterations_completed=1,
                evidence_text=hit_summary,
                trace_ids=trace_ids,
                search_stage="keyword_only",
                escalation_reason="Keyword matches found but no failing span could be confirmed",
            )

    # ── Stage 3: FALLBACK to the tag-based window scan ────────────────────
    if tags:
        if skip_reason:
            # LOCATE never ran, so the fallback's answer is the only evidence.
            logger.error(
                f"[Investigate] LOCATE was SKIPPED ({skip_reason}) - falling back to "
                f"the Jaeger window scan alone incident={incident_id}"
            )
        logger.info(f"[Investigate] keyword search dry - Jaeger window scan incident={incident_id}")
        from new_flow.agents.execute_agent_jaeger import run_jaeger_only_async

        fallback = await run_jaeger_only_async(
            plan_output=plan_output,
            incident_description=incident_description,
            app=app_key,
            customer_identifiers=customer_identifiers,
            problem_category=problem_category,
            max_iterations=max_iterations,
            incident_id=incident_id,
            discovered_services=discovered_services,
        )
        tool_calls.append({
            "stage": "fallback",
            "tool": "jaeger_window_scan",
            "confidence": getattr(fallback, "confidence", 0.0),
        })
        diagnosis = getattr(fallback, "diagnosis", "") or ""
        fallback_confidence = float(getattr(fallback, "confidence", 0.0) or 0.0)
        # The window scan pins `customer_id` on one service across seven 24h
        # windows; coming back under the probe's tier means it found no failure
        # either, so the probe is the better answer and the scan text becomes
        # supporting detail.
        if planned_service_state and fallback_confidence < CONF_NO_FAULT_FOUND:
            logger.info(
                f"[Investigate] window scan inconclusive ({fallback_confidence}) - "
                f"reporting planned service {planned_service_state} incident={incident_id}"
            )
            result = _no_fault_result(diagnosis)
            result.tool_calls = tool_calls + list(getattr(fallback, "tool_calls", []) or [])
            result.locate_skipped_reason = skip_reason
            return result
        return InvestigationResult(
            resolved=bool(getattr(fallback, "resolved", False)),
            diagnosis=_with_activity(diagnosis),
            solution=getattr(fallback, "solution", "") or "",
            questions=list(getattr(fallback, "questions", []) or []),
            tool_calls=tool_calls + list(getattr(fallback, "tool_calls", []) or []),
            final_state=f"Keyword search dry; {getattr(fallback, 'final_state', '')}",
            confidence=fallback_confidence,
            iterations_completed=int(getattr(fallback, "iterations_completed", 0) or 0),
            evidence_text=_with_activity(diagnosis),
            search_stage="jaeger_fallback",
            planned_service_state=planned_service_state,
            locate_skipped_reason=skip_reason,
            escalation_reason=(
                f"LOCATE unavailable ({skip_reason}); diagnosis rests on the Jaeger scan alone"
                if skip_reason
                else "Neither keyword search nor trace scan explained the issue"
            ),
        )

    reason = (
        f"No ELK/Jaeger configuration for app '{app_key}'"
        if located.get("status") == "unconfigured"
        else "No spans matched by keyword search, and no identifier available for a trace scan"
    )
    if skip_reason:
        reason = f"{reason} (LOCATE skipped: {skip_reason})"
    logger.info(f"[Investigate] exhausted incident={incident_id}: {reason}")
    return InvestigationResult(
        resolved=False,
        diagnosis=reason,
        solution="Assign to an engineer for manual investigation.",
        tool_calls=tool_calls,
        final_state=reason,
        confidence=CONF_NOTHING,
        iterations_completed=1,
        search_stage="exhausted",
        locate_skipped_reason=skip_reason,
        escalation_reason=reason,
    )
