"""Discover service names from the span index itself.

Why this exists
---------------
LOCATE filters on `serviceName`, and that filter sits in `filter` context - a
hard AND. A name that doesn't exist in the index returns zero hits *silently*,
which is indistinguishable from "this incident never happened". So the list of
services has to be a fact about the data, not a guess.

None of the three pre-existing sources is that fact:

  * `app_config.<env>.json`'s `services` - for optimus this is
    `optimus-api`/`-login`/`-web`/`-mobile`, which shares not one name with the
    23 services `service_metadata.yaml` lists for the same app. Only 3 of ~25
    apps have the key populated at all. It reads as a placeholder.
  * `service_metadata.yaml` - hand-written, richer (per-service tags and
    purposes) and probably closer to reality, but still a static file that can
    drift from what is deployed.
  * `new_flow.tools.discovery_tools.discover_jaeger_services_impl` - authoritative
    in principle (`GET /services` off live Jaeger), but it reads
    `get_jaeger_endpoint(app)`, which takes only the FIRST semicolon-separated
    entry. For optimus that is the *UAT* tracing host while the spans we search
    live on the *prod* cluster. On any error it returns the human-readable string
    "Service discovery unavailable", which a naive parser reads as zero services.

The span index can answer authoritatively and cheaply: `serviceName` is
`keyword`-mapped, so a `terms` aggregation is served from doc values with
`size: 0` - no documents fetched, scored, or returned. That is the same index,
the same cluster and the same field LOCATE will filter on, so agreement is
guaranteed by construction.

Output is formatted as `- <name>` lines, matching what
`new_flow.agents.plan_agents._parse_plan_output` already parses and validates the
LLM's choice against. So the LLM picks from real names and a hallucinated or
stale pick is rejected by machinery that already exists.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# How far back to look for active services. Long enough to survive a quiet
# period on a low-traffic service, short enough to stay cheap.
DISCOVERY_LOOKBACK_HOURS = 24

# Upper bound on distinct service names returned. The bank runs a few hundred
# services; 500 is generous and caps the aggregation's memory.
MAX_SERVICES = 500

# Aggregating over ~299 indices is metadata-cheap but not free.
DISCOVERY_TIMEOUT_SECONDS = 30


def build_service_agg_query(
    lookback_hours: int = DISCOVERY_LOOKBACK_HOURS,
    time_field: str = "startTimeMillis",
    now: Optional[datetime] = None,
) -> Dict:
    """Aggregation-only query listing distinct service names.

    `size: 0` means no hits are fetched, scored or returned - only the
    aggregation. Both service fields are aggregated because which one the
    collector populates varies, and `serviceName` is top-level *in addition* to
    the standard `process.serviceName`.
    """
    now = now or datetime.now()
    start = now - timedelta(hours=lookback_hours)
    scale = 1000 if time_field == "startTimeMillis" else 1_000_000
    if time_field in ("startTimeMillis", "startTime"):
        time_range = {"gte": int(start.timestamp() * scale), "lte": int(now.timestamp() * scale)}
    else:
        time_range = {"gte": start.isoformat(), "lte": now.isoformat()}

    return {
        "size": 0,
        "track_total_hits": False,
        "timeout": f"{DISCOVERY_TIMEOUT_SECONDS}s",
        "query": {"range": {time_field: time_range}},
        "aggs": {
            "services": {"terms": {"field": "serviceName", "size": MAX_SERVICES}},
            "process_services": {
                "terms": {"field": "process.serviceName", "size": MAX_SERVICES}
            },
        },
    }


def parse_service_agg(response: Dict) -> List[Tuple[str, int]]:
    """(service_name, span_count) pairs, busiest first, merged across both fields."""
    counts: Dict[str, int] = {}
    aggs = (response or {}).get("aggregations") or {}
    for key in ("services", "process_services"):
        for bucket in (aggs.get(key) or {}).get("buckets") or []:
            name = str(bucket.get("key") or "").strip()
            if not name:
                continue
            # Same service seen via both fields: keep the larger count rather
            # than summing, which would double-count the same spans.
            counts[name] = max(counts.get(name, 0), int(bucket.get("doc_count") or 0))
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))


def format_discovered(services: List[Tuple[str, int]], source: str) -> str:
    """Render for the plan agent's prompt.

    The `- <name>` shape is required: `_parse_plan_output` builds its set of
    valid services from lines starting with "- " and rejects any LLM choice
    outside it. Span counts are included because they help the LLM tell a
    high-traffic API from an incidental one.
    """
    if not services:
        return "No services discovered"

    lines = [f"Discovered {len(services)} services from {source}:", ""]
    lines.extend(f"  - {name}" for name, _ in services)
    lines.extend([
        "",
        "Span volume over the discovery window (busiest first):",
    ])
    lines.extend(f"  {name}: {count:,} spans" for name, count in services[:15])
    return "\n".join(lines)


def service_names(discovered_text: str) -> List[str]:
    """Names back out of the formatted block.

    Same parse the plan agent uses, so what LOCATE filters on and what the LLM
    was allowed to choose from cannot disagree.
    """
    names: List[str] = []
    for line in (discovered_text or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            name = stripped[2:].strip()
            if name and name not in names:
                names.append(name)
    return names


async def discover_services_from_spans(
    searcher,
    lookback_hours: int = DISCOVERY_LOOKBACK_HOURS,
) -> List[Tuple[str, int]]:
    """Ask the span index which services it actually holds.

    Runs against the same clusters/index patterns `SpanSearcher` searches, so
    the names returned are exactly the ones a `serviceName` filter can match.
    Returns [] on any failure - callers fall back to the other sources rather
    than treating "discovery broke" as "no services exist".
    """
    return await searcher.aggregate_services(lookback_hours=lookback_hours)
