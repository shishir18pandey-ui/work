"""Does this error explain the incident that was reported, or is it just present?

A trace for a logged-in customer contains failures that have nothing to do with
what the branch reported. The app fans out on every screen load, and several of
those calls come back empty as a matter of course: no pre-approved offer for this
customer, no investment holdings, a lookup with no matching rows. None of that is
a fault, and none of it is why the customer complained.

The error ranker (`new_flow.tools.query_tools._score_record`) is purely
structural - leaf-ness, payload presence, HTTP class, depth - so an absent-thing
404 at the leaf of its own subtree outranks the real failure, and `summary_agent`
is *required* to quote error codes verbatim. The result, measured on 16-Sep, is
replies telling a branch that a fixed-deposit problem was caused by an offers
lookup finding no offer (7 incidents) or a wealth lookup finding no portfolio (4).
Both were true statements about the trace and both were irrelevant.

**The test is topic overlap, not a list of codes.** An earlier version of this
module carried the specific signatures observed on optimus. That is wrong as
shared code: the same offers-not-found is exactly the evidence for an app whose
job is servicing offers, and a hardcoded list would demote precisely the
incidents that app needs. So nothing here names a business domain. Two
app-neutral conditions have to hold before an error is set aside:

1. **It has to be an absence, not a fault.** Only "there was nothing there"
   outcomes are demotable - a 404, a `*_NOT_FOUND`, an empty result set. A 5xx, a
   timeout or an exception is a real malfunction and is always kept, whatever it
   is about, because a broken dependency is worth reporting even when we cannot
   tie it to the symptom.
2. **Its own vocabulary has to be unrelated to the report.** The words in the
   error's code, service and operation are compared against the incident text. An
   error about offers stays when the ticket says "offer", and is set aside when
   the ticket says "fixed deposit" - the judgement is made per incident, so the
   same code goes both ways depending on what was asked.

An error on a service the plan agent named is never set aside: that service is
where the symptom pointed, so its failures are on-topic by construction.

Demoted, never deleted. Set-aside errors are still rendered, under a header
saying what they are, and when they are all we have the confidence is capped
below the resolve line rather than presented as a diagnosis. An omitted error is
one somebody can see in the trace and we could not explain.

Apps may add measured signatures via `relevance.absent_outcomes` in
`service_metadata.yaml` if the generic patterns miss a house style; that is a
per-app extension, not a default.
"""

import re
from typing import Dict, Iterable, List, Sequence, Set, Tuple

# "Nothing was there" outcomes. Generic phrasings only - no business domain
# appears in this list, and none may be added. An error must match one of these
# (or carry a 404) before topic relevance is even considered, so a genuine
# malfunction can never be set aside on vocabulary grounds alone.
ABSENCE_PATTERNS: Tuple[str, ...] = (
    "not_found",
    "notfound",
    "not found",
    "no documents in result",
    "no document in result",
    "no_account_present",
    "no account present",
    "not_present",
    "not present",
    "no_eligible",
    "no eligible",
    "not_available",
    "not available",
    "no_data",
    "no data found",
    "no record",
    "no_records",
    "empty_result",
    "resource_not",
    "does not exist",
    "doesn't exist",
    "no rows",
)

# Faults. A record matching any of these is kept even if its wording looks like an
# absence, because "the dependency is down" outranks "the dependency said no".
FAULT_PATTERNS: Tuple[str, ...] = (
    "timeout",
    "timed out",
    "connection refused",
    "connection reset",
    "socket",
    "unavailable",
    "internal_server",
    "internal server",
    "exception",
    "deadlock",
    "circuit",
    "too many requests",
    "gateway",
    "unable to",
    "failed to",
    "rollback",
    "null pointer",
    "nullpointer",
)

# Words that appear in error codes and service names without saying anything about
# *what* failed. Overlap on these would make every error relevant to every
# incident, which is the same as having no rule at all.
_TOPIC_STOPWORDS: Set[str] = {
    "err", "error", "errors", "err_code", "code", "codes", "exception", "fail",
    "failed", "failure", "not", "no", "none", "found", "present", "missing",
    "invalid", "unknown", "resource", "result", "results", "response", "request",
    "req", "res", "api", "apis", "service", "services", "svc", "server", "client",
    "internal", "external", "system", "app", "application", "backend", "frontend",
    "web", "mobile", "get", "post", "put", "patch", "delete", "http", "https",
    "url", "uri", "endpoint", "call", "calls", "data", "details", "detail",
    "info", "list", "fetch", "fetching", "get_all", "v1", "v2", "v3", "v4",
    "id", "ids", "key", "keys", "type", "types", "status", "value", "values",
    "prod", "uat", "dev", "test", "the", "and", "for", "with", "from", "this",
    "that", "was", "are", "has", "have", "not_found", "document", "documents",
    "mongo", "sql", "db", "database", "query", "table", "index", "record",
    "records", "row", "rows", "empty", "null", "true", "false",
    "customer", "user", "users", "account", "accounts", "number", "cust",
    "eligible", "eligibility", "available", "unavailable", "timeout", "gateway",
}

_WORD = re.compile(r"[a-z0-9]+")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
# Codes are the one thing a branch quotes verbatim, so they are matched whole as
# well as tokenised: SCREAMING_SNAKE runs of two or more segments.
_CODE = re.compile(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+){1,}\b")


def _record_text(record: Dict) -> str:
    """All error text on one record, lowercased - the same fields the header shows."""
    parts: List[str] = [
        str(record.get("service") or ""),
        str(record.get("operation") or ""),
    ]
    for event in record.get("error_events") or []:
        parts.append(str((event or {}).get("message") or ""))
    for payload in record.get("error_payloads") or []:
        parts.append(str((payload or {}).get("body") or "")[:2000])
    return " ".join(parts).lower()


def _raw_record_text(record: Dict) -> str:
    """Same fields, original casing - needed to find SCREAMING_SNAKE codes."""
    parts: List[str] = [
        str(record.get("service") or ""),
        str(record.get("operation") or ""),
    ]
    for event in record.get("error_events") or []:
        parts.append(str((event or {}).get("message") or ""))
    for payload in record.get("error_payloads") or []:
        parts.append(str((payload or {}).get("body") or "")[:2000])
    return " ".join(parts)


def _tokens(text: str) -> Set[str]:
    """Topic words in a blob: camelCase and snake_case split, stopwords dropped."""
    split = _CAMEL.sub(" ", text or "")
    words = {w for w in _WORD.findall(split.lower()) if len(w) > 2}
    return {w for w in words if w not in _TOPIC_STOPWORDS}


def error_codes(record: Dict) -> List[str]:
    """SCREAMING_SNAKE codes on a record, in order of appearance."""
    found: List[str] = []
    for code in _CODE.findall(_raw_record_text(record)):
        if code not in found:
            found.append(code)
    return found


def _matches_any(text: str, patterns: Iterable[str]) -> bool:
    return any(p in text for p in patterns)


def is_absence_outcome(record: Dict, extra_patterns: Sequence[str] = ()) -> bool:
    """Did this error say "there is nothing here" rather than "I am broken"?

    Only absences are demotable. A 5xx, timeout or exception is a malfunction and
    is always kept - being unable to relate it to the symptom is not a reason to
    hide a broken dependency.
    """
    text = _record_text(record)
    if _matches_any(text, FAULT_PATTERNS):
        return False

    status = str(record.get("http_status") or "").strip()
    if status.startswith("5"):
        return False
    if _matches_any(text, ABSENCE_PATTERNS):
        return True
    if _matches_any(text, [str(p).lower() for p in extra_patterns if p]):
        return True
    # A bare 404 with no message is an absence by definition of the status code.
    return status.startswith("404")


def _topic_overlap(record: Dict, incident_text: str) -> Set[str]:
    """Topic words shared by the error and the incident wording."""
    return _tokens(_record_text(record)) & _tokens(incident_text)


def is_ambient(
    record: Dict,
    search_text: str = "",
    planned_services: Sequence[str] = (),
    absent_outcomes: Sequence[str] = (),
) -> bool:
    """Is this error unrelated to what was reported?

    True only when all of these hold:
      - it is not on a service the plan agent named (those are on-topic by
        construction - the symptom is what sent us there);
      - the incident text does not quote its error code;
      - it is an absence outcome, not a fault (see `is_absence_outcome`);
      - its own vocabulary shares no topic word with the incident text.

    With no incident text there is nothing to be off-topic *to*, so every error is
    treated as relevant - the behaviour before this module existed.
    """
    incident = (search_text or "").strip()
    if not incident:
        return False

    service = str(record.get("service") or "").lower()
    if service and any(service == str(s or "").lower() for s in planned_services):
        return False

    incident_lower = incident.lower()
    for code in error_codes(record):
        if code.lower() in incident_lower:
            return False  # the branch quoted this code - it is the subject

    if not is_absence_outcome(record, absent_outcomes):
        return False

    return not _topic_overlap(record, incident)


def partition_records(
    records: List[Dict],
    search_text: str = "",
    planned_services: Sequence[str] = (),
    absent_outcomes: Sequence[str] = (),
) -> Tuple[List[Dict], List[Dict]]:
    """(explains the incident, set aside as unrelated). Order within each is kept."""
    primary: List[Dict] = []
    ambient: List[Dict] = []
    for record in records:
        off_topic = is_ambient(record, search_text, planned_services, absent_outcomes)
        (ambient if off_topic else primary).append(record)
    return primary, ambient


def get_absent_outcomes(app: str) -> List[str]:
    """Per-app additions to `ABSENCE_PATTERNS`, from `service_metadata.yaml`.

    Optional, and empty for every app today. It exists so that an app whose
    house style names absences in a way the generic patterns miss can extend the
    list *for itself* - without any business domain entering shared code. Fails
    open: unreadable or absent config means the generic patterns only.
    """
    try:
        from new_flow.tools.service_metadata import get_app_config

        config = get_app_config(app) or {}
        section = config.get("relevance") or {}
        values = section.get("absent_outcomes") or []
        return [str(v).strip().lower() for v in values if str(v).strip()]
    except Exception:  # noqa: BLE001 - relevance must never break the fetch
        return []


def ambient_codes(records: List[Dict]) -> List[str]:
    """Codes present in a record set, for logging and span attributes."""
    found: List[str] = []
    for record in records:
        for code in error_codes(record):
            if code not in found:
                found.append(code)
    return found
