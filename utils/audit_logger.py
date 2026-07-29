"""
Audit Logger for Incident Manager.

Logs all tool executions to PostgreSQL for BFSI compliance and audit trail.
"""

import os
import logging
from datetime import datetime
from typing import Optional, Dict, Any

os.environ["OTEL_SDK_DISABLED"] = "true"

logger = logging.getLogger(__name__)

# Import asyncpg for database operations
import asyncpg

# Global pool for audit logs
_audit_pool = None

# Table name for audit logs
AUDIT_TABLE_NAME = "audit_logs"


async def get_audit_db_pool():
    """Get or create database connection pool for audit logs."""
    global _audit_pool
    if _audit_pool is None:
        connection_string = os.environ.get(
            'INCIDENT_AGENT_POSTGRES_CONNECTION_STRING',
            'postgresql://incidentagent:incidentagent123@postgres:5432/incidentagent'
        )
        _audit_pool = await asyncpg.create_pool(
            connection_string,
            min_size=2,
            max_size=10
        )
    return _audit_pool


class AuditLogger:
    """
    Audit logger for tracking all tool executions.
    
    Used for BFSI compliance - every query and action must be logged.
    """
    
    def __init__(self):
        # Disabled by default - enable when audit DB is properly set up
        self.enabled = False
    
    async def log_query(
        self,
        incident_id: str,
        tool_name: str,
        query: str,
        result: str,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Log a tool execution to the audit table.
        
        Args:
            incident_id: The incident ID
            tool_name: Name of the tool executed (e.g., 'execute_db_query', 'search_elk_logs')
            query: The query or input parameters (truncated if too long)
            result: The result/output (truncated if too long)
            user_id: Optional user ID
            metadata: Optional additional metadata (e.g., service_name from Jaeger)
            
        Returns:
            True if logged successfully, False otherwise
        """
        if not self.enabled:
            return True
            
        try:
            pool = await get_audit_db_pool()
            
            # Truncate long queries/results for storage
            query_truncated = str(query)[:2000] if query else ""
            result_truncated = str(result)[:2000] if result else ""
            
            # Build metadata string if provided
            metadata_str = None
            if metadata:
                metadata_str = str(metadata)[:500]
            
            async with pool.acquire() as conn:
                await conn.execute(
                    f"""
                    INSERT INTO {AUDIT_TABLE_NAME} 
                    (incident_id, tool_name, query, result, user_id, metadata, timestamp)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    """,
                    incident_id,
                    tool_name,
                    query_truncated,
                    result_truncated,
                    user_id,
                    metadata_str,
                    datetime.utcnow()
                )
            
            logger.info(f"[AuditLogger] Logged: {incident_id} | {tool_name}")
            return True
            
        except Exception as e:
            logger.error(f"[AuditLogger] Failed to log: {e}")
            return False
    
    async def log_sql_query(
        self,
        incident_id: str,
        sql_query: str,
        result: str,
        db_instance: str = "main",
        service_name: Optional[str] = None
    ) -> bool:
        """
        Log a SQL query execution with additional context.
        
        This specifically handles SQL queries, including those extracted
        from Jaeger traces (tag.sql@query, process.serviceName).
        
        Args:
            incident_id: The incident ID
            sql_query: The SQL query executed
            result: Query result or error message
            db_instance: Database instance (main, idp, etc.)
            service_name: Service that executed the query (from Jaeger)
        """
        metadata = {
            "db_instance": db_instance,
            "service_name": service_name,
            "query_source": "direct" if service_name is None else "jaeger_trace"
        }
        
        return await self.log_query(
            incident_id=incident_id,
            tool_name="execute_db_query",
            query=sql_query,
            result=result,
            metadata=metadata
        )
    
    async def log_tool_execution(
        self,
        incident_id: str,
        tool_name: str,
        input_params: Dict[str, Any],
        output: str,
        success: bool = True
    ) -> bool:
        """
        Log a generic tool execution.
        
        Args:
            incident_id: The incident ID
            tool_name: Name of the tool
            input_params: Input parameters as dict
            output: Tool output
            success: Whether execution was successful
        """
        metadata = {
            "success": success,
            "input_params": str(input_params)[:500]
        }
        
        return await self.log_query(
            incident_id=incident_id,
            tool_name=tool_name,
            query=str(input_params),
            result=output,
            metadata=metadata
        )


# Global instance for easy import
audit_logger = AuditLogger()


# ─────────────────────────────────────────────────────────────────────────────
# Database setup function
# ─────────────────────────────────────────────────────────────────────────────

async def create_audit_table_if_not_exists():
    """
    Create the audit_logs table if it doesn't exist.
    
    Run this on application startup.
    """
    pool = await get_audit_db_pool()
    
    async with pool.acquire() as conn:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_logs (
                id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                incident_id VARCHAR NOT NULL,
                tool_name VARCHAR NOT NULL,
                query TEXT,
                result TEXT,
                user_id VARCHAR,
                metadata TEXT,
                timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        
        # Create index for faster queries
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_audit_incident_id 
            ON audit_logs(incident_id)
            """
        )
        
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_audit_tool_name 
            ON audit_logs(tool_name)
            """
        )
        
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_audit_timestamp 
            ON audit_logs(timestamp)
            """
        )
    
    logger.info("[AuditLogger] Audit table ready")


async def get_audit_logs_for_incident(incident_id: str, limit: int = 100):
    """
    Retrieve audit logs for a specific incident.
    
    Args:
        incident_id: The incident ID
        limit: Maximum number of logs to return
        
    Returns:
        List of audit log records
    """
    pool = await get_audit_db_pool()
    
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, incident_id, tool_name, query, result, user_id, metadata, timestamp
            FROM audit_logs
            WHERE incident_id = $1
            ORDER BY timestamp DESC
            LIMIT $2
            """,
            incident_id,
            limit
        )
        
        return [dict(row) for row in rows]



this is circuit breaker.py

import os
import time
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field
from threading import Lock

os.environ["OTEL_SDK_DISABLED"] = "true"

logger = __import__('logging').getLogger(__name__)


@dataclass
class FailureRecord:
    """Record of failures for a specific key."""
    count: int = 0
    last_failure_time: float = 0
    first_failure_time: float = 0


@dataclass
class CircuitBreakerStats:
    """Statistics about circuit breaker state."""
    is_open: bool
    failure_count: int
    key: str
    time_since_last_failure: float


class CircuitBreaker:
    """
    Circuit Breaker to prevent repeated failures.
    
    Tracks failures per customer_id + issue_type combination.
    Opens circuit after max_failures, preventing further attempts.
    """
    
    def __init__(
        self,
        max_failures: int = 3,
        reset_window_seconds: float = 3600  # 1 hour
    ):
        self.max_failures = max_failures
        self.reset_window_seconds = reset_window_seconds
        self._failures: Dict[str, FailureRecord] = {}
        self._lock = Lock()
    
    def _make_key(self, customer_id: str, issue_type: str) -> str:
        """Create a unique key for customer + issue combination."""
        return f"{customer_id}:{issue_type}"
    
    def record_failure(self, customer_id: str, issue_type: str) -> bool:
        """
        Record a failure and return True if circuit should open.
        
        Args:
            customer_id: Customer identifier
            issue_type: Type of issue/problem category
            
        Returns:
            True if circuit is now open (max failures reached)
        """
        key = self._make_key(customer_id, issue_type)
        current_time = time.time()
        
        with self._lock:
            if key not in self._failures:
                self._failures[key] = FailureRecord(
                    count=0,
                    last_failure_time=current_time,
                    first_failure_time=current_time
                )
            
            record = self._failures[key]
            
            # Check if we're in reset window
            if current_time - record.first_failure_time > self.reset_window_seconds:
                # Reset the failure count
                record.count = 0
                record.first_failure_time = current_time
            
            record.count += 1
            record.last_failure_time = current_time
            
            is_open = record.count >= self.max_failures
            
            if is_open:
                logger.warning(
                    f"Circuit opened for {key} after {record.count} failures"
                )
            
            return is_open
    
    def record_success(self, customer_id: str, issue_type: str) -> None:
        """Record a success and reset failure count."""
        key = self._make_key(customer_id, issue_type)
        
        with self._lock:
            if key in self._failures:
                logger.info(f"Circuit reset for {key} after successful resolution")
                del self._failures[key]
    
    def is_open(self, customer_id: str, issue_type: str) -> Tuple[bool, Optional[str]]:
        """
        Check if circuit is open for customer + issue combination.
        
        Args:
            customer_id: Customer identifier
            issue_type: Type of issue/problem category
            
        Returns:
            Tuple of (is_open, reason_if_open)
        """
        key = self._make_key(customer_id, issue_type)
        current_time = time.time()
        
        with self._lock:
            if key not in self._failures:
                return False, None
            
            record = self._failures[key]
            
            # Check if we're past the reset window
            if current_time - record.first_failure_time > self.reset_window_seconds:
                del self._failures[key]
                return False, None
            
            # Circuit is open
            if record.count >= self.max_failures:
                return True, f"Circuit open: {record.count} failures in last hour"
            
            return False, None
    
    def get_stats(self) -> Dict[str, CircuitBreakerStats]:
        """Get statistics for all tracked circuits."""
        current_time = time.time()
        stats = {}
        
        with self._lock:
            for key, record in self._failures.items():
                stats[key] = CircuitBreakerStats(
                    is_open=record.count >= self.max_failures,
                    failure_count=record.count,
                    key=key,
                    time_since_last_failure=current_time - record.last_failure_time
                )
        
        return stats
    
    def reset(self, customer_id: str = None, issue_type: str = None) -> None:
        """Reset circuit for specific customer/issue or all if no params."""
        if customer_id is None and issue_type is None:
            with self._lock:
                self._failures.clear()
            logger.info("All circuits reset")
        else:
            key = self._make_key(customer_id, issue_type)
            with self._lock:
                if key in self._failures:
                    del self._failures[key]
            logger.info(f"Circuit reset for {key}")


# Global circuit breaker instance
_circuit_breaker: Optional[CircuitBreaker] = None


def get_circuit_breaker() -> CircuitBreaker:
    """Get the global circuit breaker instance."""
    global _circuit_breaker
    if _circuit_breaker is None:
        _circuit_breaker = CircuitBreaker(max_failures=3, reset_window_seconds=3600)
    return _circuit_breaker


# Tool Call Failure Tracker
class ToolCallFailureTracker:
    """
    Tracks consecutive tool call failures to stop failing crews early.
    
    Instead of running all max_iterations, stops after N consecutive tool failures.
    """
    
    def __init__(self, max_consecutive_failures: int = 3):
        self.max_consecutive_failures = max_consecutive_failures
        self._consecutive_failures: Dict[str, int] = {}  # incident_id -> count
        self._lock = Lock()
    
    def record_tool_failure(self, incident_id: str) -> bool:
        """
        Record a tool call failure.
        
        Args:
            incident_id: The incident ID
            
        Returns:
            True if should stop (max consecutive failures reached)
        """
        with self._lock:
            current = self._consecutive_failures.get(incident_id, 0) + 1
            self._consecutive_failures[incident_id] = current
            
            should_stop = current >= self.max_consecutive_failures
            
            if should_stop:
                logger.warning(
                    f"Stopping crew for {incident_id} after {current} consecutive tool failures"
                )
            
            return should_stop
    
    def record_tool_success(self, incident_id: str) -> None:
        """Reset failure count on successful tool call."""
        with self._lock:
            if incident_id in self._consecutive_failures:
                del self._consecutive_failures[incident_id]
    
    def get_consecutive_failures(self, incident_id: str) -> int:
        """Get current consecutive failure count for incident."""
        with self._lock:
            return self._consecutive_failures.get(incident_id, 0)
    
    def should_stop_early(
        self,
        incident_id: str,
        tool_calls: list,
        current_iteration: int
    ) -> Tuple[bool, Optional[str]]:
        """
        Determine if we should stop the crew early due to failures.
        
        Args:
            incident_id: The incident ID
            tool_calls: List of tool calls made so far
            current_iteration: Current iteration number
            
        Returns:
            Tuple of (should_stop, reason)
        """
        if not tool_calls:
            return False, None
        
        # Check last N tool calls for failures
        recent_calls = tool_calls[-self.max_consecutive_failures:]
        failure_count = 0
        
        for call in recent_calls:
            output = call.get('output', '').lower()
            # Check if tool actually failed (not just if output contains error info)
            # A tool fails if: returns 0 results, has explicit error, or returns empty
            is_failure = False
            
            # Check for explicit failure indicators
            if 'no traces found' in output or '0 traces' in output:
                is_failure = True
            elif 'error:' in output and ('api returned' in output or 'exception' in output or 'traceback' in output):
                is_failure = True
            elif 'failed to' in output and ('connect' in output or 'timeout' in output or 'auth' in output):
                is_failure = True
            elif not output or output.strip() == '' or output == 'none':
                is_failure = True
            
            if is_failure:
                failure_count += 1
        
        if failure_count >= self.max_consecutive_failures:
            return True, f"Stopping after {failure_count} consecutive tool failures"
        
        return False, None
    
    def reset(self, incident_id: str = None) -> None:
        """Reset failure tracking for incident."""
        with self._lock:
            if incident_id:
                if incident_id in self._consecutive_failures:
                    del self._consecutive_failures[incident_id]
            else:
                self._consecutive_failures.clear()


# Global tool failure tracker
_tool_failure_tracker: Optional[ToolCallFailureTracker] = None


def get_tool_failure_tracker() -> ToolCallFailureTracker:
    """Get the global tool failure tracker instance."""
    global _tool_failure_tracker
    if _tool_failure_tracker is None:
        _tool_failure_tracker = ToolCallFailureTracker(max_consecutive_failures=3)
    return _tool_failure_tracker


this is context compression.py
import os
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

os.environ["OTEL_SDK_DISABLED"] = "true"

logger = __import__('logging').getLogger(__name__)


# Token estimation constants
# Average tokens to characters ratio varies by model, but ~4 chars/token is a safe estimate
CHARS_PER_TOKEN = 4
DEFAULT_MAX_TOKENS = 200000
TRUNCATE_OUTPUT_CHARS = 500


@dataclass
class CompressionStats:
    original_tokens: int
    compressed_tokens: int
    tool_calls_summarized: int
    tool_calls_kept: int


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return len(text) / CHARS_PER_TOKEN


def truncate_output(output: str, max_chars: int = TRUNCATE_OUTPUT_CHARS) -> str:
    if not output:
        return ""
    
    if len(output) <= max_chars:
        return output
    
    return output[:max_chars].rstrip() + "..."


def extract_key_info(tool_name: str, output: str) -> str:
    output_lower = output.lower()
    
    if "error" in output_lower or "exception" in output_lower or "failed" in output_lower:
        # Find the error message
        error_match = re.search(r'(error|exception|failed)[:\s]*(.+)', output_lower, re.IGNORECASE)
        if error_match:
            return f"Error: {error_match.group(2)[:200]}"
        return f"Error: {output[:150]}"

    if "select" in output_lower and ("row" in output_lower or "result" in output_lower):
        row_match = re.search(r'(\d+)\s*(row|result)', output_lower)
        if row_match:
            return f"Returned {row_match.group(1)} rows"
    
    if "jaeger" in tool_name.lower():
        if "no trace" in output_lower or "no result" in output_lower:
            return "No traces found"
        span_match = re.search(r'(\d+)\s*span', output_lower)
        if span_match:
            return f"Found {span_match.group(1)} spans"
    
    if "elk" in tool_name.lower() or "log" in tool_name.lower():
        hit_match = re.search(r'(\d+)\s*hit', output_lower)
        if hit_match:
            return f"Found {hit_match.group(1)} log entries"
    
    # Default: first 100 chars
    return f"Result: {output[:100]}..."


def summarize_tool_calls(
    tool_calls: List[Dict],
    keep_recent: int = 3,
    max_tokens: int = DEFAULT_MAX_TOKENS
) -> Tuple[str, CompressionStats]:
    if not tool_calls:
        return "", CompressionStats(0, 0, 0, 0)
    
    # Start with recent calls (keep fully)
    recent_calls = tool_calls[-keep_recent:]
    older_calls = tool_calls[:-keep_recent] if len(tool_calls) > keep_recent else []
    
    # Build recent section
    recent_section = "=== RECENT TOOL CALLS (full) ===\n"
    recent_tokens = estimate_tokens(recent_section)
    
    for call in recent_calls:
        tool_name = call.get('tool_name', 'unknown')
        input_params = call.get('input_params', {})
        output = truncate_output(call.get('output', ''), TRUNCATE_OUTPUT_CHARS)
        
        call_text = f"\n- Tool: {tool_name}\n  Input: {input_params}\n  Output: {output}\n"
        recent_section += call_text
        recent_tokens += estimate_tokens(call_text)
    
    # Build summarized section for older calls
    summarized_section = ""
    if older_calls:
        summarized_section = "\n=== EARLIER TOOL CALLS (summarized) ===\n"
        for call in older_calls:
            tool_name = call.get('tool_name', 'unknown')
            output = call.get('output', '')
            key_info = extract_key_info(tool_name, output)
            summarized_section += f"- {tool_name}: {key_info}\n"
    
    summarized_tokens = estimate_tokens(summarized_section)
    
    # Check if we need to further compress
    total_tokens = recent_tokens + summarized_tokens
    
    # If still over budget, reduce kept recent calls
    if total_tokens > max_tokens and keep_recent > 1:
        # Recursive call with fewer kept calls
        return summarize_tool_calls(tool_calls, keep_recent - 1, max_tokens)
    
    # Combine sections
    context = recent_section + summarized_section
    
    original_tokens = sum(estimate_tokens(call.get('output', '')) for call in tool_calls)
    
    stats = CompressionStats(
        original_tokens=int(original_tokens),
        compressed_tokens=int(total_tokens),
        tool_calls_summarized=len(older_calls),
        tool_calls_kept=len(recent_calls)
    )
    
    return context, stats


def compress_context(
    context: str,
    tool_calls: List[Dict],
    max_tokens: int = DEFAULT_MAX_TOKENS
) -> Tuple[str, Optional[CompressionStats]]:

    estimated_current = estimate_tokens(context)
    
    if estimated_current <= max_tokens:
        return context, None
    
    logger.info(f"Context compression needed: {estimated_current} tokens > {max_tokens} max")
    
    tool_calls_pattern = r'=== (PREVIOUS|RECENT|EARLIER) TOOL CALLS ===[\s\S]*?(?==== |$)'
    
    header_match = re.split(tool_calls_pattern, context)
    header = header_match[0] if header_match else ""
    
    footer = ""
    if len(header_match) > 1:
        parts = re.split(tool_calls_pattern, context, maxsplit=1)
        if len(parts) > 1:
            footer = parts[-1] if len(parts) > 1 else ""
    
    calls_to_compress = tool_calls if tool_calls else []
    
    compressed_tool_calls, stats = summarize_tool_calls(calls_to_compress, max_tokens=max_tokens - estimate_tokens(header + footer))
    
    compressed_context = header + compressed_tool_calls + footer
    
    return compressed_context, stats


def create_compressed_context(
    header: str,
    tool_calls: List[Dict],
    footer: str = "",
    max_tokens: int = DEFAULT_MAX_TOKENS
) -> str:
    full_context = header

    if tool_calls:
        tool_section, stats = summarize_tool_calls(tool_calls, max_tokens=max_tokens - estimate_tokens(header + footer))
        full_context += "\n\n" + tool_section

    full_context += "\n\n" + footer if footer else ""

    # If still too large, truncate header
    if estimate_tokens(full_context) > max_tokens:
        header_tokens = int(max_tokens * 0.3)  # Use 30% for header
        tool_tokens = int(max_tokens * 0.6)  # Use 60% for tool calls
        footer_tokens = int(max_tokens * 0.1)  # Use 10% for footer
        
        # Re-compress with specific budgets
        compressed_tool_calls, _ = summarize_tool_calls(tool_calls, max_tokens=tool_tokens)
        
        # Truncate header and footer
        if len(header) > header_tokens * CHARS_PER_TOKEN:
            header = header[:header_tokens * CHARS_PER_TOKEN] + "...[truncated]..."
        
        full_context = header + "\n\n" + compressed_tool_calls
        
        if footer and len(footer) > footer_tokens * CHARS_PER_TOKEN:
            full_context += "\n\n" + footer[:footer_tokens * CHARS_PER_TOKEN] + "...[truncated]"
        elif footer:
            full_context += "\n\n" + footer
    
    return full_context




this si flow.py

from pydantic import BaseModel
from typing import List, Optional, Dict
import httpx
import os
import json
import re
from dotenv import load_dotenv
from crewai.flow.flow import Flow, start, listen, router
from utils.incident_db_async import upsert_incident_payload_async
from agents.context_builder import run_incident_context_crew_async
from agents.intent_classifier import run_classifier_with_enrichment_async
from agents.plan_agent import run_plan_agent_async
# from agents.execute_agent import run_execute_agent_async
from agents.execute_agent_jaeger import run_jaeger_only_async
from agents.summary_agent import run_summary_agent_async
from agents.self_critique import run_self_critique_async, should_escalate
from utils.llm import run_crew_with_retry_async
from tools.discovery_tools import discover_jaeger_services_impl


load_dotenv()

CA_CERT_FILE = os.getenv("CA_CERT_FILE", "./IDFCBANKCA.pem")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "/app/models/MiniMax-M2.5")
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
    agent_output: dict = {}
    snow_status: str = ""
    ucic: str = ""
    current_comment: Optional[str] = None
    app: str = "" 
    customer_identifiers: Dict[str, str] = {}
    problem_category: str = "" 
    plan_output: Optional[Dict] = None
    execution_result: Optional[Dict] = None
    summary_output: Optional[Dict] = None
    enriched_prompt: str = ""
    discovered_services: str = ""


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
            # "cause": "Resolved by Bot.",
            # "state": "Resolved",
            # "resolutionCode": "Solved (Permanently)",
            # "solutionType": "Other",
            # "outageType": "No Outage"
            "state":"On Hold",
            "onHoldReason": "User Action Required"
        })
        result = await send_update_to_servicenow_async(payload, resolution, None)
        return result, 'on_hold'    


def payload_to_incident_description(payload):
    from utils.files_processor import process_attachments 

    tracer = get_tracer(__name__)
    with tracer.start_as_current_span("payload_to_incident_description") as span:
        span.set_attribute("short_description_length", len(payload.get('shortDescription','')))
        span.set_attribute("description_length", len(payload.get('description','')))
        span.set_attribute("individualUCIC", payload.get('individualUCIC','i'))
        
        short_description = payload.get('shortDescription','')
        description = payload.get('description','')
        individualUCIC = payload.get('individualUCIC','i')
        result = f"Short Description: {short_description}\nDescription: {description}"

        file_text = process_attachments(payload.get('files') or [])   
        if file_text:                                                  
            result = result + "\n\n" + file_text                       
        print(f"Generated incident description for UCIC {individualUCIC}")
        return result, individualUCIC



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

            comment = self.state.current_comment if self.state.current_comment else None

            classifier_output = await run_crew_with_retry_async(
                lambda: run_classifier_with_enrichment_async(
                    payload=self.state.payload,
                    incident_description=self.state.incident_description,
                    user_qa_pairs=self.state.user_qa_pairs,
                    comment=comment
                )
            )
            self.state.intent = classifier_output.intent
            self.state.customer_identifiers = classifier_output.customer_identifiers
            self.state.problem_category = classifier_output.problem_category
            self.state.app = classifier_output.app
            self.state.enriched_prompt = classifier_output.enriched_prompt
            
            self.state.payload['__agent_data']['classifier_output'] = classifier_output.model_dump()
            
            print(f"Enhanced classifier | incident={self.state.incident_id} | app={self.state.app} | category={self.state.problem_category}")
            
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

            classifier_data = self.state.payload.get('__agent_data', {}).get('classifier_output', {})
            if classifier_data.get('needs_user_input'):
                print(f"Need more info for incident {self.state.incident_id}")
                self.state.agent_output = {
                    "resolved": "no",
                    "diagnosis": "Additional information needed",
                    "solution": "Please provide more details",
                    "questions": [classifier_data.get('clarification_question', 'Could you please provide more information? Please provide trace ID/ Correlation ID/ Customer Number/ Account Number if possible')]
                }
                return "update_servicenow"
            
            print(f"Fresh incident {self.state.incident_id}, gather context")
            return "gather_context"


    @listen('gather_context')
    async def semantic_search(self):
        tracer = get_tracer(__name__)
        with tracer.start_as_current_span("gather_context") as span:
            span.set_attribute("incident_id", self.state.incident_id)
            span.set_attribute("incident_description_length", len(self.state.incident_description))

            # Use already-generated description from initialize_and_classify
            # desc = f"{self.state.incident_description}\nUCIC: {self.state.ucic}"
            desc = self.state.enriched_prompt
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

            app = self.state.payload.get("tier1", "cbs").lower().strip()
            self.state.app = app
            span.set_attribute("app", app)
            logger.info(f"Resolver | incident={self.state.incident_id} app={app}")

            return await self._run_agentic_resolver(app)

    async def _run_agentic_resolver(self, app: str):
        tracer = get_tracer(__name__)

        customer_ids = self.state.customer_identifiers if hasattr(self.state, 'customer_identifiers') else {}
        problem_cat = self.state.problem_category if hasattr(self.state, 'problem_category') else ""

        try:
            self.state.discovered_services = await discover_jaeger_services_impl(app)
            logger.info(f"Discovered services for app={app}")
        except Exception as e:
            logger.warning(f"Failed to discover services: {e}")
            self.state.discovered_services = "Service discovery unavailable"

        with tracer.start_as_current_span("plan_agent") as span:
            span.set_attribute("app", app)
            plan_output = await run_crew_with_retry_async(
                lambda: run_plan_agent_async(
                    enriched_prompt=self.state.enriched_prompt,
                    app=app,
                    customer_identifiers=customer_ids,
                    problem_category=problem_cat,
                    incident_context=self.state.incident_context,
                    discovered_services=self.state.discovered_services
                )
            )
            self.state.plan_output = plan_output.model_dump() if hasattr(plan_output, 'model_dump') else plan_output
            print(f"Plan Agent completed for incident {self.state.incident_id}")
        
        if plan_output.needs_more_info:
            self.state.agent_output = {
                "resolved": "no",
                "diagnosis": "Additional information needed",
                "solution": plan_output.question_for_user or "Please provide more details",
                "questions": [plan_output.question_for_user] if plan_output.question_for_user else []
            }
            return "update_servicenow"
        
        with tracer.start_as_current_span("execute_agent") as span:
            span.set_attribute("issue_summary", plan_output.issue_summary[:100] if plan_output.issue_summary else "")
            execution_result = await run_crew_with_retry_async(
                lambda: run_jaeger_only_async(
                    plan_output=plan_output,
                    incident_description=self.state.incident_description,
                    app=app,
                    customer_identifiers=customer_ids,
                    problem_category=problem_cat,
                    max_iterations=5,
                    incident_id=self.state.incident_id,
                    discovered_services=self.state.discovered_services
                )
            )
            self.state.execution_result = execution_result.model_dump() if hasattr(execution_result, 'model_dump') else execution_result
            print(f"Execute Agent completed for incident {self.state.incident_id}")
            
            # Handle escalation if confidence is low
            if hasattr(execution_result, 'confidence') and execution_result.confidence < 0.3:
                # Trigger escalation via ServiceNow
                escalation_msg = execution_result.escalation_reason or "BOT unable to resolve - assigning to L2 Engineer"
                (status, info), incident_status = await send_rejection_to_servicenow_async(
                    self.state.payload, 
                    escalation_msg
                )
                self.state.payload['__agent_data']['snow_logs'].append({
                    "type": "escalation", 
                    "reason": escalation_msg,
                    "confidence": execution_result.confidence,
                    "status": status, 
                    "response": info
                })
                print(f"Escalated incident {self.state.incident_id} due to low confidence: {execution_result.confidence}")
        
        with tracer.start_as_current_span("summary_agent") as span:
            summary_output = await run_crew_with_retry_async(
                lambda: run_summary_agent_async(
                    incident_description=self.state.incident_description,
                    execution_result=execution_result,
                    historic_context=self.state.incident_context,
                    user_qa_pairs=self.state.user_qa_pairs
                )
            )
            self.state.summary_output = summary_output.model_dump() if hasattr(summary_output, 'model_dump') else summary_output
            print(f"Summary Agent completed for incident {self.state.incident_id}")
        
        # Self-Critique Loop: DISABLED - causing false escalations
        # if summary_output.resolved == "yes":
        #     tool_calls = self.state.execution_result.get('tool_calls', []) if isinstance(self.state.execution_result, dict) else []
        #     critique = await run_self_critique_async(
        #         incident_description=self.state.incident_description,
        #         diagnosis=summary_output.diagnosis,
        #         solution=summary_output.solution,
        #         tool_calls=tool_calls,
        #         user_qa_pairs=self.state.user_qa_pairs
        #     )
        #     
        #     # Log critique results
        #     self.state.payload['__agent_data']['snow_logs'].append({
        #         "type": "self_critique",
        #         "is_valid": critique.is_valid,
        #         "confidence": critique.confidence,
        #         "issues_found": critique.issues_found
        #     })
        #     
        #     # Use revised output if critique found issues
        #     if critique.revised_diagnosis:
        #         summary_output.diagnosis = critique.revised_diagnosis
        #     if critique.revised_solution:
        #         summary_output.solution = critique.revised_solution
        #     
        #     # Check if should escalate based on critique
        #     if should_escalate(critique):
        #         print(f"Self-critique triggered escalation for incident {self.state.incident_id}")
        #         (status, info), incident_status = await send_rejection_to_servicenow_async(
        #             self.state.payload,
        #             f"Auto-escalated: Low confidence ({critique.confidence:.2f}). Issues: {', '.join(critique.issues_found[:2])}"
        #         )
        #         self.state.payload['__agent_data']['snow_logs'].append({
        #             "type": "escalation", 
        #             "reason": "self_critique_escalation",
        #             "confidence": critique.confidence,
        #             "status": status, 
        #             "response": info
        #         })
        #         return "update_servicenow"
        
        self.state.agent_output = {
            "resolved": summary_output.resolved,
            "diagnosis": summary_output.diagnosis,
            "solution": summary_output.solution,
            "questions": summary_output.questions
        }
        
        return "update_servicenow"

    @listen(run_resolver_crew)
    async def update_servicenow(self):
        tracer = get_tracer(__name__)
        with tracer.start_as_current_span("update_servicenow") as span:
            span.set_attribute("incident_id", self.state.incident_id)
            span.set_attribute("resolution_result", self.state.agent_output.get("resolved", "unknown"))

            res = self.state.agent_output
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
