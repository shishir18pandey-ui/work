
import os
import logging
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

os.environ["OTEL_SDK_DISABLED"] = "true"

logger = logging.getLogger(__name__)

from crewai import Agent, Task, Crew
from utils.llm import llm_config
from tools.query_tools import JaegerTraceTool
from utils.audit_logger import audit_logger
from utils.llm_cache import create_cached_llm


class JaegerExecutionResult(BaseModel):
    resolved: bool = Field(description="Whether issue was found/resolved")
    diagnosis: str = Field(description="Root cause or findings")
    solution: str = Field(description="Steps to resolve")
    questions: List[str] = Field(
        default_factory=list,
        description="Clarification questions if needed"
    )
    tool_calls: List[Dict] = Field(
        default_factory=list,
        description="All Jaeger calls made during execution"
    )
    final_state: str = Field(description="Final state description")
    confidence: float = Field(default=0.5, description="Confidence score 0-1")
    iterations_completed: int = Field(default=0, description="Number of iterations completed")
    ranges_exhausted_per_iteration: List[int] = Field(
        default_factory=list,
        description="Number of time ranges tried in each iteration"
    )


class JaegerOnlyAgent:
    def __init__(self, max_iterations: int = 5, customer_identifiers: Dict[str, str] = None):
        self.max_iterations = max_iterations
        self.jaeger_tool = JaegerTraceTool()
        self.customer_identifiers = customer_identifiers or {}
        
        self.llm = create_cached_llm(
            model_name=llm_config.model_name,
            temperature=0.0,
            base_url=llm_config.url,
            api_key=llm_config.token
        )
        
        # State tracking
        self.incident_id = "unknown"
        self.current_service = None
        self.current_tag_name = None
        self.current_tag_value = None
    
    async def execute(
        self,
        app: str,
        service: str,
        tag_name: str,
        tag_value: str,
        incident_id: str = "unknown",
        context: str = ""
    ) -> JaegerExecutionResult:
        self.incident_id = incident_id
        tool_calls = []
        ranges_exhausted_per_iteration = []
        
        for iteration in range(1, self.max_iterations + 1):
            logger.info(f"[JaegerOnlyAgent] === ITERATION {iteration} START ===")
            logger.info(f"[JaegerOnlyAgent] Service: {service}, Tag: {tag_name}={tag_value}")
            
            ranges_tried = 0
            iteration_results = []
            
            # Deterministic 7-call loop (time_range_index 0-6)
            for time_range_index in range(7):
                logger.info(f"[JaegerOnlyAgent] Calling Jaeger with time_range_index={time_range_index}")
                
                # Execute Jaeger call
                result = await self._execute_jaeger(
                    app=app,
                    service=service,
                    tag_name=tag_name,
                    tag_value=tag_value,
                    time_range_index=time_range_index
                )
                
                ranges_tried += 1
                
                # Record the call
                call_record = {
                    "tool_name": "search_jaeger_traces",
                    "input_params": {
                        "app": app,
                        "service": service,
                        "tag_name": tag_name,
                        "tag_value": tag_value,
                        "time_range_index": time_range_index
                    },
                    "output": result,
                    "iteration": iteration,
                    "time_range_index": time_range_index
                }
                tool_calls.append(call_record)
                iteration_results.append(result)
                
                # Check for break conditions
                break_reason = self._check_break_conditions(result)
                
                if break_reason == "invalid_service_tag":
                    # WARNING/ERROR about invalid service or tag → break to new iteration
                    logger.warning(f"[JaegerOnlyAgent] Invalid service/tag detected - breaking to NEW ITERATION")
                    ranges_exhausted_per_iteration.append(ranges_tried)
                    
                    # Ask LLM for new service/tag
                    service, tag_name, tag_value = await self._decide_next_service_tag(
                        context=context,
                        previous_results=tool_calls,
                        iteration=iteration,
                        reason="invalid_service_tag"
                    )
                    
                    if not service or not tag_name:
                        # LLM couldn't find valid service/tag
                        logger.warning(f"[JaegerOnlyAgent] LLM couldn't determine new service/tag - ending")
                        return JaegerExecutionResult(
                            resolved=False,
                            diagnosis="Could not find valid service/tag combination",
                            solution="Manual investigation required",
                            tool_calls=tool_calls,
                            final_state="Incomplete - no valid service/tag",
                            confidence=0.3,
                            iterations_completed=iteration,
                            ranges_exhausted_per_iteration=ranges_exhausted_per_iteration
                        )
                    
                    # Break to next iteration with new service/tag
                    break
                
                elif break_reason == "no_traces_continue":
                    # No traces found → continue to next time range
                    logger.info(f"[JaegerOnlyAgent] No traces at range {time_range_index}, continuing...")
                    continue
                
                elif break_reason == "traces_found":
                    # Traces found → return to LLM for decision
                    logger.info(f"[JaegerOnlyAgent] Traces found at range {time_range_index}")
                    ranges_exhausted_per_iteration.append(ranges_tried)
                    
                    # Ask LLM what to do next
                    decision = await self._decide_after_traces(
                        context=context,
                        jaeger_result=result,
                        tool_calls=tool_calls,
                        iteration=iteration
                    )
                    
                    if decision.get("action") == "resolve":
                        return JaegerExecutionResult(
                            resolved=True,
                            diagnosis=decision.get("diagnosis", "Issue identified in traces"),
                            solution=decision.get("solution", "See trace analysis"),
                            tool_calls=tool_calls,
                            final_state="Resolved",
                            confidence=0.8,
                            iterations_completed=iteration,
                            ranges_exhausted_per_iteration=ranges_exhausted_per_iteration
                        )
                    elif decision.get("action") == "new_iteration":
                        # LLM wants to try new service/tag
                        new_service = decision.get("new_service")
                        new_tag_name = decision.get("new_tag_name")
                        new_tag_value = decision.get("new_tag_value")
                        
                        if new_service and new_tag_name:
                            service = new_service
                            tag_name = new_tag_name
                            tag_value = new_tag_value
                            logger.info(f"[JaegerOnlyAgent] LLM decided to try new service={service}, tag={tag_name}")
                            break
                        else:
                            # No new service/tag provided, continue with next iteration
                            pass
                    
                    # Continue to next time range
                    continue
            
            # End of inner loop - check if all 7 ranges exhausted
            if ranges_tried >= 7:
                logger.info(f"[JaegerOnlyAgent] All 7 time ranges exhausted for iteration {iteration}")
                ranges_exhausted_per_iteration.append(ranges_tried)
                
                # Ask LLM for next action
                decision = await self._decide_after_all_ranges(
                    context=context,
                    tool_calls=tool_calls,
                    iteration=iteration
                )
                
                if decision.get("action") == "new_iteration":
                    new_service = decision.get("new_service")
                    new_tag_name = decision.get("new_tag_name")
                    new_tag_value = decision.get("new_tag_value")
                    
                    if new_service and new_tag_name:
                        service = new_service
                        tag_name = new_tag_name
                        tag_value = new_tag_value
                        logger.info(f"[JaegerOnlyAgent] Starting new iteration with service={service}, tag={tag_name}")
                        continue
                    else:
                        # No new service/tag - end execution
                        return JaegerExecutionResult(
                            resolved=False,
                            diagnosis="All time ranges exhausted - no traces found",
                            solution="Manual investigation required",
                            tool_calls=tool_calls,
                            final_state="Incomplete - no traces found",
                            confidence=0.4,
                            iterations_completed=iteration,
                            ranges_exhausted_per_iteration=ranges_exhausted_per_iteration
                        )
                elif decision.get("action") == "resolve":
                    return JaegerExecutionResult(
                        resolved=True,
                        diagnosis=decision.get("diagnosis", "Investigation complete"),
                        solution=decision.get("solution", "No action needed"),
                        tool_calls=tool_calls,
                        final_state="Resolved",
                        confidence=0.6,
                        iterations_completed=iteration,
                        ranges_exhausted_per_iteration=ranges_exhausted_per_iteration
                    )
                else:
                    # LLM wants to stop
                    return JaegerExecutionResult(
                        resolved=False,
                        diagnosis="Investigation ended by LLM",
                        solution="See tool calls for details",
                        tool_calls=tool_calls,
                        final_state="Stopped",
                        confidence=0.5,
                        iterations_completed=iteration,
                        ranges_exhausted_per_iteration=ranges_exhausted_per_iteration
                    )
        
        # Max iterations reached
        logger.warning(f"[JaegerOnlyAgent] Max {self.max_iterations} iterations reached")
        return JaegerExecutionResult(
            resolved=False,
            diagnosis="Max iterations reached - investigation incomplete",
            solution="Manual investigation required",
            tool_calls=tool_calls,
            final_state="Incomplete - max iterations",
            confidence=0.3,
            iterations_completed=self.max_iterations,
            ranges_exhausted_per_iteration=ranges_exhausted_per_iteration
        )
    
    async def _execute_jaeger(
        self,
        app: str,
        service: str,
        tag_name: str,
        tag_value: str,
        time_range_index: int
    ) -> str:
        """Execute a single Jaeger trace query."""
        try:
            result = await self.jaeger_tool._arun(
                app=app,
                service=service,
                tag_name=tag_name,
                tag_value=tag_value,
                time_range_index=time_range_index
            )
            
            # Audit log
            await audit_logger.log_tool_execution(
                incident_id=self.incident_id,
                tool_name="search_jaeger_traces",
                input_params={
                    "app": app,
                    "service": service,
                    "tag_name": tag_name,
                    "tag_value": tag_value,
                    "time_range_index": time_range_index
                },
                output=result
            )
            
            return result
            
        except Exception as e:
            error_msg = f"Jaeger execution error: {str(e)}"
            logger.error(f"[JaegerOnlyAgent] {error_msg}")
            return error_msg
    
    def _check_break_conditions(self, result: str) -> str:
        result_lower = result.lower()
        
        # Check for warning about invalid service or tag
        warning_indicators = [
            "warning",
            "not in known tags",
            "not a valid service",
            "unknown tag",
            "unknown service"
        ]
        has_warning = "warning" in result_lower
        has_invalid = any(indicator in result_lower for indicator in warning_indicators[1:])
        
        if has_warning and has_invalid:
            return "invalid_service_tag"
        
        # Check for no traces
        no_trace_indicators = [
            "no traces found",
            "no traces in time range",
            "0 traces",
            "traces found: 0"
        ]
        if any(indicator in result_lower for indicator in no_trace_indicators):
            return "no_traces_continue"
        
        # Traces found
        return "traces_found"
    

    async def _decide_next_service_tag(
        self,
        context: str,
        previous_results: List[Dict],
        iteration: int,
        reason: str
    ) -> tuple:
        history_text = self._format_tool_calls(previous_results)
        
        prompt = f"""
{context}

=== CURRENT STATE ===
Iteration: {iteration} of {self.max_iterations}
Reason for new service/tag: {reason}

=== PREVIOUS JAEGER CALLS ===
{history_text}

=== TASK ===
The previous service and/or tag combination was invalid (warning/error from Jaeger).
You need to decide a NEW service and tag combination to try.

Available services (from discovery):
- Use GetServicesTool to discover valid services for the app
- Common services: upi-api, idp-api, cbs-backend, kyc-service, etc.

IMPORTANT: Only use "customer_id" or "mobile_number" as TAG_NAME. Do NOT use "ucic", "txn_id", or any other tag.
- If TAG_NAME is "customer_id", the value will be automatically retrieved from customer_identifiers
- If TAG_NAME is "mobile_number", provide the mobile number value directly

Output in this format:
SERVICE: [new service name]
TAG_NAME: [new tag name]
REASONING: [why you chose this combination]
"""
        
        decision_agent = Agent(
            role="Service/Tag Selector",
            goal="Select valid service and tag for Jaeger search",
            backstory=(
                "You are a Jaeger search expert. Your job is to select "
                "valid service and tag combinations based on previous errors."
            ),
            verbose=False,
            allow_delegation=False,
            llm=self.llm,
            temperature=0
        )
        
        decision_task = Task(
            description=prompt,
            agent=decision_agent,
            expected_output="SERVICE, TAG_NAME, TAG_VALUE, REASONING"
        )
        
        crew = Crew(
            agents=[decision_agent],
            tasks=[decision_task],
            verbose=False
        )
        
        result = await crew.akickoff()
        result_text = str(result)
        
        # Parse the response
        return self._parse_service_tag_decision(result_text)
    
    async def _decide_after_traces(
        self,
        context: str,
        jaeger_result: str,
        tool_calls: List[Dict],
        iteration: int
    ) -> Dict:
        """
        Ask LLM what to do after traces are found.
        
        Returns:
            dict with action, diagnosis, solution, or new service/tag
        """
        history_text = self._format_tool_calls(tool_calls)
        
        prompt = f"""
{context}

=== CURRENT STATE ===
Iteration: {iteration} of {self.max_iterations}

=== LATEST JAEGER RESULT ===
{jaeger_result[:1000]}

=== PREVIOUS JAEGER CALLS ===
{history_text}

=== TASK ===
Traces were found in the search. Decide what to do next:

Options:
1. RESOLVE: If root cause is identified in the traces
2. NEW_ITERATION: If you want to try a different service/tag combination
3. CONTINUE: Continue searching more time ranges

IMPORTANT: Only use "customer_id" or "mobile_number" as NEW_TAG_NAME. Do NOT use "ucic", "txn_id", or any other tag.

Output in this format:
ACTION: [resolve|new_iteration|continue]
DIAGNOSIS: [if resolving, what's the root cause]
SOLUTION: [if resolving, how to fix it]
NEW_SERVICE: [if new_iteration, new service name]
NEW_TAG_NAME: [if new_iteration, new tag name]
NEW_TAG_VALUE: [if new_iteration, new tag value]
REASONING: [why you chose this action]
"""
        
        decision_agent = Agent(
            role="Trace Analyzer",
            goal="Analyze Jaeger traces and decide next action",
            backstory=(
                "You are a debugging expert. Analyze Jaeger traces to "
                "identify root causes and decide next steps."
            ),
            verbose=False,
            allow_delegation=False,
            llm=self.llm,
            temperature=0
        )
        
        decision_task = Task(
            description=prompt,
            agent=decision_agent,
            expected_output="ACTION, DIAGNOSIS, SOLUTION, NEW_SERVICE, NEW_TAG_NAME, NEW_TAG_VALUE, REASONING"
        )
        
        crew = Crew(
            agents=[decision_agent],
            tasks=[decision_task],
            verbose=False
        )
        
        result = await crew.akickoff()
        result_text = str(result)
        
        return self._parse_action_decision(result_text)
    
    async def _decide_after_all_ranges(
        self,
        context: str,
        tool_calls: List[Dict],
        iteration: int
    ) -> Dict:
        history_text = self._format_tool_calls(tool_calls)
        
        prompt = f"""
{context}

=== CURRENT STATE ===
Iteration: {iteration} of {self.max_iterations}

All 7 time ranges (0-6) have been exhausted with NO TRACES FOUND.
This means there are no traces for this service/tag combination in the last 7 days.

=== PREVIOUS JAEGER CALLS ===
{history_text}

=== TASK ===
Decide what to do next:

Options:
1. NEW_ITERATION: Try a different service/tag combination (recommended)
2. RESOLVE: Conclude that there's no runtime error (data issue)
3. STOP: End investigation

IMPORTANT: Only use "customer_id" or "mobile_number" as NEW_TAG_NAME. Do NOT use "ucic", "txn_id", or any other tag.

Output in this format:
ACTION: [new_iteration|resolve|stop]
NEW_SERVICE: [if new_iteration, new service name]
NEW_TAG_NAME: [if new_iteration, new tag name]
NEW_TAG_VALUE: [if new_iteration, new tag value]
DIAGNOSIS: [if resolving, what's the conclusion]
REASONING: [why you chose this action]
"""
        
        decision_agent = Agent(
            role="Iteration Planner",
            goal="Decide next action after exhausting time ranges",
            backstory=(
                "You are a debugging orchestrator. After exhausting all time ranges, "
                "decide whether to try new service/tag or conclude investigation."
            ),
            verbose=False,
            allow_delegation=False,
            llm=self.llm,
            temperature=0
        )
        
        decision_task = Task(
            description=prompt,
            agent=decision_agent,
            expected_output="ACTION, NEW_SERVICE, NEW_TAG_NAME, NEW_TAG_VALUE, DIAGNOSIS, REASONING"
        )
        
        crew = Crew(
            agents=[decision_agent],
            tasks=[decision_task],
            verbose=False
        )
        
        result = await crew.akickoff()
        result_text = str(result)
        
        return self._parse_action_decision(result_text)
    
    def _format_tool_calls(self, tool_calls: List[Dict]) -> str:
        """Format tool calls for LLM context."""
        if not tool_calls:
            return "No previous calls"
        
        lines = []
        for call in tool_calls[-10:]:  # Last 10 calls
            params = call.get("input_params", {})
            output = call.get("output", "")[:300]
            iteration = call.get("iteration", "?")
            range_idx = call.get("time_range_index", "?")
            lines.append(
                f"- Iteration {iteration}, Range {range_idx}: "
                f"service={params.get('service')}, tag={params.get('tag_name')}={params.get('tag_value')}"
            )
            lines.append(f"  Output: {output}...")
        
        return "\n".join(lines)
    
    def _parse_service_tag_decision(self, response: str) -> tuple:
        """Parse LLM response for service/tag decision."""
        import re
        
        service_match = re.search(r"SERVICE:\s*(.+?)(?:TAG_NAME:|$)", response, re.DOTALL)
        tag_name_match = re.search(r"TAG_NAME:\s*(.+?)(?:REASONING:|$)", response, re.DOTALL)
        
        # Clean up captured values - split on first newline to remove extra text
        service = service_match.group(1).strip().split('\n')[0] if service_match else None
        tag_name = tag_name_match.group(1).strip().split('\n')[0] if tag_name_match else None
        
        # Handle multiple customer identifier types (customer_id, mobile_number)
        if tag_name in self.customer_identifiers:
            tag_value = self.customer_identifiers.get(tag_name, "")
        elif tag_name == "customer_id" and self.customer_identifiers:
            tag_value = self.customer_identifiers.get("customer_id", "")
        else:
            tag_value = ""
        
        logger.info(f"[JaegerOnlyAgent] LLM chose: service={service}, tag={tag_name}={tag_value}")
        
        return (service, tag_name, tag_value)
    
    def _parse_action_decision(self, response: str) -> Dict:
        """Parse LLM response for action decision."""
        import re
        
        action_match = re.search(r"ACTION:\s*(\w+)", response, re.IGNORECASE)
        diagnosis_match = re.search(r"DIAGNOSIS:\s*(.+?)(?:SOLUTION:|$)", response, re.DOTALL)
        solution_match = re.search(r"SOLUTION:\s*(.+?)(?:NEW_SERVICE:|$)", response, re.DOTALL)
        new_service_match = re.search(r"NEW_SERVICE:\s*(.+?)(?:NEW_TAG_NAME:|$)", response, re.DOTALL)
        new_tag_name_match = re.search(r"NEW_TAG_NAME:\s*(.+?)(?:NEW_TAG_VALUE:|$)", response, re.DOTALL)
        new_tag_value_match = re.search(r"NEW_TAG_VALUE:\s*(.+?)(?:REASONING:|$)", response, re.DOTALL)
        
        # Clean up captured values - split on first newline to remove extra text
        def clean_value(match):
            if match:
                return match.group(1).strip().split('\n')[0]
            return None
        
        return {
            "action": action_match.group(1).lower() if action_match else "continue",
            "diagnosis": diagnosis_match.group(1).strip().split('\n')[0] if diagnosis_match else "",
            "solution": solution_match.group(1).strip().split('\n')[0] if solution_match else "",
            "new_service": clean_value(new_service_match),
            "new_tag_name": clean_value(new_tag_name_match),
            "new_tag_value": clean_value(new_tag_value_match),
        }


async def run_jaeger_only_async(
    plan_output,
    incident_description: str,
    app: str,
    customer_identifiers: Dict[str, str],
    problem_category: str,
    max_iterations: int = 5,
    incident_id: str = "unknown",
    discovered_services: str = ""
) -> JaegerExecutionResult:
    """
    Convenience function to run Jaeger-only execution.
    
    Args:
        plan_output: PlanOutput from plan agent (contains suggested_service, customer_identifiers)
        incident_description: Description of the incident
        app: Application name (optimus, cbs, idp)
        customer_identifiers: Dict of customer identifiers from incident
        problem_category: Category of the problem
        max_iterations: Maximum iterations (default 5)
        incident_id: Incident ID for logging
        discovered_services: Available Jaeger services
        
    Returns:
        JaegerExecutionResult with findings
    """
    # Import PlanOutput for type hint
    from agents.plan_agent import PlanOutput
    
    # Extract service and tag from plan_output
    service = ""
    tag_name = ""
    tag_value = ""
    
    if plan_output and isinstance(plan_output, PlanOutput):
        service = plan_output.suggested_service or ""
        
        # Extract first customer identifier as tag
        if customer_identifiers:
            # Prioritize customer_id as the tag
            if customer_identifiers and "customer_id" in customer_identifiers:
                tag_name = "customer_id"
                tag_value = customer_identifiers["customer_id"]
            else:
                tag_name = ""
                tag_value = ""
    
    # Build context for LLM decisions
    context = f"""
=== INCIDENT CONTEXT ===
Application: {app}
Problem Category: {problem_category}
Incident Description: {incident_description}

Customer Identifiers:
{chr(10).join(f"  - {k}: {v}" for k, v in customer_identifiers.items()) if customer_identifiers else "  None provided"}

Available Services:
{discovered_services}

=== PLAN AGENT FINDINGS ===
Issue Summary: {plan_output.issue_summary if plan_output else 'N/A'}
Suggested Service: {service}
"""
    
    agent = JaegerOnlyAgent(max_iterations=max_iterations, customer_identifiers=customer_identifiers)
    
    return await agent.execute(
        app=app,
        service=service,
        tag_name=tag_name,
        tag_value=tag_value,
        incident_id=incident_id,
        context=context
    )
