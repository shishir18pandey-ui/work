"""
Self-Critique Agent for Incident Manager.

Verifies the generated diagnosis and solution before sending to user.
This follows the Self-Critique Evaluator Loop pattern from agentic best practices.
"""

import os
import json
import re
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

os.environ["OTEL_SDK_DISABLED"] = "true"

from crewai import Agent, Task, Crew, LLM
from utils.llm import llm_config
from utils.observability import get_tracer


class SelfCritiqueOutput(BaseModel):
    """Output from Self-Critique Agent."""
    is_valid: bool = Field(description="Whether the diagnosis/solution is valid")
    confidence: float = Field(description="Confidence score 0-1")
    issues_found: List[str] = Field(
        default_factory=list,
        description="Issues or gaps identified"
    )
    suggested_fixes: List[str] = Field(
        default_factory=list,
        description="Suggestions to improve the resolution"
    )
    revised_diagnosis: Optional[str] = Field(
        default=None,
        description="Revised diagnosis if issues found"
    )
    revised_solution: Optional[str] = Field(
        default=None,
        description="Revised solution if issues found"
    )


async def run_self_critique_async(
    incident_description: str,
    diagnosis: str,
    solution: str,
    tool_calls: List[Dict] = None,
    user_qa_pairs: List[Dict] = None
) -> SelfCritiqueOutput:

    tracer = get_tracer(__name__)
    
    with tracer.start_as_current_span("self_critique") as span:
        span.set_attribute("incident_description_length", len(incident_description))
        span.set_attribute("diagnosis_length", len(diagnosis))
        span.set_attribute("solution_length", len(solution))
        
        llm = LLM(
            model="openai/" + llm_config.model_name,
            temperature=0.0,
            base_url=llm_config.url,
            api_key=llm_config.token
        )
        
        # Build context for critique
        tool_calls_text = ""
        if tool_calls:
            tool_calls_text = "\n=== TOOL CALLS MADE ===\n"
            for call in tool_calls[-5:]:  # Last 5 tool calls
                tool_calls_text += f"- {call.get('tool_name', 'unknown')}: {call.get('output', '')[:200]}...\n"
        
        qa_text = ""
        if user_qa_pairs:
            qa_text = "\n=== USER Q&A ===\n"
            for qa in user_qa_pairs[-3:]:  # Last 3 Q&A
                qa_text += f"Q: {qa.get('question', '')}\nA: {qa.get('answer', '')}\n"
        
        critique_agent = Agent(
            role="Quality Assurance Analyst",
            goal="Verify the accuracy and completeness of incident resolutions",
            backstory=(
                "You are a senior QA analyst specializing in technical support. "
                "Your job is to critically review diagnosis and solutions before "
                "they are sent to customers. You look for gaps, inconsistencies, "
                "and potential issues that could lead to customer dissatisfaction. "
                "You are thorough and constructive in your feedback."
            ),
            verbose=False,
            allow_delegation=False,
            llm=llm,
            temperature=0
        )
        
        critique_task = Task(
            description=(
                f"Review the following incident resolution for accuracy and completeness:\n\n"
                f"=== INCIDENT DESCRIPTION ===\n{incident_description}\n\n"
                f"=== CURRENT DIAGNOSIS ===\n{diagnosis}\n\n"
                f"=== CURRENT SOLUTION ===\n{solution}\n\n"
                f"{tool_calls_text}\n"
                f"{qa_text}\n\n"
                "Evaluate:\n"
                "1. Does the diagnosis match the incident description?\n"
                "2. Is the solution actionable and complete?\n"
                "3. Are there any gaps or inconsistencies?\n"
                "4. Could this solution cause harm or misunderstanding?\n\n"
                "Output as JSON with this exact format:\n"
                '{"is_valid": true/false, "confidence": 0.0-1.0, '
                '"issues_found": ["issue1", "issue2"], '
                '"suggested_fixes": ["fix1", "fix2"], '
                '"revised_diagnosis": "if different", '
                '"revised_solution": "if different"}'
            ),
            agent=critique_agent,
            expected_output="JSON with validation results"
        )
        
        crew = Crew(
            agents=[critique_agent],
            tasks=[critique_task],
            verbose=False
        )
        
        result = await crew.akickoff()
        
        return _parse_critique_result(str(result))


def _parse_critique_result(result_text: str) -> SelfCritiqueOutput:
    json_match = re.search(r'\{[\s\S]*\}', result_text)
    if json_match:
        try:
            data = json.loads(json_match.group())
            return SelfCritiqueOutput(
                is_valid=data.get("is_valid", True),
                confidence=data.get("confidence", 0.5),
                issues_found=data.get("issues_found", []),
                suggested_fixes=data.get("suggested_fixes", []),
                revised_diagnosis=data.get("revised_diagnosis"),
                revised_solution=data.get("revised_solution")
            )
        except json.JSONDecodeError:
            pass
    
    # Fallback: assume valid if parse fails
    return SelfCritiqueOutput(
        is_valid=True,
        confidence=0.5,
        issues_found=["Failed to parse critique result"],
        suggested_fixes=[]
    )


def should_escalate(critique: SelfCritiqueOutput) -> bool:
    # Escalate if:
    # 1. Confidence is low (< 0.4)
    # 2. Multiple issues found (>= 3)
    # 3. Solution was flagged as invalid
    if not critique.is_valid:
        return True
    if critique.confidence < 0.4:
        return True
    if len(critique.issues_found) >= 3:
        return True
    
    return False
