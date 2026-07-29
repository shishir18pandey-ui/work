import os
import re
from typing import Dict, List
from pydantic import BaseModel, Field

os.environ["OTEL_SDK_DISABLED"] = "true"

from crewai import Agent, Task, Crew, LLM
from utils.llm import llm_config
from agents.plan_agent import PlanOutput
from agents.execute_agent import ExecutionResult



class SummaryOutput(BaseModel):
    diagnosis: str = Field(description="Root cause analysis")
    solution: str = Field(description="Resolution steps")
    questions: List[str] = Field(
        default_factory=list,
        description="Clarification questions if needed"
    )
    resolved: str = Field(description="yes/no")


async def run_summary_agent_async(
    incident_description: str,
    execution_result: ExecutionResult,
    historic_context: str = "",
    user_qa_pairs: List[Dict] = None
) -> SummaryOutput:
    llm = LLM(
        model="openai/" + llm_config.model_name,
        temperature=0.0,
        base_url=llm_config.url,
        api_key=llm_config.token
    )

    # Build all context
    tool_calls_text = ""
    if execution_result.tool_calls:
        tool_calls_text = "\n=== TOOL CALLS MADE ===\n"
        for call in execution_result.tool_calls:
            tool_calls_text += f"\n{call['tool_name']}:\n{call['output']}\n"

    qa_text = ""
    if user_qa_pairs:
        qa_text = "\n=== USER Q&A ===\n"
        for qa in user_qa_pairs:
            qa_text += f"Q: {qa.get('question', '')}\nA: {qa.get('answer', '')}\n"

    # If execution already has a result, use it
    if execution_result.resolved:
        # Try to enhance the result
        summary_agent = Agent(
            role="L1/L2 Bank Support Engineer",
            goal="Create a customer-friendly resolution response",
            backstory=(
                "You are a senior bank support engineer responding to a customer issue. "
                "Your response should be professional, clear, and actionable. "
                "IMPORTANT: Never mention technical tools like Jaeger, ELK, database queries, or any debugging tools. "
                "Explain the issue and solution in simple terms that a branch employee can understand and communicate to the customer."
            ),
            verbose=False,
            allow_delegation=False,
            llm=llm,
            temperature=0
        )

        format_task = Task(
            description=(
                f"Create a final resolution for a bank customer issue.\n\n"
                f"=== INCIDENT ===\n{incident_description}\n\n"
                f"=== INVESTIGATION FINDINGS ===\n"
                f"Root Cause: {execution_result.diagnosis}\n"
                f"Resolution: {execution_result.solution}\n"
                f"{tool_calls_text}\n\n"
                f"=== HISTORIC SIMILAR INCIDENTS ===\n{historic_context}\n\n"
                f"=== USER Q&A ===\n{qa_text}\n\n"
                "IMPORTANT: Write your response as a bank support engineer would speak to a branch employee. "
                "Do NOT mention any technical tools (Jaeger, ELK, database, etc.). "
                "Use simple, clear language. Explain what happened and what action the customer needs to take.\n\n"
                f"Format the output as JSON:\n"
                f'{{"diagnosis": "...", "solution": "...", "questions": [], "resolved": "yes"}}'
            ),
            agent=summary_agent,
            expected_output="JSON with diagnosis, solution, questions, and resolved fields"
        )

        crew = Crew(
            agents=[summary_agent],
            tasks=[format_task],
            verbose=False
        )

        result = await crew.akickoff()
        return _parse_summary_result(str(result))

    diagnosis = execution_result.diagnosis if execution_result.diagnosis else "Issue not fully resolved"
    solution = execution_result.solution if execution_result.solution else "Manual investigation required"
    questions = execution_result.questions if execution_result.questions else []

    return SummaryOutput(
        diagnosis=diagnosis,
        solution=solution,
        questions=questions,
        resolved="yes" if execution_result.resolved else "no"
    )


def _parse_summary_result(result_text: str) -> SummaryOutput:
    """Parse LLM response into SummaryOutput."""
    import json
    
    # Try to find JSON
    json_match = re.search(r'\{[\s\S]*\}', result_text)
    if json_match:
        try:
            data = json.loads(json_match.group())
            return SummaryOutput(
                diagnosis=data.get("diagnosis", ""),
                solution=data.get("solution", ""),
                questions=data.get("questions", []),
                resolved=data.get("resolved", "no")
            )
        except:
            pass
    
    # Fallback: extract from text
    resolved = "yes" in result_text.lower().split("resolved")[-1].split("}")[0].lower() if "resolved" in result_text.lower() else "no"
    
    return SummaryOutput(
        diagnosis=result_text[:300],
        solution="See diagnosis",
        resolved=resolved
    )



def create_simple_summary(
    plan_output: PlanOutput,
    execution_result: ExecutionResult
) -> SummaryOutput:
    if execution_result.resolved:
        return SummaryOutput(
            diagnosis=execution_result.diagnosis,
            solution=execution_result.solution,
            questions=execution_result.questions,
            resolved="yes"
        )

    if plan_output.needs_more_info:
        return SummaryOutput(
            diagnosis="Additional information needed",
            solution="Waiting for user response",
            questions=[plan_output.question_for_user or "Please provide more details"],
            resolved="no"
        )

    # Default - use execution result or plan output
    diagnosis = execution_result.diagnosis or plan_output.issue_summary or "Investigation incomplete"
    solution = execution_result.solution or "Manual investigation required"

    return SummaryOutput(
        diagnosis=diagnosis,
        solution=solution,
        questions=[],
        resolved="no"
    )
