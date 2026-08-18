import os
import re
from typing import Dict, List
from pydantic import BaseModel, Field

os.environ["OTEL_SDK_DISABLED"] = "true"

from crewai import Agent, Task, Crew, LLM
from new_flow.utils.llm import llm_config
from new_flow.agents.execute_agent_jaeger import JaegerExecutionResult
from new_flow.agents.plan_agents import PlanOutput



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
    execution_result: JaegerExecutionResult,
    historic_context: str = "",
    user_qa_pairs: List[Dict] = None
) -> SummaryOutput:
    llm = LLM(
        model="openai/" + llm_config.model_name,
        temperature=0.0,
        base_url=llm_config.url,
        api_key=llm_config.token
    )

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

    if execution_result.resolved:
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
    
    resolved = "yes" in result_text.lower().split("resolved")[-1].split("}")[0].lower() if "resolved" in result_text.lower() else "no"
    
    return SummaryOutput(
        diagnosis=result_text[:300],
        solution="See diagnosis",
        resolved=resolved
    )



def create_simple_summary(
    plan_output: PlanOutput,
    execution_result: JaegerExecutionResult
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

    diagnosis = execution_result.diagnosis or plan_output.issue_summary or "Investigation incomplete"
    solution = execution_result.solution or "Manual investigation required"

    return SummaryOutput(
        diagnosis=diagnosis,
        solution=solution,
        questions=[],
        resolved="no"
    )


async def run_context_only_summary_async(
    incident_description: str,
    historic_context: str = "",
    user_qa_pairs: List[Dict] = None
) -> SummaryOutput:
    """
    Used when the app has no Jaeger/ELK config. Resolves purely from
    similarity-search historic context, since no live investigation is possible.
    """
    llm = LLM(
        model="openai/" + llm_config.model_name,
        temperature=0.0,
        base_url=llm_config.url,
        api_key=llm_config.token
    )

    qa_text = ""
    if user_qa_pairs:
        qa_text = "\n=== USER Q&A ===\n"
        for qa in user_qa_pairs:
            qa_text += f"Q: {qa.get('question', '')}\nA: {qa.get('answer', '')}\n"

    agent = Agent(
        role="L1/L2 Bank Support Engineer",
        goal="Resolve or clarify a customer issue using only historic incident precedent",
        backstory=(
            "You are a senior bank support engineer. No live system logs are available "
            "for this application, so you must rely only on similar past incidents. "
            "IMPORTANT: Never mention technical tools like Jaeger, ELK, or database queries. "
            "Explain things in simple terms a branch employee can understand."
        ),
        verbose=False,
        allow_delegation=False,
        llm=llm,
        temperature=0
    )

    task = Task(
        description=(
            f"=== CURRENT INCIDENT ===\n{incident_description}\n\n"
            f"=== SIMILAR HISTORIC INCIDENTS ===\n{historic_context}\n\n"
            f"=== USER Q&A ===\n{qa_text}\n\n"
            "No live logs are available for this application. Decide:\n"
            "1. If a similar historic incident clearly matches this one AND has a clear "
            "resolution, use RESOLVED=yes and state that resolution as the diagnosis/solution.\n"
            "2. Otherwise, RESOLVED=no and ask a specific clarifying question that would help "
            "narrow down which historic case applies.\n\n"
            "Do not mention any technical tools.\n\n"
            'Format the output as JSON: {"diagnosis": "...", "solution": "...", "questions": [], "resolved": "yes/no"}'
        ),
        agent=agent,
        expected_output="JSON with diagnosis, solution, questions, and resolved fields"
    )

    crew = Crew(agents=[agent], tasks=[task], verbose=False)
    result = await crew.akickoff()
    return _parse_summary_result(str(result))
