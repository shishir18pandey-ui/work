import os
import logging
from typing import Dict, Optional
from pydantic import BaseModel, Field

os.environ["OTEL_SDK_DISABLED"] = "true"

from crewai import Agent, Task, Crew, LLM
from new_flow.utils.llm import llm_config
from new_flow.tools.app_config import get_app_config

logger = logging.getLogger(__name__)


class PlanOutput(BaseModel):
    issue_identified: bool = Field(description="True if enough info to proceed")
    issue_summary: str = Field(description="What's the issue from similarity + jaeger")
    jaeger_results: Dict = Field(
        default_factory=dict,
        description="From jaeger traces"
    )
    next_steps: str = Field(description="Guidance for Execute Agent")
    needs_more_info: bool = Field(description="True if need user input")
    question_for_user: Optional[str] = Field(
        default=None,
        description="What to ask user"
    )
    suggested_service: Optional[str] = Field(
        default=None,
        description="Jaeger service to query"
    )
    customer_identifiers: Dict[str, str] = Field(
        default_factory=dict,
        description="All customer identifiers extracted from the incident"
    )


async def run_plan_agent_async(
    enriched_prompt: str,
    app: str,
    customer_identifiers: Dict[str, str],
    problem_category: str,
    incident_context: str = "",
    discovered_services: str = "",
) -> PlanOutput:

    llm = LLM(
        model="openai/" + os.environ.get('OPENAI_MODEL_NAME'),
        temperature=0.0,
        base_url=llm_config.url,
        api_key=llm_config.token
    )

    try:
        app_config = get_app_config(app)
        default_service = app_config.default_jaeger_service or None
    except ValueError:
        default_service = None

    similarity_result = incident_context if incident_context else "No similar incidents found."

    identifiers_text = (
        "\n".join(f"  - {k}: {v}" for k, v in customer_identifiers.items())
        if customer_identifiers else "  None provided"
    )

    analysis_agent = Agent(
        role="Triage Analyst",
        goal="Analyze investigation results and create a plan for resolution",
        backstory=(
            "You are a senior incident analyst. Your job is to analyze "
            "Similarity Search results and the list of available services "
            "to understand what the issue likely is and recommend which "
            "service should be investigated next.\n\n"
            "You must determine:\n"
            "1. Is the issue clear enough from the similarity search context?\n"
            "2. Which service (from the AVAILABLE JAEGER SERVICES list) is "
            "most likely responsible, based on historic incidents and problem category?\n"
            "3. If identifiers or context are missing, what should be asked?"
        ),
        tools=[],
        verbose=True,
        allow_delegation=False,
        llm=llm,
        temperature=0,
        max_iter=3,
        reasoning=True,
        max_reasoning_attempts=3
    )

    services_section = ""
    if discovered_services:
        services_section = f"\n=== AVAILABLE JAEGER SERVICES ===\n{discovered_services}\n"

    analysis_task = Task(
        description=(
            f"Analyze the following investigation results and create a plan:\n\n"
            f"=== INCIDENT ANALYSIS (from Intent Classifier) ===\n{enriched_prompt}\n\n"
            f"=== SIMILARITY SEARCH RESULTS ===\n{similarity_result}\n\n"
            f"=== CONTEXT ===\n"
            f"Application: {app}\n"
            f"Problem Category: {problem_category}\n"
            f"Customer Identifiers:\n{identifiers_text}\n"
            f"{services_section}\n"
            f"Do NOT call any tools. Do not attempt a tool call of any kind. "
            f"Based only on the text above, output your analysis in this exact format:\n\n"
            f"ISSUE_UNDERSTOOD: yes/no\n"
            f"ISSUE_SUMMARY: [brief description of what the issue likely is, based on similarity search]\n"
            f"SERVICE: [pick the single most relevant service name, copied EXACTLY as written "
            f"from the AVAILABLE JAEGER SERVICES list above. If the list is empty or none apply, write NONE]\n"
            f"NEXT_STEPS: [what Execute Agent should investigate]\n"
            f"NEEDS_MORE_INFO: yes/no\n"
            f"QUESTION_FOR_USER: [if needs more info, what to ask]\n"
        ),
        agent=analysis_agent,
        expected_output=(
            "Structured analysis with ISSUE_UNDERSTOOD, ISSUE_SUMMARY, SERVICE, "
            "NEXT_STEPS, NEEDS_MORE_INFO, and QUESTION_FOR_USER fields. No tool calls."
        )
    )

    crew = Crew(
        agents=[analysis_agent],
        tasks=[analysis_task],
        verbose=False
    )

    result = await crew.akickoff()
    result_text = str(result)

    return _parse_plan_output(
        result_text,
        customer_identifiers,
        default_service,
        discovered_services
    )


def _parse_plan_output(
    llm_response: str,
    customer_identifiers: Dict[str, str],
    default_service: Optional[str],
    discovered_services: str = ""
) -> PlanOutput:

    issue_understood = (
        "yes" in llm_response.lower().split("ISSUE_UNDERSTOOD:")[-1].split("\n")[0].lower()
        if "ISSUE_UNDERSTOOD:" in llm_response else "no"
    )

    issue_summary = ""
    if "ISSUE_SUMMARY:" in llm_response:
        summary_part = llm_response.split("ISSUE_SUMMARY:")[-1]
        next_section = summary_part.split("SERVICE:")[0] if "SERVICE:" in summary_part else summary_part
        issue_summary = next_section.strip()

    llm_service = ""
    if "SERVICE:" in llm_response:
        service_part = llm_response.split("SERVICE:")[-1]
        next_part = service_part.split("NEXT_STEPS:")[0] if "NEXT_STEPS:" in service_part else service_part
        llm_service = next_part.strip().split("\n")[0].strip()

    # ── Validate the chosen service against the actual discovered list ──
    # Prevents hallucinated / near-miss / wrong-cased service names from
    # silently slipping through to the executor.
    valid_services = set()
    for line in discovered_services.splitlines():
        line = line.strip()
        if line.startswith("- "):
            valid_services.add(line[2:].strip())

    chosen_service = ""
    if llm_service and llm_service.upper() != "NONE":
        if llm_service in valid_services:
            chosen_service = llm_service
        else:
            lower_map = {s.lower(): s for s in valid_services}
            if llm_service.lower() in lower_map:
                chosen_service = lower_map[llm_service.lower()]
            else:
                logger.warning(
                    f"[PlanAgent] LLM chose service '{llm_service}' not found in "
                    f"discovered list ({len(valid_services)} known) — "
                    f"falling back to default '{default_service}'"
                )

    next_steps = ""
    if "NEXT_STEPS:" in llm_response:
        steps_part = llm_response.split("NEXT_STEPS:")[-1]
        next_section = steps_part.split("NEEDS_MORE_INFO:")[0] if "NEEDS_MORE_INFO:" in steps_part else steps_part
        next_steps = next_section.strip()

    needs_more_info = (
        "yes" in llm_response.lower().split("NEEDS_MORE_INFO:")[-1].split("\n")[0].lower()
        if "NEEDS_MORE_INFO:" in llm_response else False
    )

    question = None
    if "QUESTION_FOR_USER:" in llm_response:
        question = llm_response.split("QUESTION_FOR_USER:")[-1].strip()

    return PlanOutput(
        issue_identified=issue_understood == "yes",
        issue_summary=issue_summary or "Unable to determine issue from initial investigation",
        jaeger_results={},
        next_steps=next_steps or "Investigate via Jaeger traces based on findings",
        needs_more_info=needs_more_info,
        question_for_user=question,
        suggested_service=chosen_service or default_service,
        customer_identifiers=customer_identifiers
    )
