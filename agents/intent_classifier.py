import os
import re
import json
from typing import Dict, List, Optional, Any

from pydantic import BaseModel
from new_flow.utils.llm import llm_config
from crewai import Agent, Task, Crew, LLM
from new_flow.tools.app_config import get_app_config, APPS_CONFIG

import logging

logger = logging.getLogger(__name__)


class IntentClassifierOutput(BaseModel):
    intent: str                              # closure, rebuttal, additional_info
    app: str                                 # cbs, optimus, idp, etc.
    problem_category: str                    # account_freeze, login_failure, etc.
    customer_identifiers: Dict[str, str]     # {ucic, mobile, account, customer_id}
    enriched_prompt: str                     # LLM-ready prompt for downstream
    suggested_approach: str                  # Initial guidance for Plan Agent
    needs_user_input: bool = False
    clarification_question: Optional[str] = None
    user_goal: str = ""
    issue_description: str = ""
    problem_summary: str = ""


class IntentAnalysisResult(BaseModel):
    intent: str = ""
    user_goal: str = ""
    issue_description: str = ""
    problem_summary: str = ""


def extract_identifiers(text: str, payload: Dict) -> Dict[str, str]:
    identifiers = {}
    ucic_patterns = [
        r'\bUCIC[:\s]*(\d{10,12})\b',
        r'\bucic[:\s]*(\d{10,12})\b',
    ]

    for pattern in ucic_patterns:
        match = re.search(pattern, text, re.IGNORECASE)

        if match:
            identifiers["ucic"] = (
                match.group(1)
                if match.lastindex
                else match.group(0)
            )
            break
    loan_account_patterns = [
    r'\bLoan\s*Account\s*Number[:\s]*(\d{6,20})\b',
    r'\bloan_account_number[:\s]*(\d{6,20})\b',
    r'\bloan\s*account[:\s]*(\d{6,20})\b',
    ]  
    for pattern in loan_account_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            identifiers["loan_account_number"] = match.group(1)
            break

    mobile_patterns = [
        r'\b(\+91[6-9]\d{9})\b',
        r'\b(0[6-9]\d{9})\b',
        r'\bmobile[:\s]*(\+91[6-9]\d{9})\b',
        r'\bmobile[:\s]*(0[6-9]\d{9})\b',
    ]

    for pattern in mobile_patterns:
        match = re.search(pattern, text, re.IGNORECASE)

        if match:
            identifiers["mobile"] = match.group(1)
            break

    account_patterns = [
        r'\bAccount[:\s]*(\d{10,12})\b',
        r'\baccount[:\s]*(\d{10,12})\b',
        r'\bAc[\s/-]*(\d{10,12})\b',
    ]

    for pattern in account_patterns:
        match = re.search(pattern, text, re.IGNORECASE)

        if match:
            identifiers["account"] = match.group(1)
            break


    customer_id_patterns = [
        r'\bCUSTOMER\s*ID[:\s]*(\d+)\b',
        r'\bcustomer_id[:\s]*(\d+)\b',
        r'\bCIF[:\s]*(\d+)\b',
    ]

    for pattern in customer_id_patterns:
        match = re.search(pattern, text, re.IGNORECASE)

        if match:
            identifiers["customer_id"] = match.group(1)
            break


    username_patterns = [
        r'\busername[:\s]*(\w+)\b',
        r'\buser[:\s]*(\w+)\b',
    ]

    for pattern in username_patterns:
        match = re.search(pattern, text, re.IGNORECASE)

        if match:
            identifiers["username"] = match.group(1)
            break


    if payload:

        if "ucic" not in identifiers:

            ucic = (
                payload.get("individualUCIC")
                or payload.get("ucic")
            )

            if ucic:
                identifiers["ucic"] = str(ucic)

                # UCIC maps to customer_id in Jaeger
                if "customer_id" not in identifiers:
                    identifiers["customer_id"] = str(ucic)


        if "mobile" not in identifiers:

            mobile = (
                payload.get("mobile_number")
                or payload.get("mobile")
            )

            if mobile:
                identifiers["mobile"] = str(mobile)

        if "loan_account_number" not in identifiers:
            loan_account_number = payload.get("loanAccountNumber")
            if loan_account_number:
                identifiers["loan_account_number"] = str(loan_account_number)

        if "account" not in identifiers:

            account = (
                payload.get("account_number")
                or payload.get("account")
            )

            if account:
                identifiers["account"] = str(account)


        if "customer_id" not in identifiers:

            customer_id = (
                payload.get("customer_id")
                or payload.get("cif")
            )

            if customer_id:
                identifiers["customer_id"] = str(customer_id)


        if "username" not in identifiers:

            username = (
                payload.get("username")
                or payload.get("user_name")
            )

            if username:
                identifiers["username"] = str(username)

    return identifiers


def guess_problem_category(description: str, app: str) -> str:
    desc_lower = description.lower()
    problem_keywords = {
        "account_freeze": [
            "freeze",
            "blocked",
            "suspended",
            "hold",
        ],
        "transaction_failure": [
            "transaction",
            "transfer",
            "payment",
            "failed",
        ],
        "balance_issue": [
            "balance",
            "balance missing",
            "incorrect balance",
        ],
        "login_failure": [
            "login",
            "cannot login",
            "password",
            "authentication",
        ],
        "session_timeout": [
            "session",
            "timeout",
            "logged out",
        ],
        "mfa_issue": [
            "mfa",
            "otp",
            "two-factor",
            "authentication code",
        ],
        "password_reset": [
            "password reset",
            "forgot password",
        ],
        "loan_issue": [
            "loan",
            "emi",
            "repayment",
        ],
        "kyc_issue": [
            "kyc",
            "verification",
            "documents",
        ],
    }

    for category, keywords in problem_keywords.items():
        for keyword in keywords:
            if keyword in desc_lower:
                return category
    return "application_issue"

def get_app_from_payload(payload: Dict) -> str:

    businessService = payload.get("businessService", "").lower().strip()

    if businessService in APPS_CONFIG:
        return businessService

    return ""

def parse_intent_result(result_str: str) -> IntentAnalysisResult:

    json_match = re.search(
        r'\{.*\}',
        result_str,
        re.DOTALL
    )
    if json_match:
        try:
            data = json.loads(json_match.group())
            return IntentAnalysisResult(
                intent=str(
                    data.get("intent", "")
                ).strip().lower(),

                user_goal=str(
                    data.get("user_goal", "")
                ).strip(),

                issue_description=str(
                    data.get("issue_description", "")
                ).strip(),

                problem_summary=str(
                    data.get("problem_summary", "")
                ).strip(),
            )

        except (json.JSONDecodeError, TypeError, AttributeError):
            pass

    try:

        data = json.loads(result_str)

        return IntentAnalysisResult(
            intent=str(
                data.get("intent", "")
            ).strip().lower(),

            user_goal=str(
                data.get("user_goal", "")
            ).strip(),

            issue_description=str(
                data.get("issue_description", "")
            ).strip(),

            problem_summary=str(
                data.get("problem_summary", "")
            ).strip(),
        )
    except (json.JSONDecodeError, TypeError, AttributeError):
        pass

    result_lower = result_str.lower()

    intent = "additional_info"

    if "closure" in result_lower:
        intent = "closure"

    elif "rebuttal" in result_lower:
        intent = "rebuttal"

    return IntentAnalysisResult(
        intent=intent,
        user_goal="",
        issue_description=result_str.strip(),
        problem_summary=result_str.strip(),
    )


async def run_intent_classifier_crew_async(
    incident_description: str,
    history: list,
    interaction: str,
    previous_classifier_output: dict = None
) -> IntentAnalysisResult:

    if not history:
        history = ["NA"]

    previous_analysis = "None"
    if previous_classifier_output:
        prev_user_goal = previous_classifier_output.get("user_goal", "")
        prev_issue_desc = previous_classifier_output.get("issue_description", "")
        prev_prob_summary = previous_classifier_output.get("problem_summary", "")
        
        if prev_user_goal or prev_issue_desc or prev_prob_summary:
            previous_analysis = f"""- user_goal: {prev_user_goal}
- issue_description: {prev_issue_desc}
- problem_summary: {prev_prob_summary}"""


    api_key = llm_config.token
    logger.info(f"[TOKEN CHECK] using token ending in {api_key[-8:] if api_key else 'NONE'}")
    if not api_key:
        import new_flow.utils.llm as llm_module
        logger.error(
            f"[TOKEN DEBUG] llm_config id={id(llm_config)} "
            f"module llm_config id={id(llm_module.llm_config)} "
            f"same_object={llm_config is llm_module.llm_config} "
            f"token_repr={llm_config.token!r}"
    )
    llm = LLM(
        model="openai//app/models/Qwen3-14B-FP8",
        temperature=0.0,
        base_url=(
            "https://llm-api.iservebetter.idfcfirstbank.com/qwen3-14b-entauth/v1"
        ),
        api_key=api_key,
    )
    print("=== Intent Classifier LLM ===")
    print(
        "Model: openai//app/models/Qwen3-14B-FP8"
    )
    print(
        "Base URL: "
        "https://llm-api.iservebetter.idfcfirstbank.com/"
        "qwen3-14b-entauth/v1"
    )
    intent_agent = Agent(

        role="Intent Classifier",
        goal=(
            "Analyze user input and categorize it into "
            "the correct intent category with detailed analysis"
        ),

        backstory=(
            "You are an expert at classifying user intents "
            "in a technical support system. "
            "Your job is to analyze the conversation history "
            "and current user input to determine what the user "
            "is trying to accomplish. "
            "Users may not always express themselves clearly, "
            "so you must infer and articulate what they actually want."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
        temperature=0,
        max_iter=2,
        reasoning=False,
        max_retry_limit=2,
    )

    intent_task = Task(
        description=(
            "Analyze the user input and categorize it "
            "with detailed analysis.\n\n"
            "Interaction History:\n"
            "```\n"
            "{history}\n"
            "```\n\n"
            "User Input:\n"
            "```\n"
            "{incident_description}\n"
            "```\n\n"
            "Latest Interaction:\n"
            "```\n"
            "{interaction}\n"
            "```\n\n"
            "Previous Analysis (USE AS BASE to refine, if available):\n"
            "```\n"
            "{previous_analysis}\n"
            "```\n\n"
            "IMPORTANT - CLASSIFICATION RULES (MUST FOLLOW EXACTLY):\n"
            "- If history shows bot asked a question AND user provides info "
            "(UCIC, account, mobile, etc.) → intent is 'additional_info'\n"
            "- If history shows bot gave a solution AND user disagrees/rejects "
            "→ intent is 'rebuttal'\n"
            "- If history shows bot gave a solution AND user thanks/expresses "
            "satisfaction → intent is 'closure'\n"
            "- If no prior bot action AND user provides info → intent is "
            "'additional_info'\n\n"
            
            "ANTI-HALLUCINATION RULES:\n"
            "- Only analyze what is EXPLICITLY STATED in user input\n"
            "- Do NOT infer, invent, or fabricate emotions, frustrations, or "
            "situations not present in the actual text\n"
            "- If user provides requested information → intent is 'additional_info' "
            "(even if user seems frustrated)\n\n"
            
            "FRUSTRATION RULE:\n"
            "- Even if user expresses frustration, anger, or impatience, if they "
            "provide information in response to a question → intent is "
            "'additional_info'\n"
            "- Frustration is metadata, NOT intent\n"
            "- Only classify as 'rebuttal' if user explicitly rejects, corrects, or "
            "challenges what the bot SAID or DID\n\n"

            "If previous analysis exists:\n"
            "- Use it as BASE and only refine/update based on new user input\n"
            "- Keep what hasn't changed, update only what is new\n\n"
            
            "If NO previous analysis (first interaction):\n"
            "- Generate from scratch using ONLY what is explicitly stated\n"
            "- Do NOT invent, infer, or fabricate any information\n\n"

            "**Categories**:\n\n"

            "- **closure**: "
            "Greeting, thanks, or ending the chat.\n\n"

            "- **rebuttal**: "
            "Use when there is a previous bot interaction AND the user explicitly "
            "contradicts, corrects, rejects, or challenges something the bot previously "
            "said or did.\n\n"

            "- **additional_info**: "
            "Providing IDs, account numbers or subsequent question/information asked.\n\n"

            "Output your response as valid JSON with "
            "the following structure:\n"

            "{\n"

            '  "intent": "<category>",\n'

            '  "user_goal": "<refined or new - based only on explicit user input>",\n'

            '  "issue_description": "<refined or new - based only on explicit user input>",\n'

            '  "problem_summary": "<refined or new - based only on explicit user input>"\n'

            "}\n"

            "Do not add any text before or after the JSON."
        ),
        agent=intent_agent,

        expected_output=(
            "Valid JSON with intent "
            "(closure/rebuttal/additional_info), "
            "user_goal, issue_description, "
            "and problem_summary fields."
        ),
    )
    crew = Crew(
        agents=[intent_agent],
        tasks=[intent_task],
        verbose=True,
    )
    result = await crew.akickoff(

        inputs={
            "incident_description": incident_description,
            "history": history,
            "interaction": interaction,
            "previous_analysis":previous_analysis
        }
    )
    result_str = str(result)

    return parse_intent_result(result_str)

async def run_classifier_with_enrichment_async(
    payload: Dict,
    incident_description: str,
    user_qa_pairs: List[Dict] = None,
    comment: str = None,
    previous_classifier_output: dict = None
) -> IntentClassifierOutput:

  
    identifiers = extract_identifiers(
        incident_description,
        payload
    )

    app = get_app_from_payload(payload)

    try:

        app_config = get_app_config(app)

    except ValueError:

        app_config = None

    problem_category = guess_problem_category(
        incident_description,
        app
    )

    needs_user_input = False
    clarification_question = None

    # if not identifiers:

        # needs_user_input = True

        # clarification_question = (
        #     "Could you please provide one of: "
        #     "UCIC, Mobile Number, or Account Number "
        #     "to help investigate this issue?"
        # )

    enriched_prompt = f"""
Incident Summary:
- Application: {app_config.name if app_config else app}
- Problem Category: {problem_category}

Customer Identifiers:
{
    chr(10).join(
        f"- {k}: {v}"
        for k, v in identifiers.items()
    )
    if identifiers
    else "- Not provided"
}

Description:
{incident_description}

Please investigate this issue starting with Similarity Search
for similar historic incidents, then check Jaeger traces to
understand the current error context.
"""

 
    suggested_approach = (
        f"1. First, use Similarity Search to find similar "
        f"resolved {problem_category} incidents for the "
        f"{app} application.\n"

        f"2. Then, fetch Jaeger traces using the customer "
        f"identifier to understand current errors.\n"

        f"3. Based on findings, decide whether to query ELK "
        f"for detailed logs or DB for account data."
    )

    interaction = comment if comment else "NA"

    intent_result = await run_intent_classifier_crew_async(

        incident_description,

        [
            str(qa)
            for qa in (user_qa_pairs or [])
        ],

        interaction,
        previous_classifier_output=previous_classifier_output,
    )
    user_goal = (
        intent_result.user_goal
        if intent_result.user_goal
        else ""
    )

    issue_description = (
        intent_result.issue_description
        if intent_result.issue_description
        else incident_description
    )

    problem_summary = (
        intent_result.problem_summary
        if intent_result.problem_summary
        else f"User reported: {incident_description}"
    )


    if (
        intent_result.user_goal
        or intent_result.problem_summary
    ):

        enriched_prompt = f"""
Incident Summary:
- Application: {app_config.name if app_config else app}
- Problem Category: {problem_category}

Customer Identifiers:
{
    chr(10).join(
        f"- {k}: {v}"
        for k, v in identifiers.items()
    )
    if identifiers
    else "- Not provided"
}

LLM Analysis:
- User Goal: {user_goal}
- Issue Description: {issue_description}
- Problem Summary: {problem_summary}

Original Description:
{incident_description}

Please investigate this issue starting with Similarity Search
for similar historic incidents, then check Jaeger traces to
understand the current error context.
"""
    return IntentClassifierOutput(

        intent=(
            intent_result.intent
            if intent_result.intent
            else "additional_info"
        ),
        app=app,
        problem_category=problem_category,
        customer_identifiers=identifiers,
        enriched_prompt=enriched_prompt,
        suggested_approach=suggested_approach,
        needs_user_input=needs_user_input,
        clarification_question=clarification_question,
        user_goal=user_goal,
        issue_description=issue_description,
        problem_summary=problem_summary,
    )
