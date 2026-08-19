2026-08-19 08:17:03,408 - __main__ - INFO - ✓ Flow completed | incident=77773632
2026-08-19 08:24:13,944 - new_flow.utils.llm - INFO - OPENAI_API_KEY refreshed
2026-08-19 08:34:13,986 - new_flow.utils.llm - INFO - OPENAI_API_KEY refreshed
2026-08-19 08:38:04,728 - __main__ - INFO - → Message received | incident=66773632 event=new_incident partition=1 offset=631
2026-08-19 08:38:04,728 - __main__ - INFO - ✓ Offset committed | incident=66773632
2026-08-19 08:38:04,728 - __main__ - INFO - → Processing | incident=66773632 event=new_incident module=new_flow.flow
2026-08-19 08:38:04,760 - __main__ - INFO -   DB status | incident=66773632 status=in_progress
2026-08-19 08:38:04,765 - __main__ - INFO -   Flow type | incident=66773632 type=new_incident
╭───────────────────────────── 🌊 Flow Execution ──────────────────────────────╮
│                                                                              │
│  Starting Flow Execution                                                     │
│  Name: IncidentManagementFlow                                                │
│  ID: 681e0fea-7309-4d18-95a6-3806a126ae6e                                    │
│                                                                              │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯

╭────────────────────────────── 🌊 Flow Started ───────────────────────────────╮
│                                                                              │
│  Flow Started                                                                │
│  Name: IncidentManagementFlow                                                │
│  ID: 681e0fea-7309-4d18-95a6-3806a126ae6e                                    │
│                                                                              │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯

Flow started with ID: 681e0fea-7309-4d18-95a6-3806a126ae6e
2026-08-19 08:38:20,794 - crewai.flow.flow - INFO - Flow started with ID: 681e0fea-7309-4d18-95a6-3806a126ae6e
2026-08-19 08:38:20,795 - new_flow.agents.intent_classifier - INFO - [TOKEN CHECK] using token ending in -nndvDhM
Generated incident description for UCIC 
=== Intent Classifier LLM ===
Model: openai//app/models/Qwen3-14B-FP8
Base URL: https://llm-api.iservebetter.idfcfirstbank.com/qwen3-14b-entauth/v1
╭─────────────────────────── 🔄 Flow Method Running ───────────────────────────╮
│                                                                              │
│  Method: initialize_and_classify                                             │
│  Status: Running                                                             │
│                                                                              │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯

╭───────────────────────── 🚀 Crew Execution Started ──────────────────────────╮
│                                                                              │
│  Crew Execution Started                                                      │
│  Name: crew                                                                  │
│  ID: a289acce-8f77-43b1-a1b9-73470249a8aa                                    │
│                                                                              │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯

08:38:36 - LiteLLM:INFO: utils.py:3427 - 
LiteLLM completion() model= /app/models/Qwen3-14B-FP8; provider = openai
2026-08-19 08:38:36,846 - LiteLLM - INFO - 
LiteLLM completion() model= /app/models/Qwen3-14B-FP8; provider = openai
╭────────────────────────────── 📋 Task Started ───────────────────────────────╮
│                                                                              │
│  Task Started                                                                │
│  Name: Analyze the user input and categorize it with detailed analysis.      │
│                                                                              │
│  Interaction History:                                                        │
│  ```                                                                         │
│  ['NA']                                                                      │
│  ```                                                                         │
│                                                                              │
│  User Input:                                                                 │
│  ```                                                                         │
│  Short Description: <script>alert('XSS')</script> Customer is unable to do   │
│  verification                                                                │
│  Description: Customer is unable to do verificstion                          │
│  ```                                                                         │
│                                                                              │
│  Latest Interaction:                                                         │
│  ```                                                                         │
│  NA                                                                          │
│  ```                                                                         │
│                                                                              │
│  Depending on history you can figure out the latest question if it exists.   │
│  Categorise intent on the basis of `Latest Interaction`. If `Latest          │
│  Interaction` is NA, classify on the basis of `User Input`.                  │
│                                                                              │
│  **Categories**:                                                             │
│                                                                              │
│  - **closure**: Greeting, thanks, or ending the chat.                        │
│                                                                              │
│  - **rebuttal**: User is disagreeing, correcting the system, or insisting    │
│  that information they previously provided is correct (e.g. 'I already told  │
│  you', 'That's wrong', 'This is correct.').                                  │
│                                                                              │
│  - **additional_info**: Providing IDs, account numbers or subsequent         │
│  question/information asked.                                                 │
│                                                                              │
│  **Rule**: If the User Input contradicts the Latest Interaction or           │
│  expresses frustration with the system's request, it MUST be 'rebuttal'.     │
│                                                                              │
│  **Analysis Required**:                                                      │
│  Based on the user input and history, provide:                               │
│  1. `user_goal`: What the user is trying to achieve or accomplish.           │
│  2. `issue_description`: The specific problem or issue they're facing.       │
│  3. `problem_summary`: A detailed write-up explaining the issue in context.  │
│                                                                              │
│  Output your response as valid JSON with the following structure:            │
│  {                                                                           │
│    "intent": "<category>",                                                   │
│    "user_goal": "<what user wants to accomplish>",                           │
│    "issue_description": "<specific problem>",                                │
│    "problem_summary": "<detailed write-up>"                                  │
│  }                                                                           │
│  Do not add any text before or after the JSON.                               │
│  ID: 00a72cf5-2d7c-4fcf-b105-76bc6853d30f                                    │
│                                                                              │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯
╭────────────────────────────── 🤖 Agent Started ──────────────────────────────╮
│                                                                              │
│  Agent: Intent Classifier                                                    │
│                                                                              │
│  Task: Analyze the user input and categorize it with detailed analysis.      │
│                                                                              │
│  Interaction History:                                                        │
│  ```                                                                         │
│  ['NA']                                                                      │
│  ```                                                                         │
│                                                                              │
│  User Input:                                                                 │
│  ```                                                                         │
│  Short Description: <script>alert('XSS')</script> Customer is unable to do   │
│  verification                                                                │
│  Description: Customer is unable to do verificstion                          │
│  ```                                                                         │
│                                                                              │
│  Latest Interaction:                                                         │
│  ```                                                                         │
│  NA                                                                          │
│  ```                                                                         │
│                                                                              │
│  Depending on history you can figure out the latest question if it exists.   │
│  Categorise intent on the basis of `Latest Interaction`. If `Latest          │
│  Interaction` is NA, classify on the basis of `User Input`.                  │
│                                                                              │
│  **Categories**:                                                             │
│                                                                              │
│  - **closure**: Greeting, thanks, or ending the chat.                        │
│                                                                              │
│  - **rebuttal**: User is disagreeing, correcting the system, or insisting    │
│  that information they previously provided is correct (e.g. 'I already told  │
│  you', 'That's wrong', 'This is correct.').                                  │
│                                                                              │
│  - **additional_info**: Providing IDs, account numbers or subsequent         │
│  question/information asked.                                                 │
│                                                                              │
│  **Rule**: If the User Input contradicts the Latest Interaction or           │
│  expresses frustration with the system's request, it MUST be 'rebuttal'.     │
│                                                                              │
│  **Analysis Required**:                                                      │
│  Based on the user input and history, provide:                               │
│  1. `user_goal`: What the user is trying to achieve or accomplish.           │
│  2. `issue_description`: The specific problem or issue they're facing.       │
│  3. `problem_summary`: A detailed write-up explaining the issue in context.  │
│                                                                              │
│  Output your response as valid JSON with the following structure:            │
│  {                                                                           │
│    "intent": "<category>",                                                   │
│    "user_goal": "<what user wants to accomplish>",                           │
│    "issue_description": "<specific problem>",                                │
│    "problem_summary": "<detailed write-up>"                                  │
│  }                                                                           │
│  Do not add any text before or after the JSON.                               │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯


╭─────────────────────────── ✅ Agent Final Answer ────────────────────────────╮
│                                                                              │
│  Agent: Intent Classifier                                                    │
│                                                                              │
│  Final Answer:                                                               │
│  {                                                                           │
│    "intent": "additional_info",                                              │
│    "user_goal": "To report an issue with verification functionality on the   │
│  platform.",                                                                 │
│    "issue_description": "Customer is unable to complete verification due to  │
│  an issue involving the execution of a script, possibly related to XSS       │
│  (Cross-Site Scripting).",                                                   │
│    "problem_summary": "The user has reported that a customer is unable to    │
│  perform verification on the platform. The issue is described with a script  │
│  tag containing an alert for XSS, which may be causing the verification      │
│  process to fail. The user input also contains a typo in the word            │
│  'verification' (spelled as 'verificstion'), but the intent is clear: the    │
│  customer is encountering a problem with the verification process,           │
│  potentially due to an XSS-related issue."                                   │
│  }                                                                           │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯

╭───────────────────────────── 📋 Task Completion ─────────────────────────────╮
│                                                                              │
│  Task Completed                                                              │
│  Name: Analyze the user input and categorize it with detailed analysis.      │
│                                                                              │
│  Interaction History:                                                        │
│  ```                                                                         │
│  ['NA']                                                                      │
│  ```                                                                         │
│                                                                              │
│  User Input:                                                                 │
│  ```                                                                         │
│  Short Description: <script>alert('XSS')</script> Customer is unable to do   │
│  verification                                                                │
│  Description: Customer is unable to do verificstion                          │
│  ```                                                                         │
│                                                                              │
│  Latest Interaction:                                                         │
│  ```                                                                         │
│  NA                                                                          │
│  ```                                                                         │
│                                                                              │
│  Depending on history you can figure out the latest question if it exists.   │
│  Categorise intent on the basis of `Latest Interaction`. If `Latest          │
│  Interaction` is NA, classify on the basis of `User Input`.                  │
│                                                                              │
│  **Categories**:                                                             │
│                                                                              │
│  - **closure**: Greeting, thanks, or ending the chat.                        │
│                                                                              │
│  - **rebuttal**: User is disagreeing, correcting the system, or insisting    │
│  that information they previously provided is correct (e.g. 'I already told  │
│  you', 'That's wrong', 'This is correct.').                                  │
│                                                                              │
│  - **additional_info**: Providing IDs, account numbers or subsequent         │
│  question/information asked.                                                 │
│                                                                              │
│  **Rule**: If the User Input contradicts the Latest Interaction or           │
│  expresses frustration with the system's request, it MUST be 'rebuttal'.     │
│                                                                              │
│  **Analysis Required**:                                                      │
│  Based on the user input and history, provide:                               │
│  1. `user_goal`: What the user is trying to achieve or accomplish.           │
│  2. `issue_description`: The specific problem or issue they're facing.       │
│  3. `problem_summary`: A detailed write-up explaining the issue in context.  │
│                                                                              │
│  Output your response as valid JSON with the following structure:            │
│  {                                                                           │
│    "intent": "<category>",                                                   │
│    "user_goal": "<what user wants to accomplish>",                           │
│    "issue_description": "<specific problem>",                                │
│    "problem_summary": "<detailed write-up>"                                  │
│  }                                                                           │
│  Do not add any text before or after the JSON.                               │
│  Agent: Intent Classifier                                                    │
│                                                                              │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯

╭────────────────────────────── Crew Completion ───────────────────────────────╮
│                                                                              │
│  Crew Execution Completed                                                    │
│  Name: crew                                                                  │
│  ID: a289acce-8f77-43b1-a1b9-73470249a8aa                                    │
│  Final Output: {                                                             │
│    "intent": "additional_info",                                              │
│    "user_goal": "To report an issue with verification functionality on the   │
│  platform.",                                                                 │
│    "issue_description": "Customer is unable to complete verification due to  │
│  an issue involving the execution of a script, possibly related to XSS       │
│  (Cross-Site Scripting).",                                                   │
│    "problem_summary": "The user has reported that a customer is unable to    │
│  perform verification on the platform. The issue is described with a script  │
│  tag containing an alert for XSS, which may be causing the verification      │
│  process to fail. The user input also contains a typo in the word            │
│  'verification' (spelled as 'verificstion'), but the intent is clear: the    │
│  customer is encountering a problem with the verification process,           │
│  potentially due to an XSS-related issue."                                   │
│  }                                                                           │
│                                                                              │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯

2026-08-19 08:38:38,533 - opentelemetry.attributes - WARNING - Invalid type NoneType for attribute 'sop_value' value. Expected one of ['bool', 'str', 'bytes', 'int', 'float'] or a sequence of those types
Enhanced classifier | incident=66773632 | app=sfdc asset org 3 | category=kyc_issue
Initialized and classified incident 66773632 with intent: additional_info
Need more info for incident 66773632
╭────────────────────────── ✅ Flow Method Completed ──────────────────────────╮
│                                                                              │
│  Method: initialize_and_classify                                             │
│  Status: Completed                                                           │
│                                                                              │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯

╭─────────────────────────── 🔄 Flow Method Running ───────────────────────────╮
│                                                                              │
│  Method: start_process                                                       │
│  Status: Running                                                             │
│                                                                              │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯
╭────────────────────────── ✅ Flow Method Completed ──────────────────────────╮
│                                                                              │
│  Method: start_process                                                       │
│  Status: Completed                                                           │
│                                                                              │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯


╭───────────────────────────── ✅ Flow Completion ─────────────────────────────╮
│                                                                              │
│  Flow Execution Completed                                                    │
│  Name: IncidentManagementFlow                                                │
│  ID: 681e0fea-7309-4d18-95a6-3806a126ae6e                                    │
│                                                                              │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯



╭────────────────────────── Tracing Preference Saved ──────────────────────────╮
│                                                                              │
│  Info: Tracing has been disabled.                                            │
│                                                                              │
│  Your preference has been saved. Future Crew/Flow executions will not        │
│  collect traces.                                                             │
│                                                                              │
│  To enable tracing later, do any one of these:                               │
│  • Set tracing=True in your Crew/Flow code                                   │
│  • Set CREWAI_TRACING_ENABLED=true in your project's .env file               │
│  • Run: crewai traces enable                                                 │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯

2026-08-19 08:38:38,541 - __main__ - INFO - ✓ Flow completed | incident=66773632
shishir.pandey_tho@0325LTPB0124444 ~ % 2
this si inteent classifier
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
    """Structured output from the intent classifier LLM"""

    intent: str = ""
    user_goal: str = ""
    issue_description: str = ""
    problem_summary: str = ""


# ============================================================
# IDENTIFIER EXTRACTION
# ============================================================

def extract_identifiers(text: str, payload: Dict) -> Dict[str, str]:
    identifiers = {}

    # --------------------------------------------------------
    # UCIC
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # MOBILE
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # ACCOUNT
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # CUSTOMER ID / CIF
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # USERNAME
    # --------------------------------------------------------

    username_patterns = [
        r'\busername[:\s]*(\w+)\b',
        r'\buser[:\s]*(\w+)\b',
    ]

    for pattern in username_patterns:
        match = re.search(pattern, text, re.IGNORECASE)

        if match:
            identifiers["username"] = match.group(1)
            break

    # ========================================================
    # SERVICE NOW PAYLOAD
    # ========================================================

    if payload:

        # ----------------------------------------------------
        # UCIC
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # MOBILE
        # ----------------------------------------------------

        if "mobile" not in identifiers:

            mobile = (
                payload.get("mobile_number")
                or payload.get("mobile")
            )

            if mobile:
                identifiers["mobile"] = str(mobile)

        # ----------------------------------------------------
        # ACCOUNT
        # ----------------------------------------------------

        if "account" not in identifiers:

            account = (
                payload.get("account_number")
                or payload.get("account")
            )

            if account:
                identifiers["account"] = str(account)

        # ----------------------------------------------------
        # CUSTOMER ID
        # ----------------------------------------------------

        if "customer_id" not in identifiers:

            customer_id = (
                payload.get("customer_id")
                or payload.get("cif")
            )

            if customer_id:
                identifiers["customer_id"] = str(customer_id)

        # ----------------------------------------------------
        # USERNAME
        # ----------------------------------------------------

        if "username" not in identifiers:

            username = (
                payload.get("username")
                or payload.get("user_name")
            )

            if username:
                identifiers["username"] = str(username)

    return identifiers


# ============================================================
# PROBLEM CATEGORY
# ============================================================

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


# ============================================================
# APPLICATION
# ============================================================

def get_app_from_payload(payload: Dict) -> str:

    businessService = payload.get("businessService", "").lower().strip()

    if businessService in APPS_CONFIG:
        return businessService

    return ""


# ============================================================
# PARSE LLM RESPONSE
# ============================================================

def parse_intent_result(result_str: str) -> IntentAnalysisResult:

    # --------------------------------------------------------
    # First try to find JSON inside the response.
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Try entire response as JSON.
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

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


# ============================================================
# INTENT CLASSIFIER
# ============================================================

async def run_intent_classifier_crew_async(
    incident_description: str,
    history: list,
    interaction: str
) -> IntentAnalysisResult:

    if not history:
        history = ["NA"]

    

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

    # ========================================================
    # TASK
    # ========================================================

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

            "Depending on history you can figure out "
            "the latest question if it exists. "

            "Categorise intent on the basis of "
            "`Latest Interaction`. "

            "If `Latest Interaction` is NA, classify "
            "on the basis of `User Input`.\n\n"

            "**Categories**:\n\n"

            "- **closure**: "
            "Greeting, thanks, or ending the chat.\n\n"

            "- **rebuttal**: "
            "User is disagreeing, correcting the system, "
            "or insisting that information they previously "
            "provided is correct "
            "(e.g. 'I already told you', "
            "'That's wrong', 'This is correct.').\n\n"

            "- **additional_info**: "
            "Providing IDs, account numbers or subsequent "
            "question/information asked.\n\n"

            "**Rule**: "
            "If the User Input contradicts the Latest "
            "Interaction or expresses frustration with "
            "the system's request, it MUST be "
            "'rebuttal'.\n\n"

            "**Analysis Required**:\n"

            "Based on the user input and history, provide:\n"

            "1. `user_goal`: "
            "What the user is trying to achieve or accomplish.\n"

            "2. `issue_description`: "
            "The specific problem or issue they're facing.\n"

            "3. `problem_summary`: "
            "A detailed write-up explaining the issue in context.\n\n"

            "Output your response as valid JSON with "
            "the following structure:\n"

            "{\n"

            '  "intent": "<category>",\n'

            '  "user_goal": "<what user wants to accomplish>",\n'

            '  "issue_description": "<specific problem>",\n'

            '  "problem_summary": "<detailed write-up>"\n'

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
        }
    )

   
    result_str = str(result)

    return parse_intent_result(result_str)


async def run_classifier_with_enrichment_async(
    payload: Dict,
    incident_description: str,
    user_qa_pairs: List[Dict] = None,
    comment: str = None
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

    # ========================================================
    # USER INPUT REQUIREMENT
    # ========================================================

    needs_user_input = False
    clarification_question = None

    if not identifiers:

        needs_user_input = True

        clarification_question = (
            "Could you please provide one of: "
            "UCIC, Mobile Number, or Account Number "
            "to help investigate this issue?"
        )

    # ========================================================
    # INITIAL ENRICHED PROMPT
    # ========================================================

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

    # ========================================================
    # SUGGESTED APPROACH
    # ========================================================

    suggested_approach = (
        f"1. First, use Similarity Search to find similar "
        f"resolved {problem_category} incidents for the "
        f"{app} application.\n"

        f"2. Then, fetch Jaeger traces using the customer "
        f"identifier to understand current errors.\n"

        f"3. Based on findings, decide whether to query ELK "
        f"for detailed logs or DB for account data."
    )

    # ========================================================
    # INTENT CLASSIFICATION
    # ========================================================

    interaction = comment if comment else "NA"

    intent_result = await run_intent_classifier_crew_async(

        incident_description,

        [
            str(qa)
            for qa in (user_qa_pairs or [])
        ],

        interaction,
    )

    # ========================================================
    # LLM ANALYSIS
    # ========================================================

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

    this is conetx builder 
    
import os
import time
import logging

os.environ["OTEL_SDK_DISABLED"] = "true"

logger = logging.getLogger(__name__)


def format_incidents_for_llm(results, max_conversation_chars=100000):
    if not results:
        return "No similar historic incidents found."

    formatted = []

    for i, inc in enumerate(results, 1):
        if 'document' in inc:
            doc = inc.get('document', {})
            incident_id = doc.get('id', 'N/A')
            content = doc.get('content', '')
            chunked_content = inc.get('chunked_content', content)

            score = inc.get('score') or inc.get('dense_score') or 0

            metadata = inc.get('metadata', {})
            assignment_group = metadata.get('assignment_group', 'N/A')

            if len(chunked_content) > max_conversation_chars:
                chunked_content = chunked_content[:max_conversation_chars] + "..."

            resolution = "No resolution notes available"
            if 'Ticket Resolution notes is:' in content:
                try:
                    resolution_start = content.find('Ticket Resolution notes is:')
                    resolution_end = content.find('\n', resolution_start + 30)
                    if resolution_end == -1:
                        resolution = content[resolution_start + 30:resolution_start + 500]
                    else:
                        resolution = content[resolution_start + 30:resolution_end].strip()
                except Exception:
                    pass

            incident_str = f"""
=== SIMILAR INCIDENT {i} (Similarity: {score:.2%}) ===
Incident ID: {incident_id}
Assignment Group: {assignment_group}

CONTENT:
{chunked_content}

RESOLUTION:
{resolution}

---
"""
        else:
            incident_str = f""" Document not available. """
        formatted.append(incident_str)

    return "\n".join(formatted)


async def run_incident_context_crew_async(incident_description: str, application: str, top_k: int = 5) -> str:
    from crewai import Agent, Task, Crew, Process, LLM
    from crewai.tools import BaseTool
    from typing import List, Dict
    import asyncio
    from new_flow.utils.http_calls import http_client_post_async
    from new_flow.utils.llm import OPENAI_MODEL_NAME, llm_config
    from utils.observability import get_tracer

    logger = logging.getLogger(__name__)

    semantic_search_endpoint = os.getenv("SEMANTIC_SEARCH_ENDPOINT")

    async def similarity_search_api_async(index: str, query_string: str, application: str) -> List[Dict]:
        headers = {
            "Content-Type": "application/json",
        }
        body = {
            "metadata": {"application": application},
            "query": query_string,
            "index_source": index
        }

        logger.info(
            f"[SemanticSearch] REQUEST START | endpoint={semantic_search_endpoint} "
            f"application={application} index={index} query_len={len(query_string)}"
        )
        start_t = time.monotonic()

        try:
            response = await http_client_post_async(
                semantic_search_endpoint,
                headers=headers,
                json=body
            )
        except Exception as e:
            elapsed = time.monotonic() - start_t
            logger.error(
                f"[SemanticSearch] REQUEST FAILED | endpoint={semantic_search_endpoint} "
                f"elapsed={elapsed:.2f}s exception_type={type(e).__name__} "
                f"message='{str(e)}'",
                exc_info=True
            )
            raise

        elapsed = time.monotonic() - start_t
        logger.info(
            f"[SemanticSearch] REQUEST DONE | endpoint={semantic_search_endpoint} "
            f"elapsed={elapsed:.2f}s status_code={response.status_code}"
        )

        json_resp: Dict = response.json
        results: List[Dict] = []
        if json_resp:
            for obj in json_resp:
                results.append(obj)

        logger.info(
            f"[SemanticSearch] PARSED | elapsed={elapsed:.2f}s "
            f"result_count={len(results)} raw_type={type(json_resp).__name__}"
        )

        return results

    _application = application

    class SearchHistoricIncidentsTool(BaseTool):
        name: str = "search_historic_incidents"
        description: str = (
            "Search for similar incidents from the historic incident database. "
            "This tool finds past incidents that are similar to the current issue "
            "and provides their resolution details and conversation context. "
            "Use this to understand how similar issues were resolved in the past."
        )

        def _run(self, incident_description: str, top_k: int = 5):
            return asyncio.run(self._arun(incident_description, top_k))

        async def _arun(self, incident_description: str, top_k: int = 5):
            tool_start = time.monotonic()
            logger.info(
                f"[SearchHistoricIncidentsTool] TOOL CALL START | "
                f"app={_application} top_k={top_k} incident_description_len={len(incident_description)}"
            )
            try:
                results = await similarity_search_api_async(
                    index='incidents',
                    query_string=incident_description,
                    application=_application
                )
                elapsed = time.monotonic() - tool_start
                logger.info(
                    f"[SearchHistoricIncidentsTool] TOOL CALL SUCCESS | "
                    f"app={_application} elapsed={elapsed:.2f}s result_count={len(results)}"
                )
                return format_incidents_for_llm(results)

            except Exception as e:
                elapsed = time.monotonic() - tool_start
                logger.error(
                    f"[SearchHistoricIncidentsTool] TOOL CALL FAILED | "
                    f"app={_application} elapsed={elapsed:.2f}s "
                    f"exception_type={type(e).__name__} message='{str(e)}'",
                    exc_info=True
                )
                # Fail soft instead of raising — prevents CrewAI's internal
                # retry loop from stacking on top of http_calls.py's own
                # retry/backoff, which is what turns one slow call into
                # a multi-minute stall.
                return "No similar historic incidents found (search temporarily unavailable)."

    tracer = get_tracer(__name__)

    with tracer.start_as_current_span("_run_incident_context_crew_async") as span:
        span.set_attribute("incident_description", incident_description[:100] + "..." if len(incident_description) > 100 else incident_description)
        span.set_attribute("application", application)

        llm = LLM(
            model="openai/" + OPENAI_MODEL_NAME,
            temperature=0.0,
            base_url=llm_config.url,
            api_key=llm_config.token
        )

        search_tool = SearchHistoricIncidentsTool()

        agent = Agent(
            role="Historic Incident Analyst",
            goal="Find similar past incidents and provide resolution context",
            backstory=(
                "You are a senior incident analyst with access to a database of historic incidents. "
                "Your expertise is in finding similar past incidents and understanding how they were resolved. "
                "You will search for similar incidents and format the findings to help resolve the current issue."
            ),
            tools=[search_tool],
            verbose=False,
            allow_delegation=False,
            llm=llm,
            temperature=0,
            max_iter=2,
            reasoning=False,
            max_retry_limit=1
        )

        task = Task(
            name="Find similar incidents",
            description=(
                "Search for the top {top_k} most similar historic incidents to the following current incident:\n\n"
                "{incident_description}\n\n"
                "Use the search_historic_incidents tool to find similar cases. "
                "Then analyze the results and provide a summary of:\n"
                "1. The most relevant similar incidents found\n"
                "2. How those incidents were resolved\n"
                "3. Key troubleshooting steps from the conversation context\n"
                "4. Any patterns or common solutions that could apply to the current incident"
            ),
            agent=agent,
            expected_output=(
                "A structured summary of similar historic incidents with their resolutions, "
                "ready to be used as context for resolving the current incident."
            ),
        )

        crew = Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            verbose=True
        )

        crew_start = time.monotonic()
        logger.info(f"[ContextBuilder] CREW KICKOFF START | application={application}")

        with tracer.start_as_current_span("crew_kickoff_async") as crew_span:
            crew_span.set_attribute("incident_description", incident_description + "..." if len(incident_description) > 100 else incident_description)
            output = str(await crew.akickoff(
                inputs={"incident_description": incident_description, "top_k": top_k}
            ))

        crew_elapsed = time.monotonic() - crew_start
        logger.info(
            f"[ContextBuilder] CREW KICKOFF DONE | application={application} "
            f"elapsed={crew_elapsed:.2f}s output_len={len(output)}"
        )

        return output
