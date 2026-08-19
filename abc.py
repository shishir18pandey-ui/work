───────────────────────── Tracing Preference Saved ──────────────────────────╮
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

2026-08-19 05:38:38,247 - new_flow.utils.llm - ERROR - Max retries (3) reached for crew execution
╭─────────────────────────── ❌ Flow Method Failed ────────────────────────────╮
│                                                                              │
│  Method: initialize_and_classify                                             │
│  Status: Failed                                                              │
│                                                                              │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯

2026-08-19 05:38:38,251 - __main__ - ERROR - ✗ Flow FAILED | incident=02e8ad292bbe4f10ea06f771fe91bfab module=new_flow.flow error=litellm.AuthenticationError: AuthenticationError: OpenAIException - Error code: 401 - {'error': 'invalid_token', 'error_description': 'The access token is invalid or has expired'}
Traceback (most recent call last):
  File "/usr/local/lib/python3.11/site-packages/litellm/llms/openai/openai.py", line 823, in acompletion
    headers, response = await self.make_openai_chat_completion_request(
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/litellm/litellm_core_utils/logging_utils.py", line 190, in async_wrapper
    result = await func(*args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/litellm/llms/openai/openai.py", line 454, in make_openai_chat_completion_request
    raise e
  File "/usr/local/lib/python3.11/site-packages/litellm/llms/openai/openai.py", line 436, in make_openai_chat_completion_request
    await openai_aclient.chat.completions.with_raw_response.create(
  File "/usr/local/lib/python3.11/site-packages/openai/_legacy_response.py", line 386, in wrapped
    return cast(LegacyAPIResponse[R], await func(*args, **kwargs))
                                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/openai/resources/chat/completions/completions.py", line 2907, in create
    return await self._post(
           ^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/openai/_base_client.py", line 1992, in post
    return await self.request(cast_to, opts, stream=stream, stream_cls=stream_cls)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/openai/_base_client.py", line 1777, in request
    raise self._make_status_error_from_response(err.response) from None
openai.AuthenticationError: Error code: 401 - {'error': 'invalid_token', 'error_description': 'The access token is invalid or has expired'}

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/usr/local/lib/python3.11/site-packages/litellm/main.py", line 599, in acompletion
    response = await init_response
               ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/litellm/llms/openai/openai.py", line 870, in acompletion
    raise OpenAIError(
litellm.llms.openai.common_utils.OpenAIError: Error code: 401 - {'error': 'invalid_token', 'error_description': 'The access token is invalid or has expired'}

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/app/worker.py", line 138, in process_message
    await flow.akickoff()
  File "/usr/local/lib/python3.11/site-packages/crewai/flow/flow.py", line 2211, in akickoff
    return await self.kickoff_async(inputs, input_files)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/crewai/flow/flow.py", line 2091, in kickoff_async
    await asyncio.gather(*tasks)
  File "/usr/local/lib/python3.11/site-packages/crewai/flow/flow.py", line 2243, in _execute_start_method
    result, finished_event_id = await self._execute_method(
                                ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/crewai/flow/flow.py", line 2428, in _execute_method
    raise e
  File "/usr/local/lib/python3.11/site-packages/crewai/flow/flow.py", line 2341, in _execute_method
    result = await method(*args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/crewai/flow/flow.py", line 2288, in enhanced_method
    return await original_method(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/app/new_flow/flow.py", line 240, in initialize_and_classify
    classifier_output = await run_crew_with_retry_async(
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/app/new_flow/utils/llm.py", line 136, in run_crew_with_retry_async
    raise last_error
  File "/app/new_flow/utils/llm.py", line 96, in run_crew_with_retry_async
    result = await crew_coro
             ^^^^^^^^^^^^^^^
  File "/app/new_flow/agents/intent_classifier.py", line 748, in run_classifier_with_enrichment_async
    intent_result = await run_intent_classifier_crew_async(
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/app/new_flow/agents/intent_classifier.py", line 639, in run_intent_classifier_crew_async
    result = await crew.akickoff(
             ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/crewai/crew.py", line 976, in akickoff
    result = await self._arun_sequential_process()
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/crewai/crew.py", line 1035, in _arun_sequential_process
    return await self._aexecute_tasks(self.tasks)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/crewai/crew.py", line 1097, in _aexecute_tasks
    task_output = await task.aexecute_sync(
                  ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/crewai/task.py", line 578, in aexecute_sync
    return await self._aexecute_core(agent, context, tools)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/crewai/task.py", line 694, in _aexecute_core
    raise e  # Re-raise the exception after emitting the event
    ^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/crewai/task.py", line 602, in _aexecute_core
    result = await agent.aexecute_task(
             ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/crewai/agent/core.py", line 892, in aexecute_task
    result = await self._handle_execution_error_async(e, task, context, tools)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/crewai/agent/core.py", line 697, in _handle_execution_error_async
    self._check_execution_error(e, task)
  File "/usr/local/lib/python3.11/site-packages/crewai/agent/core.py", line 643, in _check_execution_error
    raise e
  File "/usr/local/lib/python3.11/site-packages/crewai/agent/core.py", line 879, in aexecute_task
    result = await self._aexecute_without_timeout(task_prompt, task)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/crewai/agent/core.py", line 937, in _aexecute_without_timeout
    result = await self.agent_executor.ainvoke(
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/crewai/agents/crew_agent_executor.py", line 1131, in ainvoke
    formatted_answer = await self._ainvoke_loop()
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/crewai/agents/crew_agent_executor.py", line 1170, in _ainvoke_loop
    return await self._ainvoke_loop_react()
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/crewai/agents/crew_agent_executor.py", line 1285, in _ainvoke_loop_react
    raise e
  File "/usr/local/lib/python3.11/site-packages/crewai/agents/crew_agent_executor.py", line 1195, in _ainvoke_loop_react
    answer = await aget_llm_response(
             ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/crewai/utilities/agent_utils.py", line 531, in aget_llm_response
    raise e
  File "/usr/local/lib/python3.11/site-packages/crewai/utilities/agent_utils.py", line 521, in aget_llm_response
    answer = await llm.acall(
             ^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/crewai/llm.py", line 1931, in acall
    return await self._ahandle_non_streaming_response(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/crewai/llm.py", line 1339, in _ahandle_non_streaming_response
    response = await litellm.acompletion(**params)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/app/utils/minimax_tool_call_patch.py", line 65, in acompletion
    return _recover(await _orig_acompletion(*args, **kwargs))
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/litellm/utils.py", line 1642, in wrapper_async
    raise e
  File "/usr/local/lib/python3.11/site-packages/litellm/utils.py", line 1488, in wrapper_async
    result = await original_function(*args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/litellm/main.py", line 618, in acompletion
    raise exception_type(
          ^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/litellm/litellm_core_utils/exception_mapping_utils.py", line 2328, in exception_type
    raise e
  File "/usr/local/lib/python3.11/site-packages/litellm/litellm_core_utils/exception_mapping_utils.py", line 482, in exception_type
    raise AuthenticationError(
litellm.exceptions.AuthenticationError: litellm.AuthenticationError: AuthenticationError: OpenAIException - Error code: 401 - {'error': 'invalid_token', 'error_description': 'The access token is invalid or has expired'}
2026-08-19 05:38:39,225 - __main__ - INFO -   Fallback rejection sent | incident=02e8ad292bbe4f10ea06f771fe91bfab
2026-08-19 05:41:24,002 - new_flow.utils.llm - ERROR - HTTPSConnectionPool(host='app.auth.idfcfirstbank.com', port=443): Max retries exceeded with url: /api/session-management/v1/token (Caused by SSLError(SSLCertVerificationError(1, '[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate (_ssl.c:1006)')))
2026-08-19 05:41:24,017 - new_flow.utils.llm - ERROR - OPENAI_API_KEY refresh failed: HTTPSConnectionPool(host='app.auth.idfcfirstbank.com', port=443): Max retries exceeded with url: /api/session-management/v1/token (Caused by SSLError(SSLCertVerificationError(1, '[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate (_ssl.c:1006)')))
shishir.pandey_tho@0325LTPB0124444 ~ % 




bewlos is intent classifer.py
import os
import re
import json
from typing import Dict, List, Optional, Any

from pydantic import BaseModel
from new_flow.utils.llm import llm_config
from crewai import Agent, Task, Crew, LLM
from new_flow.tools.app_config import get_app_config, APPS_CONFIG



CA_CERT = os.getenv("SSL_CERT_FILE")
if CA_CERT and os.path.exists(CA_CERT):
    os.environ["SSL_CERT_FILE"] = CA_CERT
    os.environ["REQUESTS_CA_BUNDLE"] = CA_CERT

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

    # ========================================================
    # SSL CHECK
    # ========================================================

    ca_cert = os.getenv("SSL_CERT_FILE")

    if not ca_cert:

        local_ca_cert = (
            "/Users/shishir.pandey_tho/script/"
            "IDFCBANKCA.pem"
        )

        if os.path.exists(local_ca_cert):
            ca_cert = local_ca_cert

    if ca_cert:

        if os.path.exists(ca_cert):

            os.environ["SSL_CERT_FILE"] = ca_cert
            os.environ["REQUESTS_CA_BUNDLE"] = ca_cert

            print(
                f"Using CA certificate: {ca_cert}"
            )

        else:

            print(
                f"WARNING: SSL_CERT_FILE is set but "
                f"certificate does not exist: {ca_cert}"
            )

    else:

        print(
            "WARNING: No CA certificate configured. "
            "HTTPS certificate verification may fail."
        )
    api_key = llm_config.token

    if not api_key:
        import new_flow.utils.llm as llm_module
        logger.error(
            f"[TOKEN DEBUG] llm_config id={id(llm_config)} "
            f"module llm_config id={id(llm_module.llm_config)} "
            f"same_object={llm_config is llm_module.llm_config} "
            f"token_repr={llm_config.token!r}"
    )

    # ========================================================
    # LLM
    # ========================================================

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
    print(
        "CA certificate:",
        ca_cert or "NOT CONFIGURED"
    )
    print("==============================")

    # ========================================================
    # AGENT
    # ========================================================

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
    
