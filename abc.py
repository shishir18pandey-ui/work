import re
from typing import Dict, List, Optional, Any
from pydantic import BaseModel
from new_flow.utils.llm import llm_config
from crewai import Agent, Task, Crew, LLM
from new_flow.tools.app_config import get_app_config,APPS_CONFIG


class IntentClassifierOutput(BaseModel):
    intent: str                              # closure, rebuttal, additional_info
    app: str                                 # cbs, optimus, idp, etc.
    problem_category: str                    # account_freeze, login_failure, etc.
    customer_identifiers: Dict[str, str]     # {ucic, mobile, account, customer_id}
    enriched_prompt: str                     # LLM-ready prompt for downstream
    suggested_approach: str                  # Initial guidance for Plan Agent
    needs_user_input: bool = False           # True if additional info needed
    clarification_question: Optional[str] = None  # Question to ask user
    user_goal: str = ""                      # What the user is trying to accomplish
    issue_description: str = ""              # What problem they're facing
    problem_summary: str = ""                # Detailed write-up of the issue



def extract_identifiers(text: str, payload: Dict) -> Dict[str, str]:
    identifiers = {}

    ucic_patterns = [
        r'\bUCIC[:\s]*(\d{10,12})\b',
        r'\bucic[:\s]*(\d{10,12})\b',
    ]
    for pattern in ucic_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            identifiers['ucic'] = match.group(1) if match.lastindex else match.group(0)
            break

    mobile_patterns = [
        r'\b(\+91[6-9]\d{9})\b',
        r'\b(0[6-9]\d{9})\b',
        r'\bmobile[:\s]*(\+91[6-9]\d{9})\b',
        r'\bmobile[:\s]*(0[6-9]\d{9})\b',
    ]

    for pattern in mobile_patterns:
        match = re.search(pattern, text)
        if match:
            identifiers['mobile'] = match.group(1)
            break

    account_patterns = [
        r'\Account[:\s]*(\d{10,12})\b',
        r'\baccount[:\s]*(\d{10,12})\b',
        r'\bAc[\s/-]*(\d{10,12})\b',
    ]
    for pattern in account_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            identifiers['account'] = match.group(1)
            break
    
    customer_id_patterns = [
        r'\bCUSTOMER\s*ID[:\s]*(\d+)\b',
        r'\bcustomer_id[:\s]*(\d+)\b',

        r'\bCIF[:\s]*(\d+)\b',
    ]
    for pattern in customer_id_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            identifiers['customer_id'] = match.group(1)
            break
    
    username_patterns = [
        r'\busername[:\s]*(\w+)\b',
        r'\buser[:\s]*(\w+)\b',
    ]
    for pattern in username_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            identifiers['username'] = match.group(1)
            break
    
    # Extract identifiers from payload (ServiceNow fields)
    # These take precedence over text-extracted values if not already found
    if payload:
        # UCIC from payload
        if 'ucic' not in identifiers:
            ucic = payload.get('individualUCIC') or payload.get('ucic')
            if ucic:
                identifiers['ucic'] = str(ucic)
                # Also add as customer_id (UCIC maps to customer_id in Jaeger)
                identifiers['customer_id'] = str(ucic)
        
        # Mobile from payload
        if 'mobile' not in identifiers:
            mobile = payload.get('mobile_number') or payload.get('mobile')
            if mobile:
                identifiers['mobile'] = str(mobile)
        
        # Account from payload
        if 'account' not in identifiers:
            account = payload.get('account_number') or payload.get('account')
            if account:
                identifiers['account'] = str(account)
        
        # Customer ID from payload
        if 'customer_id' not in identifiers:
            customer_id = payload.get('customer_id') or payload.get('cif')
            if customer_id:
                identifiers['customer_id'] = str(customer_id)
        
        # Username from payload
        if 'username' not in identifiers:
            username = payload.get('username') or payload.get('user_name')
            if username:
                identifiers['username'] = str(username)
    
    return identifiers


def guess_problem_category(description: str, app: str) -> str:
    desc_lower = description.lower()
    
    problem_keywords = {
        "account_freeze": ["freeze", "blocked", "suspended", "hold"],
        "transaction_failure": ["transaction", "transfer", "payment", "failed"],
        "balance_issue": ["balance", "balance missing", "incorrect balance"],
        "login_failure": ["login", "cannot login", "password", "authentication"],
        "session_timeout": ["session", "timeout", "logged out"],
        "mfa_issue": ["mfa", "otp", "two-factor", "authentication code"],
        "password_reset": ["password reset", "forgot password"],
        "loan_issue": ["loan", "emi", "repayment"],
        "kyc_issue": ["kyc", "verification", "documents"],
    }
    
    for category, keywords in problem_keywords.items():
        for keyword in keywords:
            if keyword in desc_lower:
                return category

    return "application_issue"


def get_app_from_payload(payload: Dict) -> str:
    tier1 = payload.get("tier1", "").lower().strip()
    if tier1 in APPS_CONFIG:
        return tier1


class IntentAnalysisResult(BaseModel):
    """Structured output from the intent classifier LLM"""
    intent: str = ""                         # closure, rebuttal, or additional_info
    user_goal: str = ""                      # What the user is trying to accomplish
    issue_description: str = ""              # What problem they're facing
    problem_summary: str = ""                # Detailed write-up of the issue


def parse_intent_result(result_str: str) -> IntentAnalysisResult:
    import json
    
    json_match = re.search(r'\{[^{}]*\}', result_str, re.DOTALL)
    
    if json_match:
        try:
            data = json.loads(json_match.group())
            return IntentAnalysisResult(
                intent=data.get("intent", "").strip().lower(),
                user_goal=data.get("user_goal", "").strip(),
                issue_description=data.get("issue_description", "").strip(),
                problem_summary=data.get("problem_summary", "").strip()
            )
        except json.JSONDecodeError:
            pass
    
    try:
        data = json.loads(result_str)
        return IntentAnalysisResult(
            intent=data.get("intent", "").strip().lower(),
            user_goal=data.get("user_goal", "").strip(),
            issue_description=data.get("issue_description", "").strip(),
            problem_summary=data.get("problem_summary", "").strip()
        )
    except json.JSONDecodeError:
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
        problem_summary=result_str.strip()
    )


async def run_intent_classifier_crew_async(
    incident_description: str,
    history: list,
    interaction: str
) -> IntentAnalysisResult:
    if not history:
        history = ["NA"]

    llm = LLM(
        model="openai/" + llm_config.model_name,
        temperature=0.0,
        base_url=llm_config.url,
        api_key=llm_config.token
    )

    intent_agent = Agent(
        role="Intent Classifier",
        goal="Analyze user input and categorize it into the correct intent category with detailed analysis",
        backstory=(
            "You are an expert at classifying user intents in a technical support system. "
            "Your job is to analyze the conversation history and current user input to determine "
            "what the user is trying to accomplish. You provide both the category and a detailed "
            "analysis of the user's issue. Users may not always express themselves clearly, so "
            "you must infer and articulate what they actually want."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
        temperature=0,
        max_iter=2,
        reasoning=False,
        max_retry_limit=2
    )

    intent_task = Task(
        description=(
            "Analyze the user input and categorize it with detailed analysis.\n\n"
            "Interaction History: \n```\n{history}\n```\n\n"
            "User Input: \n```\n{incident_description}\n```\n\n"
            "Latest Interaction: \n```\n{interaction}\n```\n\n"
            "Depending on history you can figure out latest question if it exists. Catagorise intent on basis of `Latest Interaction` if it is NA, then classify on baisis of `User Input` \n\n"
            "**Categories**:\n\n"
            "\t- **closure**: Greeting, thanks, or ending the chat.\n\n"
            "\t- **rebuttal**: User is disagreeing, correcting the system, or insisting that information they previously provided is correct (e.g., 'I already told you', 'That's wrong', 'This is correct.').\n\n"
            "\t- **additional_info**: Providing IDs, account numbers or subsequent question/information asked.\n\n"
            "**Rule**: If the User Input contradicts the Latest Interaction or expresses frustration with the system's request, it MUST be 'rebuttal'.\n\n"
            "**Analysis Required**:\n"
            "Based on the user input and history, provide:\n"
            "1. `user_goal`: What the user is trying to achieve or accomplish\n"
            "2. `issue_description`: The specific problem or issue they're facing\n"
            "3. `problem_summary`: A detailed write-up explaining the issue in context\n\n"
            "Output your response as JSON with the following structure:\n"
            "```json\n"
            "{\n"
            "  \"intent\": \"<category>\",\n"
            "  \"user_goal\": \"<what user wants to accomplish>\",\n"
            "  \"issue_description\": \"<specific problem>\",\n"
            "  \"problem_summary\": \"<detailed write-up>\"\n"
            "}\n"
            "```"
        ),
        agent=intent_agent,
        expected_output=(
            "JSON with intent (closure/rebuttal/additional_info), user_goal, issue_description, "
            "and problem_summary fields"
        )
    )

    crew = Crew(
        agents=[intent_agent],
        tasks=[intent_task],
        verbose=True
    )

    result = await crew.akickoff(
        inputs={
            "incident_description": incident_description,
            "history": history,
            "interaction": interaction
        }
    )
    
    # Parse the JSON response
    result_str = str(result)
    return parse_intent_result(result_str)


async def run_classifier_with_enrichment_async(
    payload: Dict,
    incident_description: str,
    user_qa_pairs: List[Dict] = None,
    comment: str = None
) -> IntentClassifierOutput:

    identifiers = extract_identifiers(incident_description, payload)

    app = get_app_from_payload(payload)

    try:
        app_config = get_app_config(app)
    except ValueError:
        app_config = None
    
    problem_category = guess_problem_category(incident_description, app)

    needs_user_input = False
    clarification_question = None

    if not identifiers:
        needs_user_input = True
        clarification_question = (
            "Could you please provide one of: UCIC, Mobile Number, "
            "or Account Number to help investigate this issue?"
        )
    
    enriched_prompt = f"""
Incident Summary:
- Application: {app_config.name if app_config else app}
- Problem Category: {problem_category}

Customer Identifiers:
{chr(10).join(f"- {k}: {v}" for k, v in identifiers.items()) if identifiers else "- Not provided"}

Description:
{incident_description}

Please investigate this issue starting with Similarity Search for similar historic incidents,
then check Jaeger traces to understand the current error context.
"""
    
    suggested_approach = (
        f"1. First, use Similarity Search to find similar resolved {problem_category} incidents "
        f"for the {app} application.\n"
        f"2. Then, fetch Jaeger traces using the customer identifier to understand current errors.\n"
        f"3. Based on findings, decide whether to query ELK for detailed logs or DB for account data."
    )
    
    interaction = comment if comment else "NA"
    intent_result = await run_intent_classifier_crew_async(
        incident_description,
        [str(qa) for qa in (user_qa_pairs or [])],
        interaction
    )
    
    # Use LLM-provided analysis if available, otherwise fall back to heuristics
    user_goal = intent_result.user_goal if intent_result.user_goal else ""
    issue_description = intent_result.issue_description if intent_result.issue_description else incident_description
    problem_summary = intent_result.problem_summary if intent_result.problem_summary else f"User reported: {incident_description}"
    
    if intent_result.user_goal or intent_result.problem_summary:
        enriched_prompt = f"""
Incident Summary:
- Application: {app_config.name if app_config else app}
- Problem Category: {problem_category}

Customer Identifiers:
{chr(10).join(f"- {k}: {v}" for k, v in identifiers.items()) if identifiers else "- Not provided"}

LLM Analysis:
- User Goal: {user_goal}
- Issue Description: {issue_description}
- Problem Summary: {problem_summary}

Original Description:
{incident_description}

Please investigate this issue starting with Similarity Search for similar historic incidents,
then check Jaeger traces to understand the current error context.
"""
    
    return IntentClassifierOutput(
        intent=intent_result.intent if intent_result.intent else "additional_info",
        app=app,
        problem_category=problem_category,
        customer_identifiers=identifiers,
        enriched_prompt=enriched_prompt,
        suggested_approach=suggested_approach,
        needs_user_input=needs_user_input,
        clarification_question=clarification_question,
        user_goal=user_goal,
        issue_description=issue_description,
        problem_summary=problem_summary
    )

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
            "3. `rebuttal`\n"
            "Use when there is a previous bot interaction AND the user explicitly "
            "contradicts, corrects, rejects, or challenges something the bot previously "
            "said or did.\n\n"

            "A request to check, verify, search, query, investigate, or use a specific "
            "tag, ID, service, account, or other piece of information is NOT a "
            "`rebuttal` by itself. Such information should be classified as "
            "`additional_info` unless the user is explicitly correcting or rejecting "
            "the bot's previous action or statement.\n\n"

            "Examples of `additional_info` (NOT rebuttal):\n"
            "- 'Check for this tag: customer_id=12345'\n"
            "- 'Search using this service name.'\n"
            "- 'Check the trace for this ID.'\n"
            "- 'Use this tag: account_id=12345'\n"
            "- 'Please check the `error_code` tag.'\n"
            "- 'Look for this value in Jaeger.'\n\n"

            "Examples of `rebuttal`:\n"
            "- 'You are checking the wrong tag. Check customer_id instead.'\n"
            "- 'That's not the correct service. Check DEBITCARD-API.'\n"
            "- 'I already gave you this information.'\n"
            "- 'No, that's incorrect.'\n"
            "- 'You misunderstood what I said.'\n"
            "- 'Don't check that tag; I told you to check customer_id.'\n\n"
            "**Analysis Required**:\n"
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

026-08-21 03:37:13,047 - __main__ - INFO - ✓ Flow completed | incident=dbe6a7313b7a4350b6986f34c3e45a7c
2026-08-21 03:42:11,848 - new_flow.utils.llm - INFO - OPENAI_API_KEY refreshed
2026-08-21 03:43:17,558 - __main__ - INFO - → Message received | incident=ec72a7462bbe0710ea06f771fe91bf00 event=new_incident partition=1 offset=685
2026-08-21 03:43:17,558 - __main__ - INFO - ✓ Offset committed | incident=ec72a7462bbe0710ea06f771fe91bf00
2026-08-21 03:43:17,559 - __main__ - INFO - → Processing | incident=ec72a7462bbe0710ea06f771fe91bf00 event=new_incident module=new_flow.flow
2026-08-21 03:43:17,589 - __main__ - INFO -   DB status | incident=ec72a7462bbe0710ea06f771fe91bf00 status=in_progress
2026-08-21 03:43:17,593 - __main__ - INFO -   Flow type | incident=ec72a7462bbe0710ea06f771fe91bf00 type=new_incident
╭───────────────────────────── 🌊 Flow Execution ──────────────────────────────╮
│                                                                              │
│  Starting Flow Execution                                                     │
│  Name: IncidentManagementFlow                                                │
│  ID: 0af3c41c-dfb7-48c3-853f-9dfc5e37606a                                    │
│                                                                              │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯

╭────────────────────────────── 🌊 Flow Started ───────────────────────────────╮
│                                                                              │
│  Flow Started                                                                │
│  Name: IncidentManagementFlow                                                │
│  ID: 0af3c41c-dfb7-48c3-853f-9dfc5e37606a                                    │
│                                                                              │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯

Flow started with ID: 0af3c41c-dfb7-48c3-853f-9dfc5e37606a
2026-08-21 03:43:33,623 - crewai.flow.flow - INFO - Flow started with ID: 0af3c41c-dfb7-48c3-853f-9dfc5e37606a
2026-08-21 03:43:33,624 - new_flow.agents.intent_classifier - INFO - [TOKEN CHECK] using token ending in a0_3jHFM
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
│  ID: e52d6201-984d-4685-91f1-74a9e119d307                                    │
│                                                                              │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯

03:43:49 - LiteLLM:INFO: utils.py:3427 - 
LiteLLM completion() model= /app/models/Qwen3-14B-FP8; provider = openai
2026-08-21 03:43:49,662 - LiteLLM - INFO - 
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
│  Short Description: I'm not able to see the reposne quickly they are         │
│  getting blocker                                                             │
│  Description: I'm not able to see the reposne quickly they are getting       │
│  blocker                                                                     │
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
│  - **rebuttal**: 3. `rebuttal`                                               │
│  Use when there is a previous bot interaction AND the user explicitly        │
│  contradicts, corrects, rejects, or challenges something the bot previously  │
│  said or did.                                                                │
│                                                                              │
│  A request to check, verify, search, query, investigate, or use a specific   │
│  tag, ID, service, account, or other piece of information is NOT a           │
│  `rebuttal` by itself. Such information should be classified as              │
│  `additional_info` unless the user is explicitly correcting or rejecting     │
│  the bot's previous action or statement.                                     │
│                                                                              │
│  Examples of `additional_info` (NOT rebuttal):                               │
│  - 'Check for this tag: customer_id=12345'                                   │
│  - 'Search using this service name.'                                         │
│  - 'Check the trace for this ID.'                                            │
│  - 'Use this tag: account_id=12345'                                          │
│  - 'Please check the `error_code` tag.'                                      │
│  - 'Look for this value in Jaeger.'                                          │
│                                                                              │
│  Examples of `rebuttal`:                                                     │
│  - 'You are checking the wrong tag. Check customer_id instead.'              │
│  - 'That's not the correct service. Check DEBITCARD-API.'                    │
│  - 'I already gave you this information.'                                    │
│  - 'No, that's incorrect.'                                                   │
│  - 'You misunderstood what I said.'                                          │
│  - 'Don't check that tag; I told you to check customer_id.'                  │
│                                                                              │
│  **Analysis Required**:                                                      │
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
│  ID: c4886ac9-9ec7-40c1-a406-3993c02bcd7f                                    │
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
│  Short Description: I'm not able to see the reposne quickly they are         │
│  getting blocker                                                             │
│  Description: I'm not able to see the reposne quickly they are getting       │
│  blocker                                                                     │
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
│  - **rebuttal**: 3. `rebuttal`                                               │
│  Use when there is a previous bot interaction AND the user explicitly        │
│  contradicts, corrects, rejects, or challenges something the bot previously  │
│  said or did.                                                                │
│                                                                              │
│  A request to check, verify, search, query, investigate, or use a specific   │
│  tag, ID, service, account, or other piece of information is NOT a           │
│  `rebuttal` by itself. Such information should be classified as              │
│  `additional_info` unless the user is explicitly correcting or rejecting     │
│  the bot's previous action or statement.                                     │
│                                                                              │
│  Examples of `additional_info` (NOT rebuttal):                               │
│  - 'Check for this tag: customer_id=12345'                                   │
│  - 'Search using this service name.'                                         │
│  - 'Check the trace for this ID.'                                            │
│  - 'Use this tag: account_id=12345'                                          │
│  - 'Please check the `error_code` tag.'                                      │
│  - 'Look for this value in Jaeger.'                                          │
│                                                                              │
│  Examples of `rebuttal`:                                                     │
│  - 'You are checking the wrong tag. Check customer_id instead.'              │
│  - 'That's not the correct service. Check DEBITCARD-API.'                    │
│  - 'I already gave you this information.'                                    │
│  - 'No, that's incorrect.'                                                   │
│  - 'You misunderstood what I said.'                                          │
│  - 'Don't check that tag; I told you to check customer_id.'                  │
│                                                                              │
│  **Analysis Required**:                                                      │
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
│    "user_goal": "The user is trying to convey that they are facing an issue  │
│  where they are unable to see the response quickly and it is causing a       │
│  blocker.",                                                                  │
│    "issue_description": "The user is unable to see the response quickly,     │
│  which is causing a blocker in their workflow.",                             │
│    "problem_summary": "The user has reported that they are unable to see     │
│  the response quickly, which is causing a blocker in their process. The      │
│  issue seems to be related to the system's response time or visibility of    │
│  the response, which is impacting their ability to proceed with their task.  │
│  The user has not provided any specific IDs, tags, or additional             │
│  information to help further investigate the issue, so the intent is         │
│  categorized as 'additional_info' as the user is providing context about     │
│  the problem they are facing."                                               │
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
│  Short Description: I'm not able to see the reposne quickly they are         │
│  getting blocker                                                             │
│  Description: I'm not able to see the reposne quickly they are getting       │
│  blocker                                                                     │
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
│  - **rebuttal**: 3. `rebuttal`                                               │
│  Use when there is a previous bot interaction AND the user explicitly        │
│  contradicts, corrects, rejects, or challenges something the bot previously  │
│  said or did.                                                                │
│                                                                              │
│  A request to check, verify, search, query, investigate, or use a specific   │
│  tag, ID, service, account, or other piece of information is NOT a           │
│  `rebuttal` by itself. Such information should be classified as              │
│  `additional_info` unless the user is explicitly correcting or rejecting     │
│  the bot's previous action or statement.                                     │
│                                                                              │
│  Examples of `additional_info` (NOT rebuttal):                               │
│  - 'Check for this tag: customer_id=12345'                                   │
│  - 'Search using this service name.'                                         │
│  - 'Check the trace for this ID.'                                            │
│  - 'Use this tag: account_id=12345'                                          │
│  - 'Please check the `error_code` tag.'                                      │
│  - 'Look for this value in Jaeger.'                                          │
│                                                                              │
│  Examples of `rebuttal`:                                                     │
│  - 'You are checking the wrong tag. Check customer_id instead.'              │
│  - 'That's not the correct service. Check DEBITCARD-API.'                    │
│  - 'I already gave you this information.'                                    │
│  - 'No, that's incorrect.'                                                   │
│  - 'You misunderstood what I said.'                                          │
│  - 'Don't check that tag; I told you to check customer_id.'                  │
│                                                                              │
│  **Analysis Required**:                                                      │
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
│  ID: e52d6201-984d-4685-91f1-74a9e119d307                                    │
│  Final Output: {                                                             │
│    "intent": "additional_info",                                              │
│    "user_goal": "The user is trying to convey that they are facing an issue  │
│  where they are unable to see the response quickly and it is causing a       │
│  blocker.",                                                                  │
│    "issue_description": "The user is unable to see the response quickly,     │
│  which is causing a blocker in their workflow.",                             │
│    "problem_summary": "The user has reported that they are unable to see     │
│  the response quickly, which is causing a blocker in their process. The      │
│  issue seems to be related to the system's response time or visibility of    │
│  the response, which is impacting their ability to proceed with their task.  │
│  The user has not provided any specific IDs, tags, or additional             │
│  information to help further investigate the issue, so the intent is         │
│  categorized as 'additional_info' as the user is providing context about     │
│  the problem they are facing."                                               │
│  }                                                                           │
│                                                                              │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯

2026-08-21 03:43:51,101 - opentelemetry.attributes - WARNING - Invalid type NoneType for attribute 'sop_value' value. Expected one of ['bool', 'str', 'bytes', 'int', 'float'] or a sequence of those types
Enhanced classifier | incident=ec72a7462bbe0710ea06f771fe91bf00 | app=finnone | category=application_issue
Initialized and classified incident ec72a7462bbe0710ea06f771fe91bf00 with intent: additional_info
Need more info for incident ec72a7462bbe0710ea06f771fe91bf00
interaction_counter: 1
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
╭─────────────────────────── 🔄 Flow Method Running ───────────────────────────╮
│                                                                              │
│  Method: handle_needs_more_info                                              │
│  Status: Running                                                             │
│                                                                              │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯

Request - url:https://api.aws-uat.idfcfirstbank.com/snow-incident-mgmt-sys/api/v1/update-incident json:{'callerId': 'shishir.pandey_tho@idfcfirstext.bank.in', 'incidentType': 'Application', 'businessService': 'FinnOne', 'tier1': 'FinnOne', 'tier2': 'Database', 'tier3': 'Response time degradation', 'impact': 'Medium', 'urgency': 'Medium', 'shortDescription': "I'm not able to see the reposne quickly they are getting blocker", 'description': "I'm not able to see the reposne quickly they are getting blocker", 'contactType': 'Self Service', 'sourceIncidentNum': 'INC000006217696', 'sourceIncidentId': '', 'assignmentGroup': 'FinnOne BTO Support', 'businessImpact': '', 'cause': '', 'businessCorrectiveAction': '', 'techCorrectiveAction': '', 'dataSource': '', 'descriptionOfOutage': '', 'entityUCIC': '', 'hashValues': '', 'ipDetails': '', 'ldwNotifyInformation': '', 'loanAccountNumber': '', 'loginId': '', 'mobileNumber': '', 'businessPreventiveAction': '', 'techPreventiveAction': '', 'resoultionTeam': '', 'rootCause': '', 'systemName': '', 'urlOrDomain': '', 'userDetail': '', 'individualUCIC': '', 'sourceIncCreateddttime': '21-Aug-2026 09:13:15', 'incidentId': 'ec72a7462bbe0710ea06f771fe91bf00', 'state': 'On Hold', 'additionalComments': 'Could you please provide one of: UCIC, Mobile Number, or Account Number to help investigate this issue?', 'onHoldReason': 'User Action Required', 'resolutionNotes': '', 'userLocation': ''} headers:{'Content-Type': 'application/json', 'correlationId': '970f5e08-68a8-4d77-b7a0-10b4632054f5', 'source': 'IncidentBot', 'transactionId': 'fe1828ca-f7a4-4f33-b5c8-0104f795cb83', 'Authorization': 'Basic MThiNDhmNzctNzMyMy00MWJiLWJmNmUtYzZmODNlODdhOTgyOkI3TGJ1NVhkcX5JT1paaW1SLThNVDE3eWtF'}

Successfully updated incident ec72a7462bbe0710ea06f771fe91bf00 in ServiceNow
Response: {'message': 'Incident has been updated successfully.', 'incidentNumber': 'INC000006217696', 'incidentId': 'ec72a7462bbe0710ea06f771fe91bf00', 'incidentType': 'Application', 'businessService': 'FinnOne', 'tier1': 'FinnOne', 'tier2': 'Database', 'tier3': 'Response time degradation', 'impact': 'Low', 'urgency': 'Medium', 'priority': 'Low', 'shortDescription': "I'm not able to see the reposne quickly they are getting blocker", 'description': "I'm not able to see the reposne quickly they are getting blocker", 'contactType': 'Self Service', 'state': 'On Hold', 'onHoldReason': 'User Action Required', 'vendorGroup': '', 'causedByPatch': None, 'resolutionCode': None, 'solutionType': None, 'outageType': None, 'resolutionNotes': '', 'additionalComments': '21-Aug-2026 09:13:51 - Incident BOT (Additional comments)\nCould you please provide one of: UCIC, Mobile Number, or Account Number to help investigate this issue?\n\n', 'assignmentGroup': 'FinnOne BTO Support', 'sourceIncidentNum': 'INC000006217696', 'sourceIncidentId': '970f5e08-68a8-4d77-b7a0-10b4632054f5', 'businessImpact': None, 'cause': None, 'businessCorrectiveAction': None, 'techCorrectiveAction': None, 'dataSource': None, 'descriptionOfOutage': None, 'emailID': None, 'entityUCIC': None, 'hashValues': None, 'ipDetails': None, 'ldwNotifyInformation': None, 'loanAccountNumber': None, 'loginId': None, 'mobileNumber': None, 'businessPreventiveAction': None, 'techPreventiveAction': None, 'resoultionTeam': None, 'rootCause': None, 'systemName': None, 'urlOrDomain': None, 'userDetail': None, 'individualUCIC': None, 'sourceIncCreateddttime': '21-Aug-2026 09:13:15', 'userLocation': '', 'incidentURL': 'https://idfcfirstbanktest2.service-now.com/isupport?sys_id=ec72a7462bbe0710ea06f771fe91bf00&view=sp&id=ticket&table=incident'}
Question sent for incident ec72a7462bbe0710ea06f771fe91bf00
DB updated for incident ec72a7462bbe0710ea06f771fe91bf00 status=on_hold
╭────────────────────────── ✅ Flow Method Completed ──────────────────────────╮
│                                                                              │
│  Method: handle_needs_more_info                                              │
│  Status: Completed                                                           │
│                                                                              │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯

╭───────────────────────────── ✅ Flow Completion ─────────────────────────────╮
│                                                                              │
│  Flow Execution Completed                                                    │
│  Name: IncidentManagementFlow                                                │
│  ID: 0af3c41c-dfb7-48c3-853f-9dfc5e37606a                                    │
│                                                                              │
│                                                                              │
╰─────────────────────────────────────────────────
