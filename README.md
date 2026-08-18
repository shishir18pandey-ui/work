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
        
    model = "openai/" + "/app/models/Qwen3-14B-FP8",
    temperature=0.0,
    base_url= "https://llm-api.iservebetter.idfcfirstbank.com/qwen3-14b/v1",
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

