this is app_config.py
from typing import Dict, List, Optional
from pydantic import BaseModel
from pathlib import Path
import json


class AppConfig(BaseModel):
    name: str
    db_instance: str
    services: List[str]
    tables: List[str]
    elk_index: str
    elk_endpoint: Optional[str] = None
    jaeger_endpoint: Optional[str] = None
    jaeger_prod_endpoint: Optional[str] = None
    default_jaeger_service: Optional[str] = None
    problem_categories: List[str]

def generate_app_config_from_file(filename: str) -> Dict[str, AppConfig]:
    filepath = Path(f"{Path(__file__).resolve().parent}/{filename}")
    with filepath.open('r') as file:
        try:
            raw_confg = json.loads(file.read().strip())
            return {key: AppConfig(**val) for key, val in raw_confg.items()}
        except json.JSONDecodeError as e:
            print(f"APP CONFIG PARSE FAIL - {e}")
            return {}

APPS_CONFIG: Dict[str, AppConfig] = generate_app_config_from_file("app_config.json")

def get_app_config(app: str) -> AppConfig:
    app_key = app.lower().strip()
    if app_key not in APPS_CONFIG:
        raise ValueError(
            f"App '{app}' not supported. Available: {list(APPS_CONFIG.keys())}"
        )
    return APPS_CONFIG[app_key]

def get_supported_apps() -> List[str]:
    return list(APPS_CONFIG.keys())

DEFAULT_APP_CONFIG = AppConfig(
    name="Unknown Application",
    db_instance="main",
    services=[],
    tables=[],
    elk_index="elk-*",
    default_jaeger_service=None,
    problem_categories=[]
)

def get_app_config_safe(app: str) -> AppConfig:
    try:
        return get_app_config(app)
    except ValueError:
        return DEFAULT_APP_CONFIG

def get_jaeger_endpoint(app: str) -> Optional[str]:
    if not app:
        return None
    try:
        config = get_app_config(app)
        return config.jaeger_prod_endpoint
    except ValueError:
        return None



app_config.json
{
    "cbs": {
        "name": "Core Banking System",
        "db_instance": "main",
        "services": [
            "cbs-backend",
            "cbs-api",
            "cbs-core"
        ],
        "tables": [
            "GLDM",
            "INFZ",
            "NPAR",
            "ACD3",
            "STTM",
            "DPD",
            "ACCOUNT"
        ],
        "elk_endpoint": "https://DCELASDBSPRR07-prod.logging.devops.idfcbank.com:9200;https://DCELASDBSPRR08-prod.logging.devops.idfcbank.com:9200;https://DCELASDBSPRR09-prod.logging.devops.idfcbank.com:9200",
        "elk_index": "elk-cbs-prod*",
        "jaeger_endpoint": "https://tracing.uat-cbs.idfcfirstbank.com/api",
        "jaeger_prod_endpoint": "https://tracing.prod-cbs.idfcfirstbank.com/api",
        "default_jaeger_service": "cbs-backend",
        "problem_categories": [
            "account_freeze",
            "transaction_failure",
            "balance_issue",
            "payment_failure",
            "loan_issue",
            "kyc_issue"
        ]
    },
    "optimus": {
        "name": "Mobile/Net Banking",
        "db_instance": "idp",
        "services": [
            "optimus-api",
            "optimus-login",
            "optimus-web",
            "optimus-mobile"
        ],
        "tables": [
            "users",
            "devices",
            "sessions",
            "oauth_tokens",
            "mfa_records"
        ],
        "elk_endpoint":"https://dcawsmonitelasticcoord0.logging.devops.idfcbank.com:9200;https://dcawsmonitelasticcoord1.logging.devops.idfcbank.com:9200;https://dcawsmonitelasticcoord2.logging.devops.idfcbank.com:9200",
        "elk_index": "elk-optimus-prod*",
        "jaeger_endpoint": "https://tracing.uat-opt.idfcfirstbank.com/api",
        "jaeger_prod_endpoint": "https://tracing.uat-opt.idfcfirstbank.com/api",
        "default_jaeger_service": "optimus-api",
        "problem_categories": [
            "login_failure",
            "session_timeout",
            "transaction_failure",
            "app_crash",
            "password_reset_failure",
            "mfa_issue"
        ]
    },
    "idp": {
        "name": "Identity Provider",
        "db_instance": "idp",
        "services": [
            "idp-auth",
            "idp-oauth",
            "idp-mfa"
        ],
        "tables": [
            "users",
            "oauth_tokens",
            "mfa_records",
            "devices",
            "sessions"
        ],
        "elk_endpoint": "",
        "elk_index": "elk-idp-prod*",
        "jaeger_endpoint": "https://tracing.uat-idp.idfcfirstbank.com/api",
        "jaeger_prod_endpoint": "https://tracing.prod-idp.idfcfirstbank.com/api",
        "default_jaeger_service": "idp-auth",
        "problem_categories": [
            "authentication_failure",
            "oauth_error",
            "mfa_failure",
            "token_expired",
            "device_registration_failure"
        ]
    },
    "phi-upi": {
        "name": "PHI-UPI",
        "db_instance": "",
        "services": [],
        "tables": [],
        "elk_endpoint":"https://DCELASDBSPRR07-prod.logging.devops.idfcbank.com:9200;https://DCELASDBSPRR08-prod.logging.devops.idfcbank.com:9200;https://DCELASDBSPRR09-prod.logging.devops.idfcbank.com:9200",
        "elk_index": "",
        "jaeger_endpoint": "",
        "jaeger_prod_endpoint": "https://prod-jaeger-ui.obs.idfcfirstbank.com/jaeger/api",
        "default_jaeger_service": "",
        "problem_categories": []
    },
    "ntbsa": {
        "name": "New To Bank",
        "db_instance": "",
        "services": [],
        "tables": [],
        "elk_endpoint": "",
        "elk_index": "",
        "jaeger_endpoint": "",
        "jaeger_prod_endpoint":"https://dcawsmonitelasticcoord0.logging.devops.idfcbank.com:9200;https://dcawsmonitelasticcoord1.logging.devops.idfcbank.com:9200;https://dcawsmonitelasticcoord2.logging.devops.idfcbank.com:9200",
        "default_jaeger_service": "",
        "problem_categories": []
    },
    "finnone": {
        "name": "FinnOne",
        "db_instance": "",
        "services": [],
        "tables": [],
        "elk_endpoint":"https://DCELASDBSPRR07-prod.logging.devops.idfcbank.com:9200;https://DCELASDBSPRR08-prod.logging.devops.idfcbank.com:9200;https://DCELASDBSPRR09-prod.logging.devops.idfcbank.com:9200",
        "elk_index": "",
        "jaeger_endpoint": "",
        "jaeger_prod_endpoint": "",
        "default_jaeger_service": "",
        "problem_categories": []
    },
    "etoll": {
        "name": "FinnOne",
        "db_instance": "",
        "services": [],
        "tables": [],
        "elk_endpoint": "https://DCELASDBSPRR07-prod.logging.devops.idfcbank.com:9200;https://DCELASDBSPRR08-prod.logging.devops.idfcbank.com:9200;https://DCELASDBSPRR09-prod.logging.devops.idfcbank.com:9200",
        "elk_index": "",
        "jaeger_endpoint": "https://tracing.prod-etollissuer.idfcfirstbank.com/api",
        "jaeger_prod_endpoint": "",
        "default_jaeger_service": "",
        "problem_categories": []
    },
    "finesse": {
        "name": "Contact Center (Finesse, IVR)",
        "db_instance": "",
        "services": [],
        "tables": [],
        "elk_endpoint": "https://DCELASDBSPRR07-prod.logging.devops.idfcbank.com:9200;https://DCELASDBSPRR08-prod.logging.devops.idfcbank.com:9200;https://DCELASDBSPRR09-prod.logging.devops.idfcbank.com:9200",
        "elk_index": "",
        "jaeger_endpoint": "",
        "jaeger_prod_endpoint": "",
        "default_jaeger_service": "",
        "problem_categories": []
    },
    "prime": {
        "name": "Credit Card - PRIME",
        "db_instance": "",
        "services": [],
        "tables": [],
        "elk_endpoint": "https://DCELASDBSPRR07-prod.logging.devops.idfcbank.com:9200;https://DCELASDBSPRR08-prod.logging.devops.idfcbank.com:9200;https://DCELASDBSPRR09-prod.logging.devops.idfcbank.com:9200",
        "elk_index": "",
        "jaeger_endpoint": "",
        "jaeger_prod_endpoint": "",
        "default_jaeger_service": "",
        "problem_categories": []
    },
    "brnet": {
        "name": "BRNET",
        "db_instance": "",
        "services": [],
        "tables": [],
        "elk_endpoint": "https://DCEST3APPPRR01-prod.logging.devops.idfcbank.com:9200;https://DCEST3APPPRR02-prod.logging.devops.idfcbank.com:9200;https://DCEST3APPPRR03-prod.logging.devops.idfcbank.com:9200",
        "elk_index": "",
        "jaeger_endpoint": "https://prod-jaeger-ui.obs.idfcfirstbank.com/jaeger/search;https://prod-jaeger-2-ui.obs.idfcbank.com/jaeger/search",
        "jaeger_prod_endpoint": "",
        "default_jaeger_service": "",
        "problem_categories": []
    },
    "cdp": {
        "name": "Customer Data Platform",
        "db_instance": "",
        "services": [],
        "tables": [],
        "elk_endpoint": "https://dcawsmonitelasticcoord0.logging.devops.idfcbank.com:9200;https://dcawsmonitelasticcoord1.logging.devops.idfcbank.com:9200;https://dcawsmonitelasticcoord2.logging.devops.idfcbank.com:9200",
        "elk_index": "",
        "jaeger_endpoint": "https://tracing.cdp.idfcfirstbank.com/api",
        "jaeger_prod_endpoint": "",
        "default_jaeger_service": "",
        "problem_categories": []
    },
    "api": {
        "name": "API Integrations",
        "db_instance": "",
        "services": [],
        "tables": [],
        "elk_endpoint": "https://dcawsmonitelasticcoord0.logging.devops.idfcbank.com:9200;https://dcawsmonitelasticcoord1.logging.devops.idfcbank.com:9200;https://dcawsmonitelasticcoord2.logging.devops.idfcbank.com:9200",
        "elk_index": "",
        "jaeger_endpoint": "https://prod-jaeger-ui.obs.idfcfirstbank.com/jaeger/api",
        "jaeger_prod_endpoint": "",
        "default_jaeger_service": "",
        "problem_categories": []
    },
    "bxp": {
        "name": "CMS BXP",
        "db_instance": "",
        "services": [],
        "tables": [],
        "elk_endpoint":"https://DCELASDBSPRR07-prod.logging.devops.idfcbank.com:9200;https://DCELASDBSPRR08-prod.logging.devops.idfcbank.com:9200;https://DCELASDBSPRR09-prod.logging.devops.idfcbank.com:9200",
        "elk_index": "",
        "jaeger_endpoint": "",
        "jaeger_prod_endpoint": "",
        "default_jaeger_service": "",
        "problem_categories": []
    },
    "wms": {
        "name": "Wealth Management System(WMS)",
        "db_instance": "",
        "services": [],
        "tables": [],
        "elk_endpoint":"https://DCEST3APPPRR01-prod.logging.devops.idfcbank.com:9200;https://DCEST3APPPRR02-prod.logging.devops.idfcbank.com:9200;https://DCEST3APPPRR03-prod.logging.devops.idfcbank.com:9200",
        "elk_index": "",
        "jaeger_endpoint": "https://prod-jaeger-ui.obs.idfcfirstbank.com/jaeger/api",
        "jaeger_prod_endpoint": "",
        "default_jaeger_service": "",
        "problem_categories": []
    },
    "imps": {
        "name": "IMPS",
        "db_instance": "",
        "services": [],
        "tables": [],
        "elk_endpoint":"https://DCELASDBSPRR07-prod.logging.devops.idfcbank.com:9200;https://DCELASDBSPRR08-prod.logging.devops.idfcbank.com:9200;https://DCELASDBSPRR09-prod.logging.devops.idfcbank.com:9200",
        "elk_index": "",
        "jaeger_endpoint": "",
        "jaeger_prod_endpoint": "",
        "default_jaeger_service": "",
        "problem_categories": []
    },
    "newgen": {
        "name": "Newgen (IBPS, OmniDocs, Zapin)",
        "db_instance": "",
        "services": [],
        "tables": [],
        "elk_endpoint":"https://DCEST3APPPRR01-prod.logging.devops.idfcbank.com:9200;https://DCEST3APPPRR02-prod.logging.devops.idfcbank.com:9200;https://DCEST3APPPRR03-prod.logging.devops.idfcbank.com:9200",
        "elk_index": "",
        "jaeger_endpoint": "",
        "jaeger_prod_endpoint": "",
        "default_jaeger_service": "",
        "problem_categories": []
    },
    "commhub": {
        "name": "Communication Hub",
        "db_instance": "",
        "services": [],
        "tables": [],
        "elk_endpoint": "https://dcawsmonitelasticcoord0.logging.devops.idfcbank.com:9200;https://dcawsmonitelasticcoord1.logging.devops.idfcbank.com:9200;https://dcawsmonitelasticcoord2.logging.devops.idfcbank.com:9200",
        "elk_index": "",
        "jaeger_endpoint": "https://tracing.commhub.idfcbank.com/api",
        "jaeger_prod_endpoint": "",
        "default_jaeger_service": "",
        "problem_categories": []
    },
    "fmconv": {
        "name": "FM Converge",
        "db_instance": "",
        "services": [],
        "tables": [],
        "elk_endpoint":"https://DCELASDBSPRR07-prod.logging.devops.idfcbank.com:9200;https://DCELASDBSPRR08-prod.logging.devops.idfcbank.com:9200;https://DCELASDBSPRR09-prod.logging.devops.idfcbank.com:9200",
        "elk_index": "",
        "jaeger_endpoint": "",
        "jaeger_prod_endpoint": "",
        "default_jaeger_service": "",
        "problem_categories": []
    },
    "phi-neft": {
        "name": "",
        "db_instance": "",
        "services": [],
        "tables": [],
        "elk_endpoint":"https://DCEST3APPPRR01-prod.logging.devops.idfcbank.com:9200;https://DCEST3APPPRR02-prod.logging.devops.idfcbank.com:9200;https://DCEST3APPPRR03-prod.logging.devops.idfcbank.com:9200",
        "elk_index": "",
        "jaeger_endpoint": "https://prod-jaeger-ui.obs.idfcfirstbank.com/jaeger/api",
        "jaeger_prod_endpoint": "",
        "default_jaeger_service": "",
        "problem_categories": []
    },
    "ogl": {
        "name": "Oracle GL (OGL)",
        "db_instance": "",
        "services": [],
        "tables": [],
        "elk_endpoint":"https://DCEST3APPPRR01-prod.logging.devops.idfcbank.com:9200;https://DCEST3APPPRR02-prod.logging.devops.idfcbank.com:9200;https://DCEST3APPPRR03-prod.logging.devops.idfcbank.com:9200",
        "elk_index": "",
        "jaeger_endpoint": "",
        "jaeger_prod_endpoint": "",
        "default_jaeger_service": "",
        "problem_categories": []
    },
    "figateway": {
        "name": "FI Gateway",
        "db_instance": "",
        "services": [],
        "tables": [],
        "elk_endpoint":"https://DCELASDBSPRR07-prod.logging.devops.idfcbank.com:9200;https://DCELASDBSPRR08-prod.logging.devops.idfcbank.com:9200;https://DCELASDBSPRR09-prod.logging.devops.idfcbank.com:9200",
        "elk_index": "",
        "jaeger_endpoint": "ttps://prod-jaeger-ui.obs.idfcfirstbank.com/jaeger/api",
        "jaeger_prod_endpoint": "",
        "default_jaeger_service": "",
        "problem_categories": []
    }
}





flow.py
from pydantic import BaseModel
from typing import List, Optional, Dict
import httpx
import os
import json
import re
from dotenv import load_dotenv
from crewai.flow.flow import Flow, start, listen, router
from new_flow.utils.incident_db_async import upsert_incident_payload_async
from new_flow.agents.context_builder import run_incident_context_crew_async
from new_flow.agents.intent_classifier import run_classifier_with_enrichment_async
from new_flow.agents.plan_agents import run_plan_agent_async
from new_flow.agents.execute_agent_jaeger import run_jaeger_only_async
from new_flow.agents.summary_agent import run_summary_agent_async
from new_flow.agents.self_critique import run_self_critique_async, should_escalate
from new_flow.utils.llm import run_crew_with_retry_async
from new_flow.tools.discovery_tools import discover_jaeger_services_impl


load_dotenv()

CA_CERT_FILE = os.getenv("CA_CERT_FILE", "./IDFCBANKCA.pem")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "/app/models/MiniMax-M2.5")
CHAT_COMPLETIONS_URL = f"{OPENAI_API_BASE}/chat/completions"

EXEPEMPTED_PAYLOAD_KEYS = [
    "__agent_data",
    "created_at",
    "headers",
    "file_description",
    "interaction_counter",
    "incidentNumber",
    "status"
]

import logging
from utils.observability import get_tracer

logger = logging.getLogger(__name__)


def extract_json_from_output(output: str) -> dict:
    if not output or not output.strip():
        logger.warning("Empty output received, returning fallback response")
        return {"diagnosis": "Unable to process incident", "solution": "Please try again later", "questions": [], "resolved": "no"}
    
    # Try direct JSON parsing
    try:
        return json.loads(output.strip())
    except json.JSONDecodeError:
        pass
    
    json_patterns = [
        r'```json\s*([\s\S]*?)\s*```',  # ```json ... ```
        r'```\s*([\s\S]*?)\s*```',       # ``` ... ```
    ]
    
    for pattern in json_patterns:
        match = re.search(pattern, output)
        if match:
            json_str = match.group(1).strip()
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
    
   
    json_like_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    match = re.search(json_like_pattern, output)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    
    logger.error(f"Failed to parse JSON from output: {output}...")
    return {
        "diagnosis": "Unable to process incident response",
        "solution": "Please try again later",
        "questions": ["System encountered an issue processing the incident"],
        "resolved": "no"
    }


class IncidentState(BaseModel):
    incident_id: str = ""
    payload: dict = {}
    incident_description: str = ""
    incident_context: str = ""
    user_qa_pairs: List[dict] = []
    intent: str = ""
    agent_output: dict = {}
    snow_status: str = ""
    ucic: str = ""
    current_comment: Optional[str] = None
    app: str = "" 
    customer_identifiers: Dict[str, str] = {}
    problem_category: str = "" 
    plan_output: Optional[Dict] = None
    execution_result: Optional[Dict] = None
    summary_output: Optional[Dict] = None
    enriched_prompt: str = ""
    discovered_services: str = ""


async def send_update_to_servicenow_async(payload: Dict, question: str, resolution: str):
    tracer = get_tracer(__name__)
    with tracer.start_as_current_span("send_update_to_servicenow_async") as span:
        span.set_attribute("incident_id", payload.get("incidentId"))
        span.set_attribute("question_length", len(question) if question else 0)
        span.set_attribute("resolution_length", len(resolution) if resolution else 0)
        
        url = os.environ['SNOW_ENDPOINT']

        incident_id = payload.get("incidentId")
        headers = payload.get("headers", {})

        request_payload = {
            **payload,
            "additionalComments": question,
            "resolutionNotes": resolution,
        }

        for key in EXEPEMPTED_PAYLOAD_KEYS:
            del request_payload[key]
       
        # Remove None values
        request_payload = {k: v for k, v in request_payload.items() if v is not None}

        headers.update({
            "Authorization": f"Basic {os.environ['SNOW_TOKEN']}",
        })

        interaction_counter = payload.get("interaction_counter")
        print(f"interaction_counter: {interaction_counter}")
        if interaction_counter <= 3:
            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    print(f"Request - url:{url} json:{request_payload} headers:{headers}")
                    response = await client.post(url, json=request_payload, headers=headers)

                    if response.status_code == 200:
                        print(f"Successfully updated incident {incident_id} in ServiceNow")
                        print(f"Response: {response.json()}")
                        return True,{"status_code":response.status_code,"response":response.json()}
                    else:
                        logger.warning(
                            f"Failed to update incident {incident_id} in ServiceNow. "
                            f"Status code: {response.status_code}, Response: {response.text}"
                        )
                        try:
                            return False,{"status_code":response.status_code,"response":response.json()}
                        except:
                            return False,{"status_code":response.status_code,"response_text":response.text}

            except Exception as e:
                logger.error(f"Error calling ServiceNow API for incident {incident_id}: {str(e)}")
                return False,{"status_code": 0 ,"response_text":"Exception : "+str(e)}


async def send_rejection_to_servicenow_async(payload, additonal_comment: str = 'BOT is unable to resolve, assign to an Engineer'):
    tracer = get_tracer(__name__)
    with tracer.start_as_current_span("send_rejection_to_servicenow_async") as span:
        span.set_attribute("incident_id", payload.get("incidentId"))
        payload.update({"state": "On Hold","cause": "Bot is unable to resolve Assign to an Engineer."})
        result = await send_update_to_servicenow_async(payload, additonal_comment, '')
        return result, 'rejected'        


async def send_question_to_servicenow_async(payload, question):
    tracer = get_tracer(__name__)
    with tracer.start_as_current_span("send_question_to_servicenow_async") as span:
        span.set_attribute("incident_id", payload.get("incidentId"))
        payload.update({"state": "On Hold", "onHoldReason": "User Action Required"})
        result = await send_update_to_servicenow_async(payload, question, '')
        return result, 'on_hold'         


async def send_resolution_to_servicenow_async(payload, resolution):
    tracer = get_tracer(__name__)
    with tracer.start_as_current_span("send_resolution_to_servicenow_async") as span:
        span.set_attribute("incident_id", payload.get("incidentId"))
        span.set_attribute("resolution_length", len(resolution) if resolution else 0)
        payload.update({
            # "cause": "Resolved by Bot.",
            # "state": "Resolved",
            # "resolutionCode": "Solved (Permanently)",
            # "solutionType": "Other",
            # "outageType": "No Outage"
            "state":"On Hold",
            "onHoldReason": "User Action Required"
        })
        result = await send_update_to_servicenow_async(payload, resolution, None)
        return result, 'on_hold'    


def payload_to_incident_description(payload):
    from new_flow.utils.files_processor import process_attachments 

    tracer = get_tracer(__name__)
    with tracer.start_as_current_span("payload_to_incident_description") as span:
        span.set_attribute("short_description_length", len(payload.get('shortDescription','')))
        span.set_attribute("description_length", len(payload.get('description','')))
        span.set_attribute("individualUCIC", payload.get('individualUCIC','i'))
        
        short_description = payload.get('shortDescription','')
        description = payload.get('description','')
        individualUCIC = payload.get('individualUCIC','i')
        result = f"Short Description: {short_description}\nDescription: {description}"

        file_text = process_attachments(payload.get('files') or [])   
        if file_text:                                                  
            result = result + "\n\n" + file_text                       
        print(f"Generated incident description for UCIC {individualUCIC}")
        return result, individualUCIC



class IncidentManagementFlow(Flow[IncidentState]):

    @start()
    async def initialize_and_classify(self):
        tracer = get_tracer(__name__)
        with tracer.start_as_current_span("initialize_and_classify") as span:
            span.set_attribute("incident_id", self.state.incident_id)

            self.state.incident_description, self.state.ucic = payload_to_incident_description(self.state.payload)

            if '__agent_data' not in self.state.payload:
                self.state.payload['__agent_data'] = {
                    'snow_logs': [], 'qa_pairs': [], 'comments': []
                }

            snow_logs = self.state.payload.get('__agent_data', {}).get('snow_logs', [])
            if snow_logs and snow_logs[-1]['type'] == 'question' and self.state.current_comment:
                 self.state.payload['__agent_data']['qa_pairs'].append({
                     "question": snow_logs[-1]["question"],
                     "answer": self.state.current_comment
                 })

            self.state.user_qa_pairs = self.state.payload['__agent_data'].get('qa_pairs', [])

            comment = self.state.current_comment if self.state.current_comment else None

            classifier_output = await run_crew_with_retry_async(
                lambda: run_classifier_with_enrichment_async(
                    payload=self.state.payload,
                    incident_description=self.state.incident_description,
                    user_qa_pairs=self.state.user_qa_pairs,
                    comment=comment
                )
            )
            self.state.intent = classifier_output.intent
            self.state.customer_identifiers = classifier_output.customer_identifiers
            self.state.problem_category = classifier_output.problem_category
            self.state.app = classifier_output.app
            self.state.enriched_prompt = classifier_output.enriched_prompt
            
            self.state.payload['__agent_data']['classifier_output'] = classifier_output.model_dump()
            
            print(f"Enhanced classifier | incident={self.state.incident_id} | app={self.state.app} | category={self.state.problem_category}")
            
            print(f"Initialized and classified incident {self.state.incident_id} with intent: {self.state.intent}")
            return self.state.intent

    @router(initialize_and_classify)
    async def start_process(self):
        tracer = get_tracer(__name__)
        with tracer.start_as_current_span("logic_router") as span:
            span.set_attribute("intent", self.state.intent)
            span.set_attribute("sop_exists", self.state.payload['__agent_data'].get('sop') is not None)
            span.set_attribute("sop_value", self.state.payload['__agent_data'].get('sop'))

            counter = self.state.payload.get("interaction_counter", 0)
            self.state.payload["interaction_counter"] = counter + 1
            if counter >= 3:
                print(f"Interaction limit exceeded for incident {self.state.incident_id}")
                return "limit_exceeded"

            if self.state.intent == "closure": 
                print(f"Closure intent for incident {self.state.incident_id}")
                return "handle_closure"
            if self.state.intent == "rebuttal": 
                print(f"Rebuttal intent for incident {self.state.incident_id}")
                return "handle_rebuttal"

            classifier_data = self.state.payload.get('__agent_data', {}).get('classifier_output', {})
            if classifier_data.get('needs_user_input'):
                print(f"Need more info for incident {self.state.incident_id}")
                self.state.agent_output = {
                    "resolved": "no",
                    "diagnosis": "Additional information needed",
                    "solution": "Please provide more details",
                    "questions": [classifier_data.get('clarification_question', 'Could you please provide more information? Please provide trace ID/ Correlation ID/ Customer Number/ Account Number if possible')]
                }
                return "update_servicenow"
            
            print(f"Fresh incident {self.state.incident_id}, gather context")
            return "gather_context"


    @listen('gather_context')
    async def semantic_search(self):
        tracer = get_tracer(__name__)
        with tracer.start_as_current_span("gather_context") as span:
            span.set_attribute("incident_id", self.state.incident_id)
            span.set_attribute("incident_description_length", len(self.state.incident_description))

            # Use already-generated description from initialize_and_classify
            # desc = f"{self.state.incident_description}\nUCIC: {self.state.ucic}"
            desc = self.state.enriched_prompt
            app = self.state.payload.get("tier1", "CBS")
            incident_context = await run_crew_with_retry_async(
                lambda: run_incident_context_crew_async(desc,application=app)
            )
            self.state.incident_context = incident_context if incident_context else "No context found"

            return 'run_resolver'


    @listen(semantic_search)
    async def run_resolver_crew(self):
        tracer = get_tracer(__name__)
        with tracer.start_as_current_span("run_resolver") as span:
            span.set_attribute("incident_id", self.state.incident_id)
            span.set_attribute("qa_pairs_count", len(self.state.user_qa_pairs))

            app = self.state.payload.get("tier1", "cbs").lower().strip()
            self.state.app = app
            span.set_attribute("app", app)
            logger.info(f"Resolver | incident={self.state.incident_id} app={app}")

            return await self._run_agentic_resolver(app)

    async def _run_agentic_resolver(self, app: str):
        tracer = get_tracer(__name__)

        customer_ids = self.state.customer_identifiers if hasattr(self.state, 'customer_identifiers') else {}
        problem_cat = self.state.problem_category if hasattr(self.state, 'problem_category') else ""

        try:
            self.state.discovered_services = await discover_jaeger_services_impl(app)
            logger.info(f"agents.plan_agent import PlanOutput={app}")
        except Exception as e:
            logger.warning(f"Failed to discover services: {e}")
            self.state.discovered_services = "Service discovery unavailable"

        with tracer.start_as_current_span("plan_agent") as span:
            span.set_attribute("app", app)
            plan_output = await run_crew_with_retry_async(
                lambda: run_plan_agent_async(
                    enriched_prompt=self.state.enriched_prompt,
                    app=app,
                    customer_identifiers=customer_ids,
                    problem_category=problem_cat,
                    incident_context=self.state.incident_context,
                    discovered_services=self.state.discovered_services
                )
            )
            self.state.plan_output = plan_output.model_dump() if hasattr(plan_output, 'model_dump') else plan_output
            print(f"Plan Agent completed for incident {self.state.incident_id}")
        
        if plan_output.needs_more_info:
            self.state.agent_output = {
                "resolved": "no",
                "diagnosis": "Additional information needed",
                "solution": plan_output.question_for_user or "Please provide more details",
                "questions": [plan_output.question_for_user] if plan_output.question_for_user else []
            }
            return "update_servicenow"
        
        with tracer.start_as_current_span("execute_jaeger_agent") as span:
            span.set_attribute("issue_summary", plan_output.issue_summary[:100] if plan_output.issue_summary else "")
            execution_result = await run_crew_with_retry_async(
                lambda: run_jaeger_only_async(
                    plan_output=plan_output,
                    incident_description=self.state.incident_description,
                    app=app,
                    customer_identifiers=customer_ids,
                    problem_category=problem_cat,
                    max_iterations=5,
                    incident_id=self.state.incident_id,
                    discovered_services=self.state.discovered_services
                )
            )
            self.state.execution_result = execution_result.model_dump() if hasattr(execution_result, 'model_dump') else execution_result
            print(f"Execute Agent completed for incident {self.state.incident_id}")

            # Handle escalation if confidence is low
            if hasattr(execution_result, 'confidence') and execution_result.confidence < 0.3:
                # Trigger escalation via ServiceNow
                escalation_msg = execution_result.escalation_reason or "BOT unable to resolve - assigning to L2 Engineer"
                (status, info), incident_status = await send_rejection_to_servicenow_async(
                    self.state.payload, 
                    escalation_msg
                )
                self.state.payload['__agent_data']['snow_logs'].append({
                    "type": "escalation", 
                    "reason": escalation_msg,
                    "confidence": execution_result.confidence,
                    "status": status, 
                    "response": info
                })
                print(f"Escalated incident {self.state.incident_id} due to low confidence: {execution_result.confidence}")
        
        with tracer.start_as_current_span("summary_agent") as span:
            summary_output = await run_crew_with_retry_async(
                lambda: run_summary_agent_async(
                    incident_description=self.state.incident_description,
                    execution_result=execution_result,
                    historic_context=self.state.incident_context,
                    user_qa_pairs=self.state.user_qa_pairs
                )
            )
            self.state.summary_output = summary_output.model_dump() if hasattr(summary_output, 'model_dump') else summary_output
            print(f"Summary Agent completed for incident {self.state.incident_id}")
        
        # Self-Critique Loop: DISABLED - causing false escalations
        # if summary_output.resolved == "yes":
        #     tool_calls = self.state.execution_result.get('tool_calls', []) if isinstance(self.state.execution_result, dict) else []
        #     critique = await run_self_critique_async(
        #         incident_description=self.state.incident_description,
        #         diagnosis=summary_output.diagnosis,
        #         solution=summary_output.solution,
        #         tool_calls=tool_calls,
        #         user_qa_pairs=self.state.user_qa_pairs
        #     )
        #     
        #     # Log critique results
        #     self.state.payload['__agent_data']['snow_logs'].append({
        #         "type": "self_critique",
        #         "is_valid": critique.is_valid,
        #         "confidence": critique.confidence,
        #         "issues_found": critique.issues_found
        #     })
        #     
        #     # Use revised output if critique found issues
        #     if critique.revised_diagnosis:
        #         summary_output.diagnosis = critique.revised_diagnosis
        #     if critique.revised_solution:
        #         summary_output.solution = critique.revised_solution
        #     
        #     # Check if should escalate based on critique
        #     if should_escalate(critique):
        #         print(f"Self-critique triggered escalation for incident {self.state.incident_id}")
        #         (status, info), incident_status = await send_rejection_to_servicenow_async(
        #             self.state.payload,
        #             f"Auto-escalated: Low confidence ({critique.confidence:.2f}). Issues: {', '.join(critique.issues_found[:2])}"
        #         )
        #         self.state.payload['__agent_data']['snow_logs'].append({
        #             "type": "escalation", 
        #             "reason": "self_critique_escalation",
        #             "confidence": critique.confidence,
        #             "status": status, 
        #             "response": info
        #         })
        #         return "update_servicenow"
        
        self.state.agent_output = {
            "resolved": summary_output.resolved,
            "diagnosis": summary_output.diagnosis,
            "solution": summary_output.solution,
            "questions": summary_output.questions
        }
        
        return "update_servicenow"

    @listen(run_resolver_crew)
    async def update_servicenow(self):
        tracer = get_tracer(__name__)
        with tracer.start_as_current_span("update_servicenow") as span:
            span.set_attribute("incident_id", self.state.incident_id)
            span.set_attribute("resolution_result", self.state.agent_output.get("resolved", "unknown"))

            res = self.state.agent_output
            incident_status = 'in_progress'

            if res.get("resolved") == 'yes':
                msg = f"Diagnosis:\n{res['diagnosis']}\n\nSolution:\n{res['solution']}"
                (status, info), incident_status = await send_resolution_to_servicenow_async(self.state.payload, msg)
                self.state.payload['__agent_data']['snow_logs'].append({
                    "type": "resolution", "resolution": msg, "status": status, "response": info
                })
                print(f"Resolution sent for incident {self.state.incident_id}")

            elif res.get("questions"):
                msg = "\n".join(res["questions"])
                (status, info), incident_status = await send_question_to_servicenow_async(self.state.payload, msg)
                self.state.payload['__agent_data']['snow_logs'].append({
                    "type": "question", "question": msg, "status": status, "response": info
                })
                print(f"Question sent for incident {self.state.incident_id}")

            else:
                msg = f"Diagnosis:\n{res.get('diagnosis')}\n\nSolution:\n{res['solution']}"
                (status, info), incident_status = await send_question_to_servicenow_async(self.state.payload, msg)
                self.state.payload['__agent_data']['snow_logs'].append({
                    "type": "question", "question": msg, "status": status, "response": info
                })
                print(f"Question sent for incident {self.state.incident_id}")

            state = self.state.model_dump()
            payload_copy = state['payload']

            await upsert_incident_payload_async(
                self.state.incident_id,
                json.dumps(payload_copy),
                incident_status
            )
            print(f"DB updated for incident {self.state.incident_id} status={incident_status}")

    @listen('handle_rebuttal')
    async def run_rebuttal_crew(self):
        tracer = get_tracer(__name__)
        with tracer.start_as_current_span("handle_rebuttal") as span:
            span.set_attribute("incident_id", self.state.incident_id)
            msg = 'BOT is unable to resolve, assign to an Engineer'
            (status, info), incident_status = await send_rejection_to_servicenow_async(self.state.payload, msg)
            self.state.payload['__agent_data']['snow_logs'].append({
                "type": "rejection", "status": status, "response": info
            })
            state = self.state.model_dump()
            payload_copy = state['payload']
            await upsert_incident_payload_async(
                self.state.incident_id,
                json.dumps(payload_copy),
                incident_status
            )
            print(f"Rebuttal handled for incident {self.state.incident_id} status={incident_status}")

    @listen('limit_exceeded')
    async def handle_limit_exceeded(self):
        tracer = get_tracer(__name__)
        with tracer.start_as_current_span("handle_limit_exceeded") as span:
            span.set_attribute("incident_id", self.state.incident_id)
            (status, info), incident_status = await send_rejection_to_servicenow_async(self.state.payload)
            self.state.payload['__agent_data']['snow_logs'].append({
                "type": "rejection", "status": status, "response": info
            })
            state = self.state.model_dump()
            payload_copy = state['payload']
            await upsert_incident_payload_async(
                self.state.incident_id,
                json.dumps(payload_copy),
                incident_status
            )
            print(f"Limit exceeded rejected incident {self.state.incident_id} status={incident_status}")
            return

    @listen('closure')
    async def handle_incident_closure(self):
        tracer = get_tracer(__name__)
        with tracer.start_as_current_span("handle_closure") as span:
            span.set_attribute("incident_id", self.state.incident_id)
            print("No action for bot to take")
            return

    @listen("reject_incident")
    async def handle_rejection(self):
        tracer = get_tracer(__name__)
        with tracer.start_as_current_span("handle_rejection") as span:
            span.set_attribute("incident_id", self.state.incident_id)
            (status, info), incident_status = await send_rejection_to_servicenow_async(self.state.payload)
            self.state.payload['__agent_data']['snow_logs'].append({
                "type": "rejection", "status": status, "response": info
            })
            state = self.state.model_dump()
            payload_copy = state['payload']
            await upsert_incident_payload_async(
                self.state.incident_id,
                json.dumps(payload_copy),
                incident_status
            )
            print(f"Rejected incident {self.state.incident_id} status={incident_status}")
            return



summary_agent.py
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

    # Default - use execution result or plan output
    diagnosis = execution_result.diagnosis or plan_output.issue_summary or "Investigation incomplete"
    solution = execution_result.solution or "Manual investigation required"

    return SummaryOutput(
        diagnosis=diagnosis,
        solution=solution,
        questions=[],
        resolved="no"
    )




        
