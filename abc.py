from pydantic import BaseModel
from typing import List, Optional, Dict
import httpx
import os
import json
import re
from dotenv import load_dotenv
from crewai.flow.flow import Flow, start, listen, router
from new_flow.utils.incident_db_async import upsert_incident_payload_async
from new_flow.agents.context_builder import run_incident_context_crew_async, run_incident_context_deterministic_async
from new_flow.agents.intent_classifier import run_classifier_with_enrichment_async
from new_flow.agents.plan_agents import run_plan_agent_async
from new_flow.agents.execute_agent_jaeger import run_jaeger_only_async
from new_flow.agents.summary_agent import run_summary_agent_async, run_context_only_summary_async
from new_flow.agents.self_critique import run_self_critique_async, should_escalate
from new_flow.utils.llm import run_crew_with_retry_async
from new_flow.tools.discovery_tools import discover_jaeger_services_impl
from new_flow.tools.app_config import app_has_observability


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
    
    try:
        return json.loads(output.strip())
    except json.JSONDecodeError:
        pass
    
    json_patterns = [
        r'```json\s*([\s\S]*?)\s*```',
        r'```\s*([\s\S]*?)\s*```',
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
       
        request_payload = {k: v for k, v in request_payload.items() if v is not None}

        headers.update({
            "Authorization": f"Basic {os.environ['SNOW_TOKEN']}",
        })

        interaction_counter = payload.get("interaction_counter")
        print(f"interaction_counter: {interaction_counter}")

        if interaction_counter is not None and interaction_counter <= 3:
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
        else:
            logger.warning(
                f"interaction_counter={interaction_counter} exceeds limit or is missing — "
                f"skipping ServiceNow send for incident {incident_id}"
            )
            return False, {"status_code": 0, "response_text": "interaction_counter limit exceeded or missing"}


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

            # Get previous classifier output for context continuity
            previous_classifier_output = self.state.payload.get('__agent_data', {}).get('classifier_output')

            classifier_output = await run_crew_with_retry_async(
                lambda: run_classifier_with_enrichment_async(
                    payload=self.state.payload,
                    incident_description=self.state.incident_description,
                    user_qa_pairs=self.state.user_qa_pairs,
                    comment=comment,
                    previous_classifier_output=previous_classifier_output
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
                # FIX: unique event name — does NOT collide with the string
                # "update_servicenow" that _run_agentic_resolver returns on
                # the normal path, which would otherwise double-fire both listeners.
                return "send_clarification"
            
            print(f"Fresh incident {self.state.incident_id}, gather context")
            return "gather_context"


    @listen('gather_context')
    async def semantic_search(self):
        tracer = get_tracer(__name__)
        with tracer.start_as_current_span("gather_context") as span:
            span.set_attribute("incident_id", self.state.incident_id)
            span.set_attribute("incident_description_length", len(self.state.incident_description))

            desc = self.state.enriched_prompt
            app_raw = self.state.payload.get("businessService", "CBS")
            app_key = app_raw.lower().strip()

            if app_has_observability(app_key):
                incident_context = await run_crew_with_retry_async(
                    lambda: run_incident_context_crew_async(desc, application=app_raw)
                )
            else:
                incident_context = await run_crew_with_retry_async(
                    lambda: run_incident_context_deterministic_async(desc, application=app_raw)
                )
            self.state.incident_context = incident_context if incident_context else "No context found"

            return 'run_resolver'


    @listen(semantic_search)
    async def run_resolver_crew(self):
        tracer = get_tracer(__name__)
        with tracer.start_as_current_span("run_resolver") as span:
            span.set_attribute("incident_id", self.state.incident_id)
            span.set_attribute("qa_pairs_count", len(self.state.user_qa_pairs))

            app = self.state.payload.get("businessService", "cbs").lower().strip()
            self.state.app = app
            span.set_attribute("app", app)
            logger.info(f"Resolver | incident={self.state.incident_id} app={app}")

            return await self._run_agentic_resolver(app)

    async def _run_agentic_resolver(self, app: str):
        tracer = get_tracer(__name__)

        customer_ids = self.state.customer_identifiers if hasattr(self.state, 'customer_identifiers') else {}
        problem_cat = self.state.problem_category if hasattr(self.state, 'problem_category') else ""

        if not app_has_observability(app):
            logger.info(f"No Jaeger/ELK config for app={app} - resolving from similarity search context only")
            with tracer.start_as_current_span("context_only_summary") as span:
                span.set_attribute("app", app)
                summary_output = await run_crew_with_retry_async(
                    lambda: run_context_only_summary_async(
                        incident_description=self.state.incident_description,
                        historic_context=self.state.incident_context,
                        user_qa_pairs=self.state.user_qa_pairs
                    )
                )
            self.state.summary_output = summary_output.model_dump() if hasattr(summary_output, 'model_dump') else summary_output
            self.state.agent_output = {
                "resolved": summary_output.resolved,
                "diagnosis": summary_output.diagnosis,
                "solution": summary_output.solution,
                "questions": summary_output.questions
            }
            print(f"Context-only resolution completed for incident {self.state.incident_id}")
            return "update_servicenow"

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

            if hasattr(execution_result, 'confidence') and execution_result.confidence < 0.3:
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
        
        self.state.agent_output = {
            "resolved": summary_output.resolved,
            "diagnosis": summary_output.diagnosis,
            "solution": summary_output.solution,
            "questions": summary_output.questions
        }
        
        return "update_servicenow"

    # ── FIX: distinct trigger names, no possible overlap ──
    # start_process (router) emits "send_clarification" for the needs-user-input
    # short-circuit. _run_agentic_resolver returns "update_servicenow" as an
    # ordinary return value (only meaningful via @listen(run_resolver_crew)
    # matching method completion). These two strings never collide, so exactly
    # one listener fires per incident.
    @listen('send_clarification')
    async def handle_needs_more_info(self):
        await self._send_agent_output_to_servicenow()

    @listen(run_resolver_crew)
    async def update_servicenow(self):
        await self._send_agent_output_to_servicenow()

    async def _send_agent_output_to_servicenow(self):
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



─────────────────────────────────────────────────────────────────────────────╯

2026-08-21 06:21:38,033 - new_flow.utils.llm - ERROR - === FULL EXCEPTION TRACEBACK ===
2026-08-21 06:21:38,033 - new_flow.utils.llm - ERROR - Traceback (most recent call last):
  File "/usr/local/lib/python3.11/site-packages/crewai/task.py", line 937, in interpolate_inputs_and_add_conversation_history
    self.description = interpolate_only(
                       ^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/crewai/utilities/string_utils.py", line 143, in interpolate_only
    raise KeyError(
KeyError: "Template variable 'previous_analysis' not found in inputs dictionary"

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/app/new_flow/utils/llm.py", line 96, in run_crew_with_retry_async
    result = await crew_coro
             ^^^^^^^^^^^^^^^
  File "/app/new_flow/agents/intent_classifier.py", line 592, in run_classifier_with_enrichment_async
    intent_result = await run_intent_classifier_crew_async(
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/app/new_flow/agents/intent_classifier.py", line 500, in run_intent_classifier_crew_async
    result = await crew.akickoff(
             ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/crewai/crew.py", line 973, in akickoff
    inputs = prepare_kickoff(self, inputs, input_files)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/crewai/crews/utils.py", line 325, in prepare_kickoff
    crew._interpolate_inputs(normalized)
  File "/usr/local/lib/python3.11/site-packages/crewai/crew.py", line 1796, in _interpolate_inputs
    [
  File "/usr/local/lib/python3.11/site-packages/crewai/crew.py", line 1797, in <listcomp>
    task.interpolate_inputs_and_add_conversation_history(
  File "/usr/local/lib/python3.11/site-packages/crewai/task.py", line 941, in interpolate_inputs_and_add_conversation_history
    raise ValueError(
ValueError: Missing required template variable 'Template variable 'previous_analysis' not found in inputs dictionary' in description

2026-08-21 06:21:38,033 - new_flow.utils.llm - ERROR - === EXCEPTION TYPE: ValueError ===
2026-08-21 06:21:38,033 - new_flow.utils.llm - ERROR - === EXCEPTION MESSAGE: Missing required template variable 'Template variable 'previous_analysis' not found in inputs dictionary' in description ===
╭──────────────────────────────── Crew Failure ────────────────────────────────╮
│                                                                              │
│  Crew Execution Failed                                                       │
│  Name: crew                                                                  │
│  ID: 031534c7-6f98-4ee1-acae-6ae502641d10                                    │
│                                                                              │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯

╭─────────────────────────── ❌ Flow Method Failed ────────────────────────────╮
│                                                                              │
│  Method: initialize_and_classify                                             │
│  Status: Failed                                                              │
│                                                                              │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯

2026-08-21 06:21:38,035 - __main__ - ERROR - ✗ Flow FAILED | incident=82960c9a2b364710ea06f771fe91bf7b module=new_flow.flow error=Missing required template variable 'Template variable 'previous_analysis' not found in inputs dictionary' in description
Traceback (most recent call last):
  File "/usr/local/lib/python3.11/site-packages/crewai/task.py", line 937, in interpolate_inputs_and_add_conversation_history
    self.description = interpolate_only(
                       ^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/crewai/utilities/string_utils.py", line 143, in interpolate_only
    raise KeyError(
KeyError: "Template variable 'previous_analysis' not found in inputs dictionary"

The above exception was the direct cause of the following exception:

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
  File "/app/new_flow/flow.py", line 250, in initialize_and_classify
    classifier_output = await run_crew_with_retry_async(
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/app/new_flow/utils/llm.py", line 96, in run_crew_with_retry_async
    result = await crew_coro
             ^^^^^^^^^^^^^^^
  File "/app/new_flow/agents/intent_classifier.py", line 592, in run_classifier_with_enrichment_async
    intent_result = await run_intent_classifier_crew_async(
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/app/new_flow/agents/intent_classifier.py", line 500, in run_intent_classifier_crew_async
    result = await crew.akickoff(
             ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/crewai/crew.py", line 973, in akickoff
    inputs = prepare_kickoff(self, inputs, input_files)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/crewai/crews/utils.py", line 325, in prepare_kickoff
    crew._interpolate_inputs(normalized)
  File "/usr/local/lib/python3.11/site-packages/crewai/crew.py", line 1796, in _interpolate_inputs
    [
  File "/usr/local/lib/python3.11/site-packages/crewai/crew.py", line 1797, in <listcomp>
    task.interpolate_inputs_and_add_conversation_history(
  File "/usr/local/lib/python3.11/site-packages/crewai/task.py", line 941, in interpolate_inputs_and_add_conversation_history
    raise ValueError(
ValueError: Missing required template variable 'Template variable 'previous_analysis' not found in inputs dictionary' in description
shishir.pandey_tho@0325LTPB0124444 ~ % 

