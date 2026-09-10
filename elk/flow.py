"""ELK-first incident flow. Selected by USE_FLOW=elk_search in worker.py.

Same state machine, SNoW contract and DB behaviour as `new_flow/flow.py` - the
ONLY difference is the resolver: instead of scanning seven 24h Jaeger windows by
tag, this flow keyword-searches the Jaeger span index in Elasticsearch to locate
candidate traces, then fetches those exact traces for the span tree. See
`elk_search_flow/agents/investigate.py` for the rationale.

Shared, unchanged stages (classifier, context builder, plan agent, summary agent,
LLM retry, SNoW helpers' semantics) are imported from `new_flow` rather than
copied, so a fix there reaches this flow too. Nothing here mutates `new_flow`.
"""

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
from new_flow.agents.summary_agent import run_summary_agent_async, run_context_only_summary_async
from new_flow.utils.llm import run_crew_with_retry_async
from new_flow.tools.app_config import app_has_observability, get_app_config_safe
from utils.pii_masking_integration import mask_payload_output

# The one substantive difference from new_flow.
from elk_search_flow.agents.investigate import (
    discover_services_for_app,
    discovered_service_names,
    run_investigation_async,
)


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

        request_payload = mask_payload_output(request_payload, question, resolution)

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
        span.set_attribute("additional_comment", additonal_comment[:500] if additonal_comment else "")
        span.set_attribute("responded_with", "rejection")
        payload.update({"state": "On Hold","cause": "Bot is unable to resolve Assign to an Engineer."})
        result = await send_update_to_servicenow_async(payload, additonal_comment, '')
        return result, 'rejected'        


async def send_question_to_servicenow_async(payload, question):
    tracer = get_tracer(__name__)
    with tracer.start_as_current_span("send_question_to_servicenow_async") as span:
        span.set_attribute("incident_id", payload.get("incidentId"))
        span.set_attribute("question_sent", question[:500] if question else "")
        span.set_attribute("responded_with", "question")
        span.set_attribute("question_length", len(question) if question else 0)
        payload.update({"state": "On Hold", "onHoldReason": "User Action Required"})
        result = await send_update_to_servicenow_async(payload, question, '')
        return result, 'on_hold'         


async def send_resolution_to_servicenow_async(payload, resolution):
    tracer = get_tracer(__name__)
    with tracer.start_as_current_span("send_resolution_to_servicenow_async") as span:
        span.set_attribute("incident_id", payload.get("incidentId"))
        span.set_attribute("resolution_length", len(resolution) if resolution else 0)
        span.set_attribute("responded_with", "diagnosis")
        span.set_attribute("diagnosis", resolution[:500] if resolution else "")
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
            span.set_attribute("problem_category", self.state.problem_category)
            span.set_attribute("app", self.state.app)
            
            # Add needs_user_input attribute from classifier
            classifier_data = self.state.payload.get('__agent_data', {}).get('classifier_output', {})
            span.set_attribute("needs_user_input", classifier_data.get('needs_user_input', False))
            span.set_attribute("interaction_counter", self.state.payload.get("interaction_counter", 0))

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
            span.set_attribute("app", self.state.app)
            span.set_attribute("problem_category", self.state.problem_category)

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
            
            # Track semantic search results
            has_historic_context = bool(incident_context and incident_context != "No context found")
            span.set_attribute("has_historic_context", has_historic_context)

            return 'run_resolver'


    @listen(semantic_search)
    async def run_resolver_crew(self):
        tracer = get_tracer(__name__)
        with tracer.start_as_current_span("run_resolver") as span:
            span.set_attribute("incident_id", self.state.incident_id)
            span.set_attribute("qa_pairs_count", len(self.state.user_qa_pairs))
            span.set_attribute("app", self.state.app)
            span.set_attribute("problem_category", self.state.problem_category)
            span.set_attribute("has_historic_context", bool(self.state.incident_context))

            app = self.state.payload.get("businessService", "cbs").lower().strip()
            self.state.app = app
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

        # ── Discover the services the LLM is allowed to choose from ──────────
        # Discovered from the span index itself, because that is the index LOCATE
        # filters on: a `serviceName` the index has never seen returns zero hits
        # silently. Jaeger's own /services is the fallback, not the primary - it
        # reads only the FIRST jaeger_endpoint, which for optimus is a UAT host
        # while the spans we search are on the prod cluster.
        with tracer.start_as_current_span("discover_services") as span:
            self.state.discovered_services, source = await discover_services_for_app(app)
            span.set_attribute("app", app)
            span.set_attribute("discovery_source", source)
            span.set_attribute(
                "services_discovered", len(discovered_service_names(self.state.discovered_services))
            )
            logger.info(f"Service discovery for {app} via {source}")

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
        
        # ── The ELK-first resolver: keyword-locate spans, then deepen by trace ──
        with tracer.start_as_current_span("investigate_elk_first") as span:
            span.set_attribute("issue_summary", plan_output.issue_summary[:100] if plan_output.issue_summary else "")
            execution_result = await run_crew_with_retry_async(
                lambda: run_investigation_async(
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

            span.set_attribute("confidence", execution_result.confidence)
            span.set_attribute("diagnosis", (execution_result.diagnosis or "")[:500])
            span.set_attribute("solution", (execution_result.solution or "")[:500])
            span.set_attribute("resolved", execution_result.resolved)
            span.set_attribute("iterations_completed", execution_result.iterations_completed)
            span.set_attribute("tool_call_count", len(execution_result.tool_calls))
            # Which stage produced the evidence: span_tree | keyword_only |
            # jaeger_fallback | exhausted. The single most useful field for
            # telling whether leading with ELK actually paid off.
            span.set_attribute("search_stage", execution_result.search_stage)
            span.set_attribute("trace_ids_found", len(execution_result.trace_ids))
            # Which relaxation step of the span search answered (planned service
            # / whole app / any service / unfiltered). Tells you whether the plan
            # agent's service pick is pulling its weight in production.
            locate = next(
                (c for c in reversed(execution_result.tool_calls)
                 if str(c.get("stage", "")).startswith("locate")),
                {},
            )
            span.set_attribute("locate_filter_scope", str(locate.get("filter_scope") or ""))

            print(f"Investigation completed for incident {self.state.incident_id} stage={execution_result.search_stage}")

            # Elasticsearch was already the primary search here, so there is no
            # separate ELK fallback crew to run - unlike new_flow, which only
            # reaches ES at this point. Persist what we found for the SNoW payload.
            if execution_result.confidence < 0.3:
                logger.info(
                    f"LOW CONFIDENCE {execution_result.confidence} "
                    f"stage={execution_result.search_stage} - escalating"
                )
                self.state.payload["__agent_data"]["elk_context"] = (
                    execution_result.evidence_text or execution_result.diagnosis or ""
                )

                escalation_msg = execution_result.escalation_reason or "BOT unable to resolve - assigning to L2 Engineer"
                (status, info), incident_status = await send_rejection_to_servicenow_async(
                    self.state.payload,
                    escalation_msg
                )
                self.state.payload['__agent_data']['snow_logs'].append({
                    "type": "escalation",
                    "reason": escalation_msg,
                    "confidence": execution_result.confidence,
                    "search_stage": execution_result.search_stage,
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
            
            # Add response type tracking
            res = self.state.agent_output
            incident_status = 'in_progress'
            
            if res.get("resolved") == 'yes':
                span.set_attribute("response_type", "resolution")
                span.set_attribute("diagnosis", res.get('diagnosis', '')[:500])
                span.set_attribute("solution", res.get('solution', '')[:500])
                msg = f"Diagnosis:\n{res['diagnosis']}\n\nSolution:\n{res['solution']}"
                (status, info), incident_status = await send_resolution_to_servicenow_async(self.state.payload, msg)
                self.state.payload['__agent_data']['snow_logs'].append({
                    "type": "resolution", "resolution": msg, "status": status, "response": info
                })
                print(f"Resolution sent for incident {self.state.incident_id}")

            elif res.get("questions"):
                span.set_attribute("response_type", "clarification")
                span.set_attribute("questions_count", len(res.get("questions", [])))
                msg = "\n".join(res["questions"])
                (status, info), incident_status = await send_question_to_servicenow_async(self.state.payload, msg)
                self.state.payload['__agent_data']['snow_logs'].append({
                    "type": "question", "question": msg, "status": status, "response": info
                })
                print(f"Question sent for incident {self.state.incident_id}")

            else:
                span.set_attribute("response_type", "clarification")
                span.set_attribute("diagnosis", res.get('diagnosis', '')[:500])
                span.set_attribute("solution", res.get('solution', '')[:500])
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
            span.set_attribute("incident_status", incident_status)
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
