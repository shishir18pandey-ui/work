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


this is ummary_agent.pyimport os
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


async def run_incident_context_deterministic_async(incident_description: str, application: str, top_k: int = 5) -> str:
    """
    Used only for apps with no Jaeger/ELK config (context-only resolution).
    Calls the search API directly instead of via LLM tool-calling, since these
    apps have no Jaeger/ELK fallback if the search silently fails or a tool
    call leaks (as seen with MiniMax), so this path must not depend on
    tool-calling reliability.
    """
    from crewai import Agent, Task, Crew, Process, LLM
    from typing import List, Dict
    from new_flow.utils.http_calls import http_client_post_async
    from new_flow.utils.llm import OPENAI_MODEL_NAME, llm_config
    from utils.observability import get_tracer

    logger = logging.getLogger(__name__)
    semantic_search_endpoint = os.getenv("SEMANTIC_SEARCH_ENDPOINT")

    async def similarity_search_api_async(index: str, query_string: str, application: str) -> List[Dict]:
        headers = {"Content-Type": "application/json"}
        body = {"metadata": {"application": application}, "query": query_string, "index_source": index}

        logger.info(
            f"[ContextBuilderDeterministic] REQUEST START | endpoint={semantic_search_endpoint} "
            f"application={application} index={index} query_len={len(query_string)}"
        )
        start_t = time.monotonic()
        try:
            response = await http_client_post_async(semantic_search_endpoint, headers=headers, json=body)
        except Exception as e:
            logger.error(
                f"[ContextBuilderDeterministic] REQUEST FAILED | application={application} "
                f"elapsed={time.monotonic()-start_t:.2f}s exception_type={type(e).__name__} message='{str(e)}'",
                exc_info=True
            )
            raise

        elapsed = time.monotonic() - start_t
        logger.info(
            f"[ContextBuilderDeterministic] REQUEST DONE | application={application} "
            f"elapsed={elapsed:.2f}s status_code={response.status_code}"
        )

        json_resp: Dict = response.json
        results: List[Dict] = list(json_resp) if json_resp else []

        logger.info(
            f"[ContextBuilderDeterministic] PARSED | elapsed={elapsed:.2f}s "
            f"result_count={len(results)} raw_type={type(json_resp).__name__}"
        )

        return results

    tracer = get_tracer(__name__)

    with tracer.start_as_current_span("_run_incident_context_deterministic_async") as span:
        span.set_attribute("incident_description", incident_description[:100] + "..." if len(incident_description) > 100 else incident_description)
        span.set_attribute("application", application)

        try:
            results = await similarity_search_api_async(
                index='incidents',
                query_string=incident_description,
                application=application
            )
        except Exception:
            return "No similar historic incidents found (search temporarily unavailable)."

        formatted_results = format_incidents_for_llm(results)

        if not results:
            return formatted_results  # "No similar historic incidents found."

        # Summarize already-fetched real results — no tools attached, nothing to leak.
        llm = LLM(
            model="openai/" + OPENAI_MODEL_NAME,
            temperature=0.0,
            base_url=llm_config.url,
            api_key=llm_config.token
        )

        agent = Agent(
            role="Historic Incident Analyst",
            goal="Summarize similar past incidents and their resolutions",
            backstory=(
                "You are a senior incident analyst. You have already been given a set of "
                "similar historic incidents retrieved from the database. Your job is only "
                "to analyze and summarize them — you do not need to search for anything yourself."
            ),
            tools=[],
            verbose=False,
            allow_delegation=False,
            llm=llm,
            temperature=0,
            max_iter=2,
            reasoning=False,
            max_retry_limit=1
        )

        task = Task(
            name="Summarize similar incidents",
            description=(
                f"Current incident:\n\n{incident_description}\n\n"
                f"=== HISTORIC SIMILAR INCIDENTS (already retrieved) ===\n{formatted_results}\n\n"
                "Analyze the results above and provide a summary of:\n"
                "1. The most relevant similar incidents found\n"
                "2. How those incidents were resolved\n"
                "3. Key troubleshooting steps from the conversation context\n"
                "4. Any patterns or common solutions that could apply to the current incident\n\n"
                "Only use the incidents provided above. Do not invent or assume incidents not listed."
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
        logger.info(f"[ContextBuilderDeterministic] CREW KICKOFF START | application={application}")

        with tracer.start_as_current_span("crew_kickoff_async") as crew_span:
            crew_span.set_attribute("incident_description", incident_description + "..." if len(incident_description) > 100 else incident_description)
            output = str(await crew.akickoff())

        crew_elapsed = time.monotonic() - crew_start
        logger.info(
            f"[ContextBuilderDeterministic] CREW KICKOFF DONE | application={application} "
            f"elapsed={crew_elapsed:.2f}s output_len={len(output)}"
        )

        return output
