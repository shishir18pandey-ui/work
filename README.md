this si my incidnet_agent repo crew main.py file :

import os
from observability import init_telemetry, get_tracer
import logging
# Add these to existing imports
from fastapi import FastAPI, Request, HTTPException, Depends, status, BackgroundTasks, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
import httpx
from typing import Optional,List
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

telemetry_endpoint = os.getenv("TELEMETRY_ENDPOINT")
if telemetry_endpoint:
    logger.info("OpenTelemetry tracer initialization started...")
    try:
        init_telemetry()
        logger.info("OpenTelemetry tracer initialized")
    except Exception as e:
        logger.warning(f"Failed to initialize OpenTelemetry: {e}")

from fastapi import FastAPI, Request, HTTPException, Depends, status, BackgroundTasks
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import asyncio
import json
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional
import secrets
import uuid

from oracle_connection import test_oracle_connection
from incident_db import (
    get_incident,
    get_incident_status,
    set_incident_status,
    upsert_incident_payload,
    set_incident_resolution,
    get_last_25_incidents
)
from incident_db_async import upsert_incident_payload_async
from confluent_kafka import Producer

load_dotenv()
JIRA_RAG_INGEST_URL = os.getenv(
    "JIRA_RAG_INGEST_URL",
    "https://internal-app.uat-devutils.idfcfirstbank.com/api/jira-conf-rag/ingest/incidents"
)
logger.info("crew_main.py loaded successfully")
logger.info(f"INCIDENT_BOT_ENABLED = {os.getenv('INCIDENT_BOT_ENABLED')}")
logger.info(f"KAFKA_BROKER_URL = {os.getenv('KAFKA_BROKER_URL')}")
logger.info(f"KAFKA_TOPIC = {os.getenv('KAFKA_TOPIC', 'GEN-AI-DE-INCIDENT-EVENTS')}")


# ─────────────────────────────────────────
# AUTH
# ─────────────────────────────────────────

def verify_dummy_header(credentials: HTTPBasicCredentials = Depends(HTTPBasic())):
    expected_username = os.getenv("INCIDENT_AUTH_USERNAME", "admin")
    expected_password = os.getenv("INCIDENT_AUTH_TOKEN", "dummy-token-12345")

    is_correct_username = secrets.compare_digest(
        credentials.username.encode("utf8"),
        expected_username.encode("utf8")
    )
    is_correct_password = secrets.compare_digest(
        credentials.password.encode("utf8"),
        expected_password.encode("utf8")
    )

    if not (is_correct_username and is_correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


# ─────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────

app = FastAPI()

from clean_extract_bot_data import router as export_router
app.include_router(export_router)

from llm import llm_config
asyncio.get_event_loop().create_task(llm_config.refresh_loop())

os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)

app.mount("/incident_agent/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

logger.info("FastAPI app initialized")


# ─────────────────────────────────────────
# KAFKA PRODUCER
# ─────────────────────────────────────────

def get_kafka_producer():
    return Producer({
        'bootstrap.servers': os.getenv("KAFKA_BROKER_URL"),
        'security.protocol': 'SASL_SSL',
        'sasl.mechanism': 'SCRAM-SHA-512',
        'sasl.username': os.getenv("KAFKA_USERNAME", "gen-ai-de_msk_uat"),
        'sasl.password': os.getenv("KAFKA_PASSWORD"),
    })


def publish_to_kafka(incident_id: str, event_type: str, payload: dict):
    tracer = get_tracer(__name__)
    with tracer.start_as_current_span("publish_to_kafka") as span:
        span.set_attribute("incident_id", incident_id)
        span.set_attribute("event_type", event_type)
        try:
            producer = get_kafka_producer()
            message = {
                "incident_id": incident_id,
                "event_type": event_type,
                "payload": payload
            }
            producer.produce(
                topic=os.getenv("KAFKA_TOPIC", "GEN-AI-DE-INCIDENT-EVENTS"),
                key=incident_id.encode('utf-8'),
                value=json.dumps(message).encode('utf-8')
            )
            producer.flush()
            logger.info(f"✓ Kafka published | incident={incident_id} event={event_type}")
        except Exception as e:
            logger.error(f"✗ Kafka publish FAILED | incident={incident_id} error={e}")
            raise




def get_header_details():
    return {
        "Content-Type": "application/json",
        "correlationId": str(uuid.uuid4()),
        "source": "IncidentBot",
        "transactionId": str(uuid.uuid4())
    }


def handle_no_comments(incident_id: str):
    tracer = get_tracer(__name__)
    with tracer.start_as_current_span("handle_no_comments") as span:
        span.set_attribute("incident_id", incident_id)
        logger.info(f"  No comments | incident={incident_id}")

from typing import List, Optional
from pydantic import BaseModel

class Attachment(BaseModel):
    fileId: Optional[str] = None
    fileName: str
    fileType: str
    fileSize: Optional[int] = None
    contentEncoding: str
    fileContent: str

# ─────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────

class ServiceNowIncidentCreateRequest(BaseModel):
    callerId: Optional[str] = None
    incidentType: Optional[str] = None
    businessService: Optional[str] = None
    tier1: Optional[str] = None
    tier2: Optional[str] = None
    tier3: Optional[str] = None
    impact: Optional[str] = None
    urgency: Optional[str] = None
    shortDescription: Optional[str] = None
    description: Optional[str] = None
    contactType: Optional[str] = None
    sourceIncidentNum: Optional[str] = None
    sourceIncidentId: Optional[str] = None
    assignmentGroup: Optional[str] = None
    businessImpact: Optional[str] = None
    cause: Optional[str] = None
    businessCorrectiveAction: Optional[str] = None
    techCorrectiveAction: Optional[str] = None
    dataSource: Optional[str] = None
    descriptionOfOutage: Optional[str] = None
    emailId: Optional[str] = None
    entityUCIC: Optional[str] = None
    hashValues: Optional[str] = None
    ipDetails: Optional[str] = None
    ldwNotifyInformation: Optional[str] = None
    loanAccountNumber: Optional[str] = None
    loginId: Optional[str] = None
    mobileNumber: Optional[str] = None
    businessPreventiveAction: Optional[str] = None
    techPreventiveAction: Optional[str] = None
    resoultionTeam: Optional[str] = None
    rootCause: Optional[str] = None
    systemName: Optional[str] = None
    urlOrDomain: Optional[str] = None
    userDetail: Optional[str] = None
    taskEffectiveNumber: Optional[str] = None
    individualUCIC: Optional[str] = None
    sourceIncCreateddttime: Optional[str] = None
    incidentId: str = Field(..., min_length=1)
    state: Optional[str] = None
    causedByPatch: Optional[str] = None
    resolutionCode: Optional[str] = None
    solutionType: Optional[str] = None
    outageType: Optional[str] = None
    vendorGroup: Optional[str] = None
    additionalComments: Optional[str] = None
    onHoldReason: Optional[str] = None
    resolutionNotes: Optional[str] = None
    userLocation: Optional[str] = None
    incidentNumber: str
    assignedTo: Optional[str] = None
    files: Optional[List[Attachment]] = None



class ServiceNowIncidentResponse(BaseModel):
    code: str
    details: str


# ─────────────────────────────────────────
# EXCEPTION HANDLER
# ─────────────────────────────────────────

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    fields = ""
    for err in exc.errors():
        field = ".".join(str(x) for x in err["loc"] if x != "body")
        fields += f", {field}" if fields else field
    logger.warning(f"Validation error | fields={fields}")
    return JSONResponse(
        status_code=422,
        content={"code": "422", "details": f"{fields} cannot be empty or absent"}
    )


# ─────────────────────────────────────────
# UI ROUTES
# ─────────────────────────────────────────

@app.get("/incident_agent/results")
def results_page(request: Request):
    recent_incidents = get_last_25_incidents()
    return templates.TemplateResponse(
        "results.html",
        {"request": request, "results": recent_incidents}
    )

@app.post("/incident_agent/ingest-incidents")
async def ingest_incidents_proxy(
    file: UploadFile = File(...),
    application: str = Form("")
):
    logger.info(f"[ingest] Received file={file.filename} application={application}")
    try:
        file_contents = await file.read()
        logger.info(f"[ingest] File size={len(file_contents)} bytes")

        async with httpx.AsyncClient(timeout=600.0, verify=False) as client:
            response = await client.post(
                JIRA_RAG_INGEST_URL,
                files={"file": (file.filename, file_contents, file.content_type or "text/csv")},
                data={"application": application}
            )

        logger.info(f"[ingest] jira-rag responded status={response.status_code}")

        try:
            response_json = response.json()
        except Exception:
            response_json = {"raw": response.text}

        return JSONResponse(content=response_json, status_code=response.status_code)

    except Exception as e:
        logger.error(f"[ingest] FAILED: {e}", exc_info=True)
        return JSONResponse(
            content={"status": "error", "message": str(e)},
            status_code=500
        )
@app.get("/incident_agent", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "incidents_received": 0,
        "incidents_resolved": 0,
        "reference_collections": 0,
        "indexed_incidents": 0
    })


# ─────────────────────────────────────────
# SERVICENOW ENDPOINTS
# ─────────────────────────────────────────

@app.post("/incident_agent/incident/create")
async def create_incident_from_servicenow(
    request: Request,
    incident_data: ServiceNowIncidentCreateRequest,
    background_tasks: BackgroundTasks,
    auth: bool = Depends(verify_dummy_header)
):
    tracer = get_tracer(__name__)

    with tracer.start_as_current_span("create_incident_from_servicenow") as span:
        incident_id = incident_data.incidentId
        span.set_attribute("incident_id", incident_id)
        span.set_attribute("has_additional_comments", bool(incident_data.additionalComments))

        logger.info(f"→ Incoming request | incident={incident_id} has_comments={bool(incident_data.additionalComments)}")

        feature = os.getenv("INCIDENT_BOT_ENABLED", False)
        logger.info(f"  INCIDENT_BOT_ENABLED={feature}")
        if not feature:
            logger.warning(f"  Bot disabled, returning 503 | incident={incident_id}")
            return ServiceNowIncidentResponse(code="503", details="Service Unavailable.")

        incident = get_incident(incident_id)
        logger.info(f"  DB lookup | incident={incident_id} exists={bool(incident)}")

        if not incident:
            # NEW INCIDENT
            incident_record = incident_data.model_dump()
            incident_record["created_at"] = datetime.now().isoformat()
            incident_record["status"] = "created"
            incident_record["interaction_counter"] = 0
            incident_record["headers"] = get_header_details()

            try:
                await upsert_incident_payload_async(incident_id, json.dumps(incident_record))
                logger.info(f"  DB saved | incident={incident_id}")
            except Exception as e:
                logger.error(f"✗ DB upsert FAILED | incident={incident_id} error={e}")
                raise HTTPException(status_code=500, detail="Incident server error")

            publish_to_kafka(incident_id, "new_incident", incident_record)
            logger.info(f"✓ New incident done | incident={incident_id}")
            return ServiceNowIncidentResponse(code="202", details="Accepted")

        else:
            # EXISTING INCIDENT
            additional_comments = incident_data.additionalComments
            payload = incident['payload']

            if additional_comments:
                current_status = get_incident_status(incident_id)
                logger.info(f"  Existing incident | incident={incident_id} status={current_status}")
                span.set_attribute("current_status", current_status or "unknown")

                if current_status == 'on_hold':
                    payload["additionalComments"] = additional_comments
                    set_incident_status(incident_id, 'in_progress')
                    publish_to_kafka(incident_id, "additional_comments", payload)
                    logger.info(f"✓ Additional comments published | incident={incident_id}")

                elif current_status == 'in_progress':
                    logger.info(f"  Ignored | incident={incident_id} reason=already_in_progress")

                elif current_status in ['resolved', 'rejected']:
                    logger.info(f"  Ignored | incident={incident_id} reason=final_status={current_status}")

                else:
                    logger.info(f"  Ignored | incident={incident_id} reason=unknown_status={current_status}")

            else:
                handle_no_comments(incident_id)

            return ServiceNowIncidentResponse(code="202", details="Accepted")


@app.get("/incident_agent/incident/{incident_id}")
async def get_incident_details(incident_id: str, auth: bool = Depends(verify_dummy_header)):
    tracer = get_tracer(__name__)
    with tracer.start_as_current_span("get_incident_details") as span:
        span.set_attribute("incident_id", incident_id)
        logger.info(f"→ Get incident details | incident={incident_id}")
        incident = get_incident(incident_id)
        if not incident:
            logger.warning(f"  Not found | incident={incident_id}")
            raise HTTPException(status_code=404, detail="Incident not found")
        return incident['payload']


@app.get("/incident_agent/nudge_incident/{incident_id}")
async def nudge_incident(incident_id: str):
    tracer = get_tracer(__name__)
    with tracer.start_as_current_span("nudge_incident") as span:
        span.set_attribute("incident_id", incident_id)
        logger.info(f"→ Nudge incident | incident={incident_id}")
        incident = get_incident(incident_id)
        if not incident:
            logger.warning(f"  Not found | incident={incident_id}")
            raise HTTPException(status_code=404, detail="Incident not found")
        payload = incident['payload']
        publish_to_kafka(incident_id, "new_incident", payload)
        logger.info(f"✓ Nudged | incident={incident_id}")
        return {"status": "nudged"}


# ─────────────────────────────────────────
# HEALTH ENDPOINTS
# ─────────────────────────────────────────

@app.get("/incident_agent/health")
async def health_check():
    logger.info("Health check called")
    return {"status": "healthy"}


@app.get("/incident_agent/health/oracle")
async def oracle_health_check():
    logger.info("Oracle health check called")
    result = test_oracle_connection()
    if result["connected"]:
        logger.info(f"Oracle connected | sysdate={result['sysdate']}")
        return {
            "status": "healthy",
            "connected": True,
            "sysdate": result["sysdate"],
            "message": "Successfully connected to Oracle database"
        }
    else:
        logger.error(f"Oracle connection FAILED | error={result['error']}")
        return {
            "status": "unhealthy",
            "connected": False,
            "error": result["error"],
            "message": "Failed to connect to Oracle database"
        }






and now im thinking to process the image before sending in kafa  beacuse in kafka i have thi issue of memeory byte whihc is it can conatin only 1.2 mb f files but image can be of 2 3 4 5 mb right , so why not process those image before publishing this to kafka , in paylaod i will add image decriitopn feild and add the processed image descriton , so later on it ownt eb a  issue for our system base don image sixe :


second thing in my incidnt mnaager repo i will just pass the image decsritopn feildf in paylod and  and lter on willa dd that to my incidne t descritpion which is existing right :


so bekow are teh chnages which u did for image prcessing , now i will share u file before the image processing code , so you can fix the old file it would be more easy :




flow.py from incidemnt manager repo :
from pydantic import BaseModel
from typing import List, Optional, Dict
import httpx
import os
import json
import re
from dotenv import load_dotenv
from crewai.flow.flow import Flow, start, listen, router
from crewai.flow.persistence import persist
from utils.incident_db_async import upsert_incident_payload_async
from agents.debugger import run_backend_resolver_crew_async
from agents.context_builder import run_incident_context_crew_async
from agents.intent_classifier import run_intent_classifier_crew_async
from utils.llm import run_crew_with_retry_async

load_dotenv()

CA_CERT_FILE = os.getenv("CA_CERT_FILE", "./IDFCBANKCA.pem")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "/app/models/gpt-oss-120b")
CHAT_COMPLETIONS_URL = f"{OPENAI_API_BASE}/chat/completions"

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
    
    # Try to extract JSON from markdown code blocks
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
    
    # Try to find JSON-like object in the text
    # Look for {...} pattern
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
    final_output_json: dict = {}
    snow_status: str = ""
    ucic: str = ""
    current_comment: Optional[str] = None

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
            "callerId": payload.get("callerId"),
            "incidentType": payload.get("incidentType"),
            "businessService": payload.get("businessService"),
            "tier1": payload.get("tier1"),
            "tier2": payload.get("tier2"),
            "tier3": payload.get("tier3"),
            "impact": payload.get("impact"),
            "urgency": payload.get("urgency"),
            "shortDescription": payload.get("shortDescription"),
            "description": payload.get("description"),
            "contactType": payload.get("contactType"),
            "sourceIncidentNum": payload.get("sourceIncidentNum"),
            "sourceIncidentId": payload.get("sourceIncidentId"),
            "assignmentGroup": payload.get("assignmentGroup"),
            "incidentId": payload.get("incidentId"),
            "state": payload.get("state"),
            "causedByPatch": payload.get("causedByPatch"),
            "resolutionCode": payload.get("resolutionCode"),
            "solutionType": payload.get("solutionType"),
            "outageType": payload.get("outageType"),
            "additionalComments": question,
            "resolutionNotes": resolution,
            "cause": payload.get("cause"),
            "onHoldReason": payload.get("onHoldReason"),
            "correlationDisplay": payload.get("correlationDisplay"),
            "vendorGroup": payload.get("vendorGroup")
        }

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
        payload.update({"state": "On Hold","cause": "Assign to an Engineer."})
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
           #"cause": "Resolved by Bot.",
           #"state": "Resolved",
           # "resolutionCode": "Solved (Permanently)",
            #"solutionType": "Other",
            #"outageType": "No Outage"
            "state":"On Hold",
            "onHoldReason": "User Action Required"
        })
        result = await send_update_to_servicenow_async(payload, None, resolution)
        return result, 'on_hold'    


def payload_to_incident_description(payload):
    tracer = get_tracer(__name__)
    with tracer.start_as_current_span("payload_to_incident_description") as span:
        span.set_attribute("short_description_length", len(payload.get('shortDescription','')))
        span.set_attribute("description_length", len(payload.get('description','')))
        span.set_attribute("individualUCIC", payload.get('individualUCIC','i'))
        
        short_description = payload.get('shortDescription','')
        description = payload.get('description','')
        individualUCIC = payload.get('individualUCIC','i')
        # individualUCIC = individualUCIC[:-1]
        result = f"Short Description: {short_description}\nDescription: {description}"
        print(f"Generated incident description for UCIC {individualUCIC}")
        return result, individualUCIC



# @persist(key="incident_id")
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

            comment = self.state.current_comment if self.state.current_comment else "NA"

            self.state.intent = await run_crew_with_retry_async(
                lambda: run_intent_classifier_crew_async(
                    self.state.incident_description, 
                    self.state.user_qa_pairs,
                    comment
                )
            )
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
            print(f"Fresh incident {self.state.incident_id}, gather context")
            return "gather_context"


    @listen('gather_context')
    async def semantic_search(self):
        tracer = get_tracer(__name__)
        with tracer.start_as_current_span("gather_context") as span:
            span.set_attribute("incident_id", self.state.incident_id)
            span.set_attribute("incident_description_length", len(self.state.incident_description))

            incident_description, ucic = payload_to_incident_description(self.state.payload)
            desc = f"{incident_description}\nUCIC: {ucic}"
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

            # ── read app from payload ──
            app = self.state.payload.get("tier1", "cbs").lower().strip()
            span.set_attribute("app", app)
            logger.info(f"Resolver | incident={self.state.incident_id} app={app}")

            raw_resolution = await run_crew_with_retry_async(
                lambda: run_backend_resolver_crew_async(
                    self.state.incident_description,
                    self.state.incident_context,
                    self.state.user_qa_pairs,
                    self.state.ucic,
                    self.state.current_comment,
                    app=app    
                )
            )

            self.state.final_output_json = extract_json_from_output(raw_resolution)
            print(f"Resolver task completed for incident {self.state.incident_id}")
            return "update_servicenow"

    @listen(run_resolver_crew)
    async def update_servicenow(self):
        tracer = get_tracer(__name__)
        with tracer.start_as_current_span("update_servicenow") as span:
            span.set_attribute("incident_id", self.state.incident_id)
            span.set_attribute("resolution_result", self.state.final_output_json.get("resolved", "unknown"))

            res = self.state.final_output_json
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





            this is worker.py :
            import os
import json
import logging

from utils.oracle_connection import close_oracle_async_pool
from utils.observability import init_telemetry, get_tracer
from utils.llm import llm_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

telemetry_endpoint = os.getenv("TELEMETRY_ENDPOINT")

if telemetry_endpoint:
    try:
       
        init_telemetry()
        
        logger.info("OpenTelemetry initialized for incident-manager worker")
    except Exception as e:
        logger.warning(f"Failed to initialize OpenTelemetry: {e}")
else:
    logger.warning("TELEMETRY_ENDPOINT not set — running without tracing")


import asyncio
import signal
import threading
import time
from confluent_kafka import Consumer, KafkaError, KafkaException
from dotenv import load_dotenv

load_dotenv()

KAFKA_BROKER   = os.getenv("KAFKA_BROKER_URL")
KAFKA_TOPIC    = os.getenv("KAFKA_TOPIC", "GEN-AI-DE-INCIDENT-EVENTS")
KAFKA_GROUP_ID = os.getenv("KAFKA_GROUP_ID", "gen-ai-de-incident-managers")
KAFKA_USERNAME = os.getenv("KAFKA_USERNAME")
KAFKA_PASSWORD = os.getenv("KAFKA_PASSWORD")

running = True
semaphore = None
active_tasks = {}  # {task: (incident_id, payload, start_time)}
TIMEOUT_SECONDS = 3600  # 1 hour
print(telemetry_endpoint,"tetsing1")
logger.info(f"Worker config loaded")
logger.info(f"  KAFKA_BROKER    : {KAFKA_BROKER}")
logger.info(f"  KAFKA_TOPIC     : {KAFKA_TOPIC}")
logger.info(f"  KAFKA_GROUP_ID  : {KAFKA_GROUP_ID}")


def get_kafka_consumer():
    config = {
        'bootstrap.servers': KAFKA_BROKER,
        'group.id': KAFKA_GROUP_ID,
        'auto.offset.reset': 'earliest',
        'enable.auto.commit': False,
        'security.protocol': 'SASL_SSL',
        'sasl.mechanism': 'SCRAM-SHA-512',
        'sasl.username': KAFKA_USERNAME,
        'sasl.password': KAFKA_PASSWORD,
        'session.timeout.ms': 60000,
        'max.poll.interval.ms': 7200000,
    }
    return Consumer(config)


async def process_message(incident_id: str, event_type: str, payload: dict):
    from flow import IncidentManagementFlow, send_rejection_to_servicenow_async
    from utils.incident_db_async import get_incident_status_async, upsert_incident_payload_async

    tracer = get_tracer(__name__)

    async with semaphore:
        with tracer.start_as_current_span("process_message") as span:
            span.set_attribute("incident_id", incident_id)
            span.set_attribute("event_type", event_type)

            logger.info(f"→ Processing | incident={incident_id} event={event_type}")

            current_status = await get_incident_status_async(incident_id)
            logger.info(f"  DB status | incident={incident_id} status={current_status}")
            span.set_attribute("current_status", current_status or "unknown")

            if event_type == "new_incident":
                if current_status in ['resolved', 'rejected']:
                    logger.info(f"  Skipping | incident={incident_id} reason=already_{current_status}")
                    span.set_attribute("skipped", True)
                    return

            if event_type == "additional_comments":
                if current_status not in ['in_progress']:
                    logger.info(f"  Skipping | incident={incident_id} reason=status_{current_status}_not_in_progress")
                    span.set_attribute("skipped", True)
                    return

            flow = IncidentManagementFlow()
            flow.state.incident_id = incident_id
            flow.state.payload = payload

            if event_type == "additional_comments":
                comments = payload.get("additionalComments", "")
                flow.state.current_comment = comments
                logger.info(f"  Flow type | incident={incident_id} type=additional_comments")
                span.set_attribute("flow_type", "additional_comments")
            else:
                logger.info(f"  Flow type | incident={incident_id} type=new_incident")
                span.set_attribute("flow_type", "new_incident")

            try:
                await flow.akickoff()
                logger.info(f"✓ Flow completed | incident={incident_id}")
                span.set_attribute("flow_status", "completed")

            except asyncio.CancelledError:
                # ── FIX: timeout_checker_thread already sent rejection ──
                # Just log and update DB — do NOT send to ServiceNow again
                logger.error(f"Task cancelled (timeout) | incident={incident_id}")
                span.set_attribute("flow_status", "cancelled")
                try:
                    await upsert_incident_payload_async(
                        incident_id,
                        json.dumps(payload),
                        'Failed - Timeout'
                    )
                except Exception as db_err:
                    logger.error(f"DB update failed after cancel | incident={incident_id} error={db_err}")
                raise  # re-raise so asyncio knows task was cancelled

            except Exception as e:
                logger.error(f"✗ Flow FAILED | incident={incident_id} error={e}")
                span.set_attribute("flow_status", "failed")
                span.set_attribute("error", str(e))
                span.record_exception(e)
                (status, info), _ = await send_rejection_to_servicenow_async(payload)
                payload['__agent_data']['snow_logs'].append({
                    "type": "rejection", "status": status, "response": info
                })
                await upsert_incident_payload_async(
                    incident_id,
                    json.dumps(payload),
                    'Failed - LLM Error'
                )
                logger.info(f"  Fallback rejection sent | incident={incident_id}")

def timeout_checker_thread():
    global active_tasks, running
    while running:
        time.sleep(10)
        now = time.time()
        for task, (incident_id, payload, start_time) in list(active_tasks.items()):
            if task not in active_tasks:
                continue
            if task.done():
                continue
            if now - start_time > TIMEOUT_SECONDS:
                logger.error(f"Timeout | incident={incident_id} - sending rejection")
                if task in active_tasks:
                    del active_tasks[task]
                task.cancel()
                logger.info(f"Task cancelled for timeout | incident={incident_id}")


async def run_consumer():
    global semaphore, active_tasks

    asyncio.create_task(llm_config.refresh_loop())
    logger.info("LLM config initialized")

    semaphore = asyncio.Semaphore(10)
    active_tasks = {}

    tracer = get_tracer(__name__)
    consumer = get_kafka_consumer()

    timeout_thread = threading.Thread(target=timeout_checker_thread, daemon=True)
    timeout_thread.start()
    logger.info("Timeout checker thread started")

    try:
        consumer.subscribe([KAFKA_TOPIC])
        logger.info(f"✓ Worker started")
        logger.info(f"  Topic    : {KAFKA_TOPIC}")
        logger.info(f"  Group    : {KAFKA_GROUP_ID}")
        logger.info(f"  Broker   : {KAFKA_BROKER}")

        while running:
            msg = await asyncio.get_event_loop().run_in_executor(
                None, lambda: consumer.poll(1.0)
            )

            if msg is None:
                continue

            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    logger.debug(f"End of partition {msg.partition()}")
                    continue
                else:
                    logger.error(f"Kafka error: {msg.error()}")
                    raise KafkaException(msg.error())

            incident_id = None
            try:
                raw = msg.value().decode('utf-8')
                data = json.loads(raw)

                incident_id = data.get("incident_id")
                event_type  = data.get("event_type", "new_incident")
                payload     = data.get("payload", {})

                logger.info(f"→ Message received | incident={incident_id} event={event_type} partition={msg.partition()} offset={msg.offset()}")

                if not incident_id:
                    logger.warning("Message missing incident_id, skipping")
                    consumer.commit(message=msg)
                    continue

                with tracer.start_as_current_span("kafka_message_received") as span:
                    span.set_attribute("incident_id", incident_id)
                    span.set_attribute("event_type", event_type)
                    span.set_attribute("partition", msg.partition())
                    span.set_attribute("offset", msg.offset())

                task = asyncio.create_task(process_message(incident_id, event_type, payload))
                active_tasks[task] = (incident_id, payload, time.time())

                # Auto-remove from event loop when done
                def cleanup_done(t):
                    if t in active_tasks:
                        del active_tasks[t]
                task.add_done_callback(cleanup_done)

                consumer.commit(message=msg)
                logger.info(f"✓ Offset committed | incident={incident_id}")

            except Exception as e:
                logger.error(f"✗ Message handling FAILED | incident={incident_id} error={e}")
                await asyncio.sleep(5)

    except KeyboardInterrupt:
        logger.info("Worker interrupted")
    finally:
        logger.info("Closing consumer...")
        consumer.close()
        logger.info("Consumer closed cleanly")


def handle_shutdown(signum, frame):
    global running
    logger.info(f"Shutdown signal {signum} received...")
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.call_soon(lambda: asyncio.create_task(close_oracle_async_pool()))
        else:
            asyncio.run(close_oracle_async_pool())
    except RuntimeError:
        asyncio.run(close_oracle_async_pool())
    running = False


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, handle_shutdown)
    signal.signal(signal.SIGINT, handle_shutdown)
    
    logger.info("Starting incident-manager worker...")
    asyncio.run(run_consumer())




this si file processor.py:
import os
import base64
import logging
import requests
from typing import List, Dict, Optional
from utils.llm import llm_config

logger = logging.getLogger(__name__)

VISION_MODEL_NAME = os.getenv("VISION_MODEL_NAME", "/app/models/Qwen3-VL-8B-Instruct")
VISION_API_BASE = os.getenv("VISION_API_BASE")

IMAGE_MIME_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/gif", "image/webp"}
PDF_MIME_TYPES = {"application/pdf"}


def _describe_image(base64_data: str, mime_type: str, file_name: str) -> str:
    data_uri = f"data:{mime_type};base64,{base64_data}"

    payload = {
        "model": VISION_MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Describe what is in this image. If it contains text "
                            "(screenshot, error message, document, ID card), transcribe "
                            "the visible text exactly. Be concise. No speculation."
                        )
                    },
                    {"type": "image_url", "image_url": data_uri}
                ]
            }
        ],
        "temperature": 0.0,
        "max_tokens": 800
    }

    headers = {
        "Authorization": f"Bearer {llm_config.token}",
        "Content-Type": "application/json"
    }

    base_url = VISION_API_BASE or llm_config.url

    try:
        response = requests.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=60,
            verify='./IDFCBANKCA.pem'
        )
        response.raise_for_status()
        data = response.json()
        description = data["choices"][0]["message"]["content"]
        logger.info(f"Image described | file={file_name}")
        return description.strip()
    except Exception as e:
        logger.error(f"Image description failed | file={file_name} err={e}")
        return "IMAGE UNREADABLE"


def _extract_pdf(base64_data: str, file_name: str) -> str:
    try:
        import io
        from pypdf import PdfReader

        pdf_bytes = base64.b64decode(base64_data)
        reader = PdfReader(io.BytesIO(pdf_bytes))

        text_parts = []
        for i, page in enumerate(reader.pages):
            page_text = (page.extract_text() or "").strip()
            if page_text:
                text_parts.append(f"[Page {i + 1}]\n{page_text}")

        if text_parts:
            combined = "\n\n".join(text_parts)
            if len(combined) > 5000:
                combined = combined[:5000] + "\n...(truncated)"
            logger.info(f"PDF extracted | file={file_name}")
            return combined
        else:
            return "PDF appears to be scanned; no extractable text."
    except Exception as e:
        logger.error(f"PDF extraction failed | file={file_name} err={e}")
        return "PDF UNREADABLE"


def _process_one(file: Dict) -> Optional[str]:
    file_name = file.get("fileName", "unknown")
    file_type = (file.get("fileType") or "").lower()
    encoding = (file.get("contentEncoding") or "").lower()
    content = file.get("fileContent")

    if not content or encoding != "base64":
        return None

    if file_type in IMAGE_MIME_TYPES:
        desc = _describe_image(content, file_type, file_name)
        return f"[Attached image: {file_name}]\n{desc}"

    if file_type in PDF_MIME_TYPES:
        text = _extract_pdf(content, file_name)
        return f"[Attached PDF: {file_name}]\n{text}"

    logger.warning(f"Skipping unsupported file type | file={file_name} type={file_type}")
    return None

print("testing4")
def process_attachments(files: Optional[List[Dict]]) -> str:
    """
    Sync file processor. Returns "" if no files or any failure.
    Never raises — incident flow must continue regardless.
    """
    if not files:
        return ""

    try:
        descriptions = []
        for f in files:
            result = _process_one(f)
            if result:
                descriptions.append(result)
        if not descriptions:
            return ""
        return "\n\n--- ATTACHED FILES ---\n" + "\n\n".join(descriptions)
    except Exception as e:
        logger.error(f"process_attachments failed: {e}")
        return ""
