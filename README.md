import os
from observability import init_telemetry, get_tracer
import logging
import re
from contextlib import asynccontextmanager
# Add these to existing imports
from fastapi import FastAPI, Request, HTTPException, Depends, status, BackgroundTasks, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
import httpx
from typing import Optional, List
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
from pydantic import BaseModel, Field, validator
from typing import Optional
import secrets
import uuid
from file_processor import process_attachments
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
# CONTENT VALIDATION (malicious script/code detection)
# ─────────────────────────────────────────

_MALICIOUS_PATTERNS = [
    re.compile(r'<\s*script', re.IGNORECASE),
    re.compile(r'<\s*iframe', re.IGNORECASE),
    re.compile(r'javascript\s*:', re.IGNORECASE),
    re.compile(r'on(error|load|click|mouseover)\s*=', re.IGNORECASE),
    re.compile(r'\bimport\s+os\b', re.IGNORECASE),
    re.compile(r'\bimport\s+subprocess\b', re.IGNORECASE),
    re.compile(r'\bexec\s*\(', re.IGNORECASE),
    re.compile(r'\beval\s*\(', re.IGNORECASE),
    re.compile(r'os\.system\s*\(', re.IGNORECASE),
    re.compile(r'__import__\s*\(', re.IGNORECASE),
    re.compile(r'ignore\s+(all\s+)?previous\s+instructions', re.IGNORECASE),
    re.compile(r'<\|im_start\|>', re.IGNORECASE),
    re.compile(r'disregard\s+.*\binstructions\b', re.IGNORECASE),
]


def contains_malicious_content(value: str) -> bool:
    if not value:
        return False
    for pattern in _MALICIOUS_PATTERNS:
        if pattern.search(value):
            return True
    return False


# ─────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────

from llm import llm_config

llm_config.set_llm_config()   # synchronous — guarantees a valid token before any request


@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    asyncio.create_task(llm_config.refresh_loop())
    yield
    # shutdown — nothing needed here currently


app = FastAPI(lifespan=lifespan)

from clean_extract_bot_data import router as export_router
app.include_router(export_router)

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

    @validator('shortDescription', 'businessImpact')
    def validate_no_malicious_content(cls, v):
        if v and contains_malicious_content(v):
            raise ValueError("Invalid content detected — field contains disallowed script or code patterns")
        return v


class ServiceNowIncidentResponse(BaseModel):
    code: str
    details: str


# ─────────────────────────────────────────
# EXCEPTION HANDLER
# ─────────────────────────────────────────

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    fields = ""
    malicious_errors = []
    for err in exc.errors():
        field = ".".join(str(x) for x in err["loc"] if x != "body")
        fields += f", {field}" if fields else field
        if "Invalid content detected" in err.get("msg", ""):
            malicious_errors.append(f"{field}: {err['msg']}")

    logger.warning(f"Validation error | fields={fields}")

    if malicious_errors:
        return JSONResponse(
            status_code=422,
            content={"code": "422", "details": "; ".join(malicious_errors)}
        )

    return JSONResponse(
        status_code=422,
        content={"code": "422", "details": f"{fields} cannot be empty or absent"}
    )


# ─────────────────────────────────────────
# UI ROUTES
# ─────────────────────────────────────────

@app.get("/incident_agent/results")
def results_page(request: Request, auth: bool = Depends(verify_dummy_header)):
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
            file_description = ""
            if incident_data.files:
                file_dicts = [f.model_dump() for f in incident_data.files]
                file_description = process_attachments(file_dicts)
                logger.info(f"  Files processed | incident={incident_id} count={len(file_dicts)} has_description={bool(file_description)}")

            incident_record = incident_data.model_dump()
            incident_record.pop("files", None)
            incident_record["file_description"] = file_description
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

                    if incident_data.files:
                        file_dicts = [f.model_dump() for f in incident_data.files]
                        new_file_description = process_attachments(file_dicts)
                        if new_file_description:
                            payload["file_description"] = new_file_description
                            logger.info(f"  Follow-up files processed | incident={incident_id}")

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












content-validator.py

import re

_MALICIOUS_PATTERNS = [
    re.compile(r'<\s*script', re.IGNORECASE),
    re.compile(r'<\s*iframe', re.IGNORECASE),
    re.compile(r'javascript\s*:', re.IGNORECASE),
    re.compile(r'on(error|load|click|mouseover)\s*=', re.IGNORECASE),
    re.compile(r'\bimport\s+os\b', re.IGNORECASE),
    re.compile(r'\bimport\s+subprocess\b', re.IGNORECASE),
    re.compile(r'\bexec\s*\(', re.IGNORECASE),
    re.compile(r'\beval\s*\(', re.IGNORECASE),
    re.compile(r'os\.system\s*\(', re.IGNORECASE),
    re.compile(r'__import__\s*\(', re.IGNORECASE),
    re.compile(r'ignore\s+(all\s+)?previous\s+instructions', re.IGNORECASE),
    re.compile(r'<\|im_start\|>', re.IGNORECASE),
    re.compile(r'disregard\s+.*\binstructions\b', re.IGNORECASE),
]

def contains_malicious_content(value: str) -> bool:
    if not value:
        return False
    for pattern in _MALICIOUS_PATTERNS:
        if pattern.search(value):
            return True
    return False
