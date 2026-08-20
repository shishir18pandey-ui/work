import os
from observability import init_telemetry, get_tracer
import logging
import html
import pkce
import urllib.parse
# Add these to existing imports
from fastapi import FastAPI, Request, HTTPException, Depends, status, BackgroundTasks, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
import httpx
from typing import Optional,List
from starlette.middleware.sessions import SessionMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
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
from pydantic import BaseModel, Field, field_validator

from typing import Optional
import secrets
import uuid
from  file_processor import process_attachments
from oracle_connection import test_oracle_connection
from incident_db import (
    get_incident,
    get_incident_status,
    set_incident_status,
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


CLIENT_ID = os.getenv('ENT_AUTH_CLIENT_ID')
AUTHORIZE_URL = os.getenv('ENT_AUTH_URL')
TOKEN_URL = os.getenv('ENT_AUTH_TOKEN_URL')
USER_INFO_URL = os.getenv('ENT_AUTH_USER_INFO_URL')
REDIRECT_URL = os.getenv('ENT_AUTH_REDIRECT_URL', 'https://internal-app.uat-devutils.idfcfirstbank.com/incident_agent/auth/oauth/ent_auth/callback')
VALIDATION_URL = os.getenv('ENT_AUTH_VALIDATION_URL')
EXTEND_URL = os.getenv('ENT_AUTH_KEEPALIVE_URL')
SESSION_SECRET = os.getenv('SESSION_SECRET', 'incident-agent-session-secret-change-in-production')

verifier_store = {}

PUBLIC = [
    "/",
    "/incident_agent",
    "/incident_agent/login",
    "/incident_agent/auth/oauth/ent_auth/callback",
    "/incident_agent/health",
    "/incident_agent/health/oracle",
    "/incident_agent/ingest-incidents",
    "/incident_agent/incident/create",
    "/incident_agent/incident/{incident_id}"
]

class AuthMiddleware(BaseHTTPMiddleware):

    @staticmethod
    async def _is_session_valid(request: Request):
        try:
            # Check if session exists and has user data
            if not hasattr(request, 'session') or not request.session:
                templates.env.globals["user"] = None
                return False
            
            user_data = request.session.get("user")
            if not user_data:
                templates.env.globals["user"] = None
                return False
                
            access_token = user_data.get("access_token")
            if not access_token:
                templates.env.globals["user"] = None
                return False
                
            async with httpx.AsyncClient(verify=False) as client:
                validity_response = await client.get(
                    url=VALIDATION_URL,
                    headers={"Authorization": f"Bearer {access_token}", "Accept": "application/json"}
                )
                if validity_response.status_code not in range(200, 205):
                    templates.env.globals["user"] = None
                    if validity_response.status_code == 401:
                        return False
                    raise HTTPException(status_code=validity_response.status_code, detail=validity_response.text)

                templates.env.globals["user"] = user_data
                return True
        except Exception as e:
            logger.warning(f"Session validation error: {e}")
            try:
                if hasattr(request, 'session') and request.session:
                    request.session.clear()
            except Exception:
                pass
            templates.env.globals["user"] = None
            return False

    @staticmethod
    async def _is_service_token_valid(request: Request):
        try:
            service_token = request.headers.get("X-Service-Token")
            if not service_token:
                return False

            async with httpx.AsyncClient(verify=False) as client:
                validity_response = await client.get(
                    url=VALIDATION_URL,
                    headers={"Authorization": f"Bearer {service_token}", "Accept": "application/json"}
                )
                if validity_response.status_code not in range(200, 205):
                    if validity_response.status_code == 401:
                        return False
                    raise HTTPException(status_code=validity_response.status_code, detail=validity_response.text)

                return True
        except Exception:
            return False

    async def dispatch(self, request: Request, call_next):
        if request.url.path not in PUBLIC:
            if await self._is_service_token_valid(request):
                pass
            elif not await self._is_session_valid(request):
                return RedirectResponse(url="/incident_agent/login")
        response = await call_next(request)
        response.headers['X-XSS-Protection'] = '1; mode=block'
        response.headers['X-Frame-Options'] = 'DENY'
        return response

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


from contextlib import asynccontextmanager
from llm import llm_config

llm_config.set_llm_config()   # synchronous — guarantees a valid token before any request


@asynccontextmanager
async def lifespan(app:FastAPI):
    asyncio.create_task(llm_config.refresh_loop())
    yield
app = FastAPI(lifespan=lifespan)

from clean_extract_bot_data import router as export_router
app.include_router(export_router)


os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)

app.mount("/incident_agent/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

app.add_middleware(AuthMiddleware)
app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET)

logger.info("FastAPI app initialized")


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
@field_validator('shortDescription', 'businessImpact',mode='before')
@classmethod
def sanitize_text_feilds(cls,value):
    if value is None:
        return value
    return html.escape(str(value))
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


@app.get("/incident_agent/results")
def results_page(request: Request):
    recent_incidents = get_last_25_incidents()
    csp_headers = {
        "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline' https://cdn.tailwindcss.com; style-src 'self' 'unsafe-inline' https://cdn.tailwindcss.com; img-src 'self' data:;"
    }
    return templates.TemplateResponse(
        "results.html",
        {"request": request, "results": recent_incidents},
        headers=csp_headers
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
    return RedirectResponse(url="/incident_agent/login")



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
            "sysdate": result['sysdate'],
            "message": "Successfully connected to Oracle database"
        }
    else:
        logger.error(f"Oracle connection FAILED | error={result['error']}")
        return {
            "status": "unhealthy",
            "connected": False,
            "error": result['error'],
            "message": "Failed to connect to Oracle database"
        }

@app.get("/incident_agent/login")
async def login(request: Request):

    from opentelemetry import trace
    root_span = trace.get_current_span()
    if root_span and root_span.is_recording():
        root_span.update_name("GET /login [OAuth Init]")

    code_verifier, code_challenge = pkce.generate_pkce_pair()

    state = secrets.token_hex(16)

    request.session["pkce_verifier"] = code_verifier
    request.session["oauth_state"] = state

    params = {
        "client_id": CLIENT_ID,
        "response_type": "code",
        "redirect_uri": REDIRECT_URL,
        "scope": "",
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "state": state
    }

    url = f"{AUTHORIZE_URL}?{urllib.parse.urlencode(params)}"
    print(f"Redirecting to: {url}")

    return RedirectResponse(url)


@app.get("/incident_agent/logout")
async def logout(request: Request):
    request.session.clear()
    templates.env.globals["user"] = None
    return RedirectResponse(url="/incident_agent/login")


@app.get("/incident_agent/auth/oauth/ent_auth/callback")
async def callback(request: Request):
    code = request.query_params.get("code")
    state = request.query_params.get("state")

    if not code or not state:
        logger.error("Missing code or state in OAuth callback")
        return RedirectResponse("/incident_agent/login")

    expected_state = request.session.pop("oauth_state", None)
    if not expected_state or state != expected_state:
        request.session.clear()
        templates.env.globals["user"] = None
        print("[ERROR] State mismatch — possible CSRF or session expiry")
        return RedirectResponse(url="/incident_agent/login")

    code_verifier = request.session.pop("pkce_verifier", None)
    if not code_verifier:
        request.session.clear()
        templates.env.globals["user"] = None
        print("[ERROR] PKCE verifier missing from session")
        return RedirectResponse(url="/incident_agent/login")

    payload = {
        "code": code,
        "redirect_uri": REDIRECT_URL,
        "client_id": CLIENT_ID,
        "code_verifier": code_verifier
    }
    
    async with httpx.AsyncClient(verify=False) as client:
        token_response = await client.post(
            url=TOKEN_URL,
            data=json.dumps(payload),
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        logger.info("Token response %s %s", token_response.status_code, token_response.text)
        if token_response.status_code != 200:
            raise HTTPException(status_code=token_response.status_code, detail=token_response.text)

        access_token = token_response.json().get("access_token")
        expiry = token_response.json().get("expires_in")
    
    async with httpx.AsyncClient(verify=False) as client:
        user_info_response = await client.post(
            url=USER_INFO_URL,
            headers={"Authorization": f"Bearer {access_token}", "Accept": "application/json"}
        )
        logger.info("User-info response %s %s", user_info_response.status_code, user_info_response.text)

        if user_info_response.status_code != 200:
            raise HTTPException(status_code=user_info_response.status_code, detail=user_info_response.text)

        user_info = user_info_response.json()

    user = {
        "identifier": user_info.get("email"),
        "display_name": f"{user_info.get('firstName', '')} {user_info.get('lastName', '')}".strip(),
        "access_token": access_token,
        "token_expiry": expiry
    }
    request.session["user"] = user
    return RedirectResponse("/incident_agent/results")


@app.post("/incident_agent/keepalive")
async def keepalive(request: Request):
    """Extends token validity"""
    user = request.session.get("user", {})
    access_token = user.get("access_token")
    async with httpx.AsyncClient(verify=False) as client:
        resp = await client.post(
            url=EXTEND_URL,
            headers={"Authorization": f"Bearer {access_token}", "Accept": "application/json"}
        )
        if resp.status_code not in range(200, 205):
            if resp.status_code == 401:
                return RedirectResponse(url="/incident_agent/login")
    return JSONResponse({"status": 200, "message": "success"})
