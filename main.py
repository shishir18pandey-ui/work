from asyncio import Semaphore
from typing import Callable, LiteralString, Optional
from fastapi import Depends, FastAPI, Request, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from observability import init_tracer, attach_opentelemetry, is_tracing_enabled, get_tracer

import os
import logging
from datetime import datetime
from contextlib import asynccontextmanager
from io import BytesIO

import json
from dotenv import load_dotenv
import pkce
import urllib.parse
from pydantic import BaseModel


from observability import init_tracer, attach_opentelemetry, is_tracing_enabled

from sopagent.compare_documents import compare
from sopagent.db import Database, db, init_db, close_db
from sopagent.document_analyzer import DocumentAnalyzer, perform_automatic_analysis
from sopagent.knowledge_bot_client import KnowledgeBotClient
from sopagent.document_parser import serialize, parse, get_docling_doc
from docling_core.types.io import DocumentStream
from httpx import AsyncClient
from sopagent.util import HttpClient, getenv, AuditLog
from sopagent.llm import chat_completion
import sopagent.s3_utils as s3

CLIENT_ID = getenv('ENT_AUTH_CLIENT_ID')
AUTHORIZE_URL = getenv('ENT_AUTH_URL')
TOKEN_URL = getenv('ENT_AUTH_TOKEN_URL')
USER_INFO_URL = getenv('ENT_AUTH_USER_INFO_URL')
REDIRECT_URL = getenv('ENT_AUTH_REDIRECT_URL')
VALIDATION_URL = getenv('ENT_AUTH_VALIDATION_URL')
EXTEND_URL = getenv('ENT_AUTH_KEEPALIVE_URL')

load_dotenv()

# Configure logging FIRST
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)

# Initialize OpenTelemetry BEFORE FastAPI app
if is_tracing_enabled():
    logger.info(" Initializing OpenTelemetry for sop-agent...")
    init_tracer()
    logger.info("OpenTelemetry tracer initialized")
else:
    logger.warning(" OpenTelemetry is DISABLED (TELEMETRY_SERVICE_NAME not set)")

templates = Jinja2Templates(directory="templates")

http_client = HttpClient(
    async_client=AsyncClient(
        timeout=int(getenv('LLM_CLIENT__TIMEOUT', '60')),
        verify=False
    )
)

audit_log = AuditLog(db)

knowledge_bot_client = KnowledgeBotClient(
    http_client=http_client,
    document_path=getenv('KNOWLEDGE_BOT_DOCUMENT_PATH', 'http://localhost:8001/document')
)

semaphore = Semaphore(int(getenv('CONFLICT_ANALYSIS__MAX_CONCURRENCY', '5')))

async def chat_completion_with_semaphore(messages, tools, http_client, config_type='regular'):
    async with semaphore:
        return await chat_completion(messages, tools, http_client, config_type)

async def compare_documents(existing_document, incoming_document):
    return await compare(
        chat_completion=chat_completion_with_semaphore,
        chunk_size=int(getenv('CONFLICT_ANALYSIS__CHUNK_SIZE', '10000')),
        existing_document=existing_document,
        incoming_document=incoming_document
    )

document_analyser = DocumentAnalyzer(chat_completion, compare_documents)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield
    await close_db()


# -- Guardrails --

# A list of paths that do not require authentication
PUBLIC = ["/", '/login', '/auth/oauth/ent_auth/callback', '/health']

class AuthMiddleware(BaseHTTPMiddleware):

    @staticmethod
    async def _is_session_valid(request: Request):
        try:
            access_token = request.session.get("user", {}).get("access_token")
            if not access_token:
                templates.env.globals["user"] = None
                return False
                
            async with AsyncClient(verify=False) as client:
                validity_response = await client.get(
                    url=VALIDATION_URL,
                    headers={"Authorization": f"Bearer {access_token}",  "Accept": "application/json",}
                )
                if validity_response.status_code not in range(200, 205):
                    templates.env.globals["user"] = None
                    if validity_response.status_code == 401:
                        return False
                    raise HTTPException(status_code=validity_response.status_code, detail=validity_response.text)

                templates.env.globals["user"] = request.session.get("user", {})
                return True
        except Exception:
            request.session.clear()
            templates.env.globals["user"] = None
            return False

    @staticmethod
    async def _is_service_token_valid(request: Request):
        try:
            service_token = request.headers.get("X-Service-Token")
            if not service_token:
                return False

            async with AsyncClient(verify=False) as client:
                validity_response = await client.get(
                    url=VALIDATION_URL,
                    headers={"Authorization": f"Bearer {service_token}",  "Accept": "application/json",}
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
                return RedirectResponse(url="/")
        response = await call_next(request)
        response.headers['X-XSS-Protection'] = '1; mode=block'
        response.headers['X-Frame-Options'] = 'DENY'
        return response


# Create FastAPI app
app = FastAPI(lifespan=lifespan)
app.add_middleware(AuthMiddleware)
app.add_middleware(SessionMiddleware, secret_key=getenv('SESSION_SECRET', 'my-session'))

# Instrument FastAPI AFTER app creation
if is_tracing_enabled():
    logger.info(" Attaching OpenTelemetry to FastAPI...")
    attach_opentelemetry(app)
    logger.info("FastAPI instrumented with OpenTelemetry")

os.makedirs("uploads", exist_ok=True)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")


# -- Auth --
verifier_store = {}
# Store PKCE verifier temporarily per session
# Later can use a persistent session store or database

@app.get("/", response_class=HTMLResponse)
async def basic():
    return RedirectResponse(url="/login")

@app.get("/login", response_class=HTMLResponse)
async def login():
    code_verifier, code_challenge = pkce.generate_pkce_pair()
    session_id = code_verifier[:8]

    verifier_store[session_id] = code_verifier

    params = {
        "client_id": CLIENT_ID,
        "response_type": "code",
        "redirect_url": REDIRECT_URL,
        "scope": "offline_access openid",
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "state": session_id
    }
    query = urllib.parse.urlencode(params, safe=":", quote_via=urllib.parse.quote)
    url = f"{AUTHORIZE_URL}?{query}"
    return RedirectResponse(url)

@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    templates.env.globals["user"] = None
    return RedirectResponse(url="/")

@app.get("/auth/oauth/ent_auth/callback")
async def callback(request: Request):
    code = request.query_params.get("code")
    state = request.query_params.get("state")

    if not code or not state:
        logger.error("Missing code or state")
        return RedirectResponse("/login")

    code_verifier = verifier_store.pop(state, None)
    if not code_verifier:
        request.session.clear()
        templates.env.globals["user"] = None
        logger.error("Invalid session state. Please login again.")
        return RedirectResponse("/login")

    payload = {
        "code": code,
        "redirect_uri": REDIRECT_URL,
        "client_id": CLIENT_ID,
        "code_verifier": code_verifier
    }
    # Exchange code for token
    async with AsyncClient(verify=False) as client:
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
    # Get user info
    async with AsyncClient(verify=False) as client:
        user_info_response = await client.post(
            url=USER_INFO_URL,
            headers={"Authorization": f"Bearer {access_token}",  "Accept": "application/json",}
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
    return RedirectResponse("/dashboard")

@app.post("/keepalive")
async def keepalive(request: Request):
    user = request.session.get("user", {})
    access_token = user.get("access_token")
    async with AsyncClient(verify=False) as client:
        resp = await client.post(
            url=EXTEND_URL,
            headers={"Authorization": f"Bearer {access_token}",  "Accept": "application/json",}
        )
        if resp.status_code not in range(200, 205):
            if resp.status_code == 401:
                return RedirectResponse(url="/")
    return JSONResponse({"status": 200, "message": "success"})

# -- App --

@app.get("/dashboard", response_class=HTMLResponse)
async def read_index(request: Request):
    # MANUAL TRACE: Dashboard view
    with tracer.start_as_current_span("dashboard.view") as span:
        span.set_attribute("user.identifier", request.session.get("user", {}).get("identifier", "anonymous"))
        logger.info("Dashboard accessed")
        
        return templates.TemplateResponse("index.html", {
            "request": request,
        })

@app.get("/analyze", response_class=HTMLResponse)
async def analyze_page(request: Request):
    # MANUAL TRACE: Analyze page with database queries
    with tracer.start_as_current_span("analyze.page") as span:
        span.set_attribute("user.identifier", request.session.get("user", {}).get("identifier", "anonymous"))
        
        analyses = await db.get_all_analyses()
        span.set_attribute("analyses.count", len(analyses))
        
        documents = await db.get_all_documents()
        span.set_attribute("documents.count", len(documents))
        
        return templates.TemplateResponse("analyze.html", {
            "request": request,
            "analyses": analyses,
            "documents": documents
        })

@app.post("/analyze/upload", response_class=HTMLResponse)
async def upload_for_analysis(
    request: Request,
    background_tasks: BackgroundTasks,
    document: UploadFile = File(...),
    category_path: Optional[str] = Form(None)
):
    try:
        content = await document.read()
        filename = document.filename
        logger.info(f"Processing: {filename}")

        content_str = content.decode('utf-8', errors='ignore')

        if document.content_type == "application/pdf" or document.content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            d = DocumentStream(name=filename, stream=BytesIO(content))
            content_str = serialize(get_docling_doc(d))
        else:
            content_str = content.decode('utf-8', errors='ignore')

        analysis = await db.create_analysis(
            source_document=filename,
            original_content=content_str,
            status="processing",
            category_path=category_path
        )

        background_tasks.add_task(
            perform_automatic_analysis,
            document_analyser,
            content_str,
            filename,
            analysis["id"],
            category_path
        )

        analyses = await db.get_all_analyses()

        return templates.TemplateResponse("_analysis_table.html", {
            "request": request,
            "analyses": analyses
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@app.get("/analyze/results", response_class=HTMLResponse)
async def get_analysis_results(request: Request):
    analyses = await db.get_all_analyses()
    return templates.TemplateResponse("_analysis_table.html", {
        "request": request,
        "analyses": analyses
    })

@app.get("/analyze/status/{analysis_id}")
async def get_analysis_status(analysis_id: str):
    analysis = await db.get_analysis(analysis_id)

    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    progress = analysis.get('progress_details', {})

    return {
        "id": analysis_id,
        "status": analysis["status"],
        "progress": progress,
        "conflicts_count": len(analysis.get("conflicts", [])),
        "implementations_checked": len(analysis.get("internal_implementations", [])),
        "created_at": analysis["created_at"].isoformat(),
        "updated_at": analysis["updated_at"].isoformat()
    }

@app.get("/analyze/view/{analysis_id}", response_class=HTMLResponse)
async def view_analysis(request: Request, analysis_id: str):
    analysis = await db.get_analysis(analysis_id)
    analysis['internal_implementations'] = list(filter(
        lambda implementation: implementation.get('conflicts'),
        analysis['internal_implementations']
    ))

    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    from sopagent.markdown_renderer import markdown_to_html
    if 'original_content' in analysis:
        analysis['original_content'] = markdown_to_html(analysis['original_content'])

    analysis["timestamp"] = analysis["created_at"].isoformat()

    return templates.TemplateResponse("comparison.html", {
        "request": request,
        "analysis": analysis
    })

@app.get("/admin", response_class=HTMLResponse)
async def admin_page(request: Request, page: int = 1, page_size: int = 10):
    if page < 1:
        page = 1
    if page_size < 1 or page_size > 100:
        page_size = 10
    
    all_documents = await db.get_all_documents()
    total_documents = len(all_documents)
    
    offset = (page - 1) * page_size
    limit = page_size
    
    documents = await db.get_all_documents_paginated(offset, limit)
    
    total_pages = (total_documents + page_size - 1) // page_size
    has_prev = page > 1
    has_next = page < total_pages
    
    categories = await db.get_all_categories()
    return templates.TemplateResponse("admin.html", {
        "request": request,
        "documents": documents,
        "categories": categories,
        "pagination": {
            "current_page": page,
            "page_size": page_size,
            "total_documents": total_documents,
            "total_pages": total_pages,
            "has_prev": has_prev,
            "has_next": has_next,
            "prev_page": page - 1 if has_prev else None,
            "next_page": page + 1 if has_next else None
        }
    })


class CreateCategory(BaseModel):
    name: str
    parent_id: Optional[str] = None

@app.get("/admin/categories", response_class=JSONResponse)
async def get_categories():
    categories = await db.get_all_categories()
    return {"categories": categories}

@app.get("/admin/categories/{category_id}", response_class=JSONResponse)
async def get_category(category_id: str):
    category = await db.get_category(category_id)
    if not category:
        raise HTTPException(status_code=404, detail="Category not found")
    return {"category": category}

@app.post("/admin/categories", response_class=JSONResponse)
async def create_category(
    category: CreateCategory
):
    if category.parent_id:
        parent_cat = await db.get_category(category.parent_id)
        if not parent_cat:
            raise HTTPException(status_code=400, detail="Parent category not found")
        path = f"{parent_cat['path']}|{category.name}"
    else:
        path = category.name
    
    category = await db.create_category(category.name, category.parent_id, path)

    return {"category": category}

@app.put("/admin/categories/{category_id}", response_class=JSONResponse)
async def update_category(
    category_id: str,
    update_category: CreateCategory
):
    category = await db.get_category(category_id)
    if not category:
        raise HTTPException(status_code=404, detail="Category not found")

    new_path = None
    if update_category.name is not None and update_category.name != category['name']:
        if category['parent_id']:
            parent_cat = await db.get_category(category['parent_id'])
            if not parent_cat:
                raise HTTPException(status_code=400, detail="Parent category not found")
            new_path = f"{parent_cat['path']}|{update_category.name}"
        else:
            new_path = update_category.name
        
        old_path = category['path']
        if old_path:
            updated_category = await db.update_category_and_paths(category_id, update_category.name, new_path, old_path)
        
        return {"category": updated_category}

    return {"category": category}

@app.delete("/admin/categories/{category_id}", response_class=JSONResponse)
async def delete_category(category_id: str):
    """Delete a category"""
    # Check if category exists
    category = await db.get_category(category_id)
    if not category:
        raise HTTPException(status_code=404, detail="Category not found")
    
    # Delete the category
    success = await db.delete_category(category_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete category")
    
    return {"message": "Category deleted successfully"}

@app.post("/admin/upload", response_class=HTMLResponse)
async def admin_upload_document(
    request: Request,
    document: UploadFile = File(...),
    category_id: str = Form(...),
    serialize: Callable[[UploadFile], LiteralString] = Depends(lambda: serialize),
    db: Database = Depends(lambda: db),
    knowledge_bot_client: KnowledgeBotClient = Depends(lambda: knowledge_bot_client),
    templates: Jinja2Templates = Depends(lambda: templates)
):
    try:
        logger.info("Upload started")
        content = await document.read()
        filename = document.filename
        logger.info(f"Processing: {filename}")

        if document.content_type in (
            "application/pdf", 
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", 
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"):
            d = DocumentStream(name=filename, stream=BytesIO(content))
            docling_doc = get_docling_doc(d)
            chunks, images = await parse(docling_doc)
            content_str = serialize(docling_doc)
        else:
            content_str = content
        size = len(content)

        await knowledge_bot_client.upsert(
            document_id=filename,
            document_content=content_str,
            document_chunks=chunks if chunks else [content_str],
            metadata={'category_id': category_id, 'reference': filename},
            images=images
        )

        existing_docs = await db.get_all_documents()
        existing = next((doc for doc in existing_docs if doc["filename"] == filename), None)

        if existing:
            await db.update_document(
                document_id=existing["id"],
                filename=filename,
                content=content_str,
                category_id=category_id
            )
        else:
            await db.create_document(
                filename=filename,
                content=content_str,
                category_id=category_id,
                size=size
            )

        user = request.session.get("user", {}).get("identifier", "unknown")
        await audit_log.add(filename, "ADD", user)

        documents = await db.get_all_documents()
        categories = await db.get_all_categories()

        s3_upload_status = await s3.upload_file(filename, document.file)
        logger.info(s3_upload_status)

        return templates.TemplateResponse("_admin_documents.html", {
            "request": request,
            "documents": documents,
            "categories": categories
        })

    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail=f"Error uploading document: {str(e)}")

@app.delete("/admin/document/{document_id}")
async def delete_document(
    request: Request,
    document_id: str,
    db: Database = Depends(lambda: db),
    knowledge_bot_client: KnowledgeBotClient = Depends(lambda: knowledge_bot_client)
):
    try:
        document = await db.get_document(document_id)
        if document:
            await knowledge_bot_client.delete(document['filename'])
        success = await db.delete_document(document_id)
        if not success:
            raise HTTPException(status_code=404, detail="Document not found")

        user = request.session.get("user", {}).get("identifier", "unknown")
        await audit_log.add(document["filename"], "DELETE", user)
        
        s3_delete_status = await s3.delete_file(document['filename'])
        logger.info(s3_delete_status)

        return {"message": "Document deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")

@app.get("/admin/document/{filename}/audit")
async def get_audit(
    filename: str,
    db: Database = Depends(lambda: db)):
    logs = await db.get_all_actions_on_document(filename)
    serialized_logs = []
    if logs:
        serialized_logs = [
            {
                "user": log.get("performed_by"),
                "action": log.get("performed_action"),
                "timestamp": log.get("performed_at").strftime("%Y-%m-%d %H:%M:%S"),
            }
            for log in logs
        ]
    return JSONResponse(content={"logs": serialized_logs})

@app.get("/health")
async def health_check():
    # MANUAL TRACE: Health check
    with tracer.start_as_current_span("health.check") as span:
        tracing_enabled = is_tracing_enabled()
        span.set_attribute("tracing.enabled", tracing_enabled)
        
        return {
            "status": "healthy", 
            "timestamp": datetime.now().isoformat(),
            "tracing_enabled": tracing_enabled
        }

@app.get("/admin/documents/by-category-path/{category_path}", response_class=JSONResponse)
async def get_documents_by_category_path(category_path: str):
    documents = await db.get_documents_by_category_path(category_path)
    return documents

@app.get("/admin/documents/s3/list-files", response_class=JSONResponse)
async def list_all_s3_documents():
    files = await s3.list_s3_files()
    logger.info(files)
    return files
