import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import pathlib
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from starlette.middleware.sessions import SessionMiddleware
from dotenv import load_dotenv
import httpx
import pkce
import urllib.parse
import json
import asyncio
from app.agent.llm import llm_config
from opentelemetry.instrumentation.asgi import OpenTelemetryMiddleware

from httpx import AsyncClient


from app.observability import is_tracing_enabled, init_tracer, attach_opentelemetry

# Initialize tracer at module level (before app creation)
if is_tracing_enabled():
    init_tracer()
    print(" OpenTelemetry tracer initialized")


from app.agent.llm import llm_config
# set llm config
asyncio.get_event_loop().create_task(llm_config.refresh_loop())

os.environ["HOME"] = "/tmp"

# Ensure the directory exists
pathlib.Path("/tmp/.local/share").mkdir(parents=True, exist_ok=True)

# Import routers (after tracer initialization)
from app.routers import applications, integrations, procedures, query

print(f"CLIENT_ID: {os.environ.get('ENT_AUTH_CLIENT_ID', 'NOT FOUND')}")
print(f"AUTH_URL: {os.environ.get('ENT_AUTH_URL', 'NOT FOUND')}")

# OAuth config from environment variables
CLIENT_ID = os.environ['ENT_AUTH_CLIENT_ID']
AUTHORIZE_URL = os.environ['ENT_AUTH_URL']
TOKEN_URL = os.environ['ENT_AUTH_TOKEN_URL']
USER_INFO_URL = os.environ['ENT_AUTH_USER_INFO_URL']
REDIRECT_URI = os.environ.get('ENT_AUTH_REDIRECT_URL', 'http://localhost:8000/auth/oauth/ent_auth/callback')
VALIDATION_URL = os.environ.get('ENT_AUTH_VALIDATION_URL')
EXTEND_URL = os.environ.get('ENT_AUTH_KEEPALIVE_URL')

# Verifier store for OAuth
verifier_store = {}


from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
  
    # Start LLM config refresh loop
    llm_refresh_task = asyncio.create_task(llm_config.refresh_loop())
    
    print(" Application startup complete")
    
    yield  # App is running
    
    # Cleanup on shutdown
    llm_refresh_task.cancel()
    try:
        await llm_refresh_task
    except asyncio.CancelledError:
        pass
    
    print(" Application shutdown complete")

app = FastAPI(
    title="Semantic Agent",
    lifespan=lifespan
)

app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv('SESSION_SECRET', 'your-secret-key-change-in-production')
)

if is_tracing_enabled():
    attach_opentelemetry(app)
    print(" OpenTelemetry attached to FastAPI")


# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="app/templates")

@app.get("/login")
async def login():
    """Initiate OAuth flow"""
    
   
    from opentelemetry import trace
    root_span = trace.get_current_span()
    if root_span and root_span.is_recording():
        root_span.update_name("GET /login [OAuth Init]")
    
    code_verifier, code_challenge = pkce.generate_pkce_pair()
    session_id = code_verifier[:8]
    
    verifier_store[session_id] = code_verifier
    
    params = {
        "client_id": CLIENT_ID,
        "response_type": "code",
        "redirect_uri": REDIRECT_URI,
        "scope": "offline_access openid",
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "state": session_id
    }
    
    url = f"{AUTHORIZE_URL}?{urllib.parse.urlencode(params)}"
    print(f"Redirecting to: {url}")
    
    return RedirectResponse(url)



@app.get("/auth/oauth/ent_auth/callback")
async def callback(request: Request):
    """OAuth callback handler"""
    
    # ============== ENHANCE ROOT SPAN ==============
    from opentelemetry import trace
    root_span = trace.get_current_span()
    if root_span and root_span.is_recording():
        root_span.update_name("GET /auth/callback [OAuth]")
    # ===============================================
    
    code = request.query_params.get("code")
    state = request.query_params.get("state")  # session_id
    
    if not code or not state:
        print("[ERROR] Missing code or state")
        return RedirectResponse(url="/login")
    
    code_verifier = verifier_store.pop(state, None)
    if not code_verifier:
        print("[ERROR] Invalid session state")
        return RedirectResponse(url="/login")
    
    # Exchange code for token
    async with httpx.AsyncClient(verify=False) as client:
        payload = {
            "code": code,
            "redirect_uri": REDIRECT_URI,
            "client_id": CLIENT_ID,
            "code_verifier": code_verifier
        }
        
        print(f"Token request payload: {payload}")
        
        token_response = await client.post(
            url=TOKEN_URL,
            data=json.dumps(payload),
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        
        print(f"Token response: {token_response.status_code} - {token_response.text}")
        
        if token_response.status_code != 200:
            raise HTTPException(status_code=token_response.status_code, detail=token_response.text)
        
        access_token = token_response.json().get("access_token")
    
    # Get user info
    async with httpx.AsyncClient(verify=False) as client:
        user_info_response = await client.post(
            url=USER_INFO_URL,
            headers={
                "Authorization": f"Bearer {access_token}",
                "Accept": "application/json"
            }
        )
        
        print(f"User info response: {user_info_response.status_code} - {user_info_response.text}")
        
        if user_info_response.status_code != 200:
            raise HTTPException(status_code=user_info_response.status_code, detail=user_info_response.text)
        
        user_info = user_info_response.json()
    
    user = {
        "identifier": user_info.get("email"),
        "display_name": f"{user_info.get('firstName', '')} {user_info.get('lastName', '')}".strip(),
        "access_token": access_token
    }
    
    # ============== ADD USER TO ROOT SPAN ==============
    if root_span and root_span.is_recording():
        root_span.set_attribute("user.identifier", user["identifier"])
        root_span.set_attribute("auth.success", True)
    # ===================================================
    
    # Store user in session
    request.session["user"] = user
    
    # Redirect to home page after successful login
    return RedirectResponse(url="/")



@app.get("/logout")
async def logout(request: Request):
    """Clear session and logout"""
    request.session.clear()
    return RedirectResponse(url="/login")


@app.post("/keepalive")
async def keepalive(request: Request):
    """Keep session alive"""
    user = request.session.get("user", {})
    access_token = user.get("access_token")
    
    async with AsyncClient(verify=False) as client:
        resp = await client.post(
            url=EXTEND_URL,
            headers={
                "Authorization": f"Bearer {access_token}",
                "Accept": "application/json"
            }
        )
        if resp.status_code not in range(200, 205):
            if resp.status_code == 401:
                return RedirectResponse(url="/")
    
    return JSONResponse({"status": 200, "message": "success"})

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Home page - check if user is logged in"""
    user = request.session.get("user")
    print(user)
    
    if not user:
        return RedirectResponse(url="/login")
    
    return templates.TemplateResponse("index.html", {"request": request, "user": user})


@app.get("/query", response_class=HTMLResponse)
async def query_page(request: Request):
    """Query page - check if user is logged in"""
    user = request.session.get("user")
    
    # If not logged in, redirect to login
    if not user:
        return RedirectResponse(url="/login")
    
    return templates.TemplateResponse("query.html", {"request": request, "user": user})

app.include_router(applications.router)
app.include_router(integrations.router)
app.include_router(procedures.router)
app.include_router(query.router)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    from datetime import datetime
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "tracing_enabled": is_tracing_enabled()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
