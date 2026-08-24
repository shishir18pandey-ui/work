import os
import httpx
from pydantic import BaseModel, Field

import logging

logger = logging.getLogger(__name__)

# Jaeger Configuration (same as in query_tools.py)
# For local testing, use PROD endpoint
# In K8s, these will be overridden by environment variables
JAEGER_API_BASE = os.getenv("JAEGER_API_BASE", "https://tracing.uat-opt.idfcfirstbank.com/api")
JAEGER_AUTH_TOKEN = os.getenv("JAEGER_AUTH_TOKEN")


def _jaeger_auth_headers():
    if JAEGER_AUTH_TOKEN:
        return {"Authorization": f"Basic {JAEGER_AUTH_TOKEN}"}
    return {}

# Import service metadata
from new_flow.tools.service_metadata import (
    get_services,
    get_service_tags,
    get_service_purpose,
    get_app_config,
    get_elk_indexes,
)

# ELK Configuration
ELK_BASE_URL = os.getenv("ELK_BASE_URL", "https://DCELASDBSPRR07-prod.logging.devops.idfcbank.com:9200")
ELK_AUTH_HEADER = os.getenv("ELK_AUTH_HEADER")
CA_CERT_PATH = os.getenv("CA_CERT_FILE", "./IDFCBANKCA.pem")

ELK_HEADERS = {
    "Authorization": ELK_AUTH_HEADER,
    "Content-Type": "application/json"
}


class GetServicesInput(BaseModel):
    app: str = Field(description="Application name (e.g., optimus, cbs, idp)")


class GetServiceTagsInput(BaseModel):
    app: str = Field(description="Application name (e.g., optimus, cbs, idp)")
    service: str = Field(description="Service name (e.g., upi-api, idp-api)")


class GetELKFieldsInput(BaseModel):
    app: str = Field(description="Application name (e.g., optimus, cbs, idp)")
    index_type: str = Field(
        default="log",
        description="Index type: 'log' for application logs, 'trace' for Jaeger spans"
    )


class GetAppConfigInput(BaseModel):
    app: str = Field(description="Application name (e.g., optimus, cbs, idp)")


def get_services_impl(app: str) -> str:
    app = app.strip().lower()
    
    services = get_services(app)
    
    if not services:
        # Try to get from config if available
        config = get_app_config(app)
        if config:
            services = list(config.get("services", {}).keys())
    
    if not services:
        return (
            f"No predefined services found for app '{app}'. "
            f"You can try querying any service directly - discovery is flexible. "
            f"Available apps in metadata: optimus, cbs, idp"
        )
    
    purposes = []
    for svc in services:
        purpose = get_service_purpose(app, svc)
        if purpose:
            purposes.append(f"  - {svc}: {purpose}")
        else:
            purposes.append(f"  - {svc}")
    
    output = [
        f"Found {len(services)} services for app '{app}':",
        "",
    ]
    output.extend(purposes)
    
    output.extend([
        "",
        "To get tags for a specific service, use GetServiceTagsTool.",
        "To search traces, use fetch_jaeger_traces with any service name.",
    ])
    
    return "\n".join(output)


def get_service_tags_impl(app: str, service: str) -> str:
    app = app.strip().lower()
    service = service.strip().lower()
    
    tags = get_service_tags(app, service)
    
    if tags is None:
        # No metadata - suggest common tags
        return (
            f"No predefined tags found for {app}/{service}. "
            f"You can try any of these common tags: "
            f"customer_id, txn_id, mobile_number, ucic, user_id, session_tracking_id, "
            f"username, device_id, account_number, transaction_id "
            f"The query will proceed with your chosen tag."
        )
    
    if not tags:
        return f"No tags configured for {app}/{service}. Try any common tag."
    
    return (
        f"Valid tags for {app}/{service}: {tags}\n\n"
        f"Use one of these tags with fetch_jaeger_traces tool. "
        f"If you use a different tag, a warning will be shown but the query will proceed."
    )


async def get_elk_fields_impl(app: str, index_type: str = "log") -> str:
    app = app.strip().lower()
    index_type = index_type.strip().lower()
    
    # Get index pattern based on app and type
    elk_indexes = get_elk_indexes(app)
    
    if index_type == "trace":
        index_pattern = elk_indexes.get("elk_trace_index", "prod-jaeger-span-*")
    else:
        index_pattern = elk_indexes.get("elk_log_index", "elk-*")
    
    # If no config, use defaults
    if not index_pattern or index_pattern == "elk-*":
        if app == "optimus":
            index_pattern = "prod-jaeger-span-*" if index_type == "trace" else "elk-opt-*"
        elif app == "cbs":
            index_pattern = "prod-jaeger-span-cbs-*" if index_type == "trace" else "elk-cbs-prod*"
        elif app == "idp":
            index_pattern = "prod-jaeger-span-idp-*" if index_type == "trace" else "elk-idp-prod*"
        else:
            index_pattern = "prod-jaeger-span-*" if index_type == "trace" else "elk-*-prod*"
    
    try:
        # Fetch mapping to get all fields
        mapping_url = f"{ELK_BASE_URL}/{index_pattern}/_mapping"
        
        async with httpx.AsyncClient(timeout=30.0, verify=CA_CERT_PATH) as client:
            response = await client.get(mapping_url, headers=ELK_HEADERS)
            
            if response.status_code != 200:
                return f"Error fetching ELK mapping: HTTP {response.status_code}"
            
            mapping = response.json()
            
            # Extract fields from mapping
            fields = set()
            
            # Parse the mapping response
            for index_name, index_data in mapping.items():
                properties = index_data.get("mappings", {}).get("properties", {})
                
                def extract_fields(props, prefix=""):
                    for field_name, field_def in props.items():
                        full_name = f"{prefix}{field_name}" if prefix else field_name
                        fields.add(full_name)
                        # Handle nested objects
                        if "properties" in field_def:
                            extract_fields(field_def["properties"], f"{full_name}.")
                
                extract_fields(properties)
            
            if not fields:
                return f"No fields found in index pattern {index_pattern}"
            
            # Group fields by common prefixes
            field_list = sorted(fields)
            
            output = [
                f"Found {len(field_list)} fields in index pattern '{index_pattern}' for app '{app}':",
                "",
            ]
            
            # Show common fields first
            important_fields = ["customer_id", "txn_id", "ucic", "mobile_number", "user_id", 
                              "account_number", "message", "logtype", "@timestamp", "serviceName",
                              "traceId", "spanId", "error", "http.status_code"]
            
            output.append("Common fields:")
            for field in important_fields:
                if field in fields:
                    output.append(f"  - {field}")
            
            output.append("")
            output.append(f"All fields ({len(field_list)} total):")
            # Show first 50 fields
            for field in field_list[:50]:
                output.append(f"  - {field}")
            
            if len(field_list) > 50:
                output.append(f"  ... and {len(field_list) - 50} more fields")
            
            return "\n".join(output)
    
    except Exception as e:
        return f"Error fetching ELK fields: {str(e)}\n\nYou can still query with any field name - this is just for discovery."


def get_app_config_impl(app: str) -> str:
    app = app.strip().lower()
    
    config = get_app_config(app)
    
    if not config:
        return (
            f"No configuration found for app '{app}'. "
            f"Available apps: optimus, cbs, idp. "
            f"Using default endpoints."
        )
    
    output = [
        f"Configuration for app '{app}':",
        "",
    ]
    
    if config.get("jaeger_endpoint"):
        output.append(f"Jaeger Endpoint: {config.get('jaeger_endpoint')}")
    
    if config.get("elk_trace_index"):
        output.append(f"ELK Trace Index: {config.get('elk_trace_index')}")
    
    if config.get("elk_log_index"):
        output.append(f"ELK Log Index: {config.get('elk_log_index')}")
    
    services = config.get("services", {})
    if services:
        output.append(f"Services configured: {len(services)}")
    
    return "\n".join(output)


async def discover_jaeger_services_impl(app: str = None) -> str:
    import time as _time
    
    # Get Jaeger endpoint from app config or use default
    jaeger_endpoint = None
    if app:
        config = get_app_config(app)
        if config:
            jaeger_endpoint = config.get("jaeger_endpoint")
    
    if not jaeger_endpoint:
        jaeger_endpoint = JAEGER_API_BASE

    headers = _jaeger_auth_headers()

    params = {"lookback": "1h"}

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{jaeger_endpoint}/services", 
                params=params,
                headers=headers
            )
            
            if response.status_code != 200:
                return f"Error fetching services from Jaeger: HTTP {response.status_code}"
            
            data = response.json()
            services = data.get("data", [])
            
            if not services:
                return f"No services found in Jaeger. Try increasing time range."
            
            output = [
                f"Discovered {len(services)} services from Jaeger:",
                "",
            ]
            
            # Show all services
            for svc in sorted(services):
                output.append(f"  - {svc}")
            
            output.extend([
                "",
                "Note: These are ALL services that have sent traces in the last 1 hour.",
                "Use any service name directly with fetch_jaeger_traces tool.",
            ])
            
            return "\n".join(output)
    
    except Exception as e:
        return f"Error connecting to Jaeger: {str(e)}\n\nYou can still query any service directly."


async def discover_jaeger_tags_impl(service: str, app: str = None) -> str:
    jaeger_endpoint = None
    if app:
        config = get_app_config(app)
        if config:
            jaeger_endpoint = config.get("jaeger_endpoint")
    
    if not jaeger_endpoint:
        jaeger_endpoint = JAEGER_API_BASE

    headers = _jaeger_auth_headers()
    
    # Use a reasonable lookback window (1 hour)
    params = {"service": service, "lookback": "1h"}
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{jaeger_endpoint}/tags", 
                params=params,
                headers=headers
            )
            
            if response.status_code != 200:
                return f"Error fetching tags from Jaeger: HTTP {response.status_code}"
            
            data = response.json()
            tag_data = data.get("data", [])
            
            if not tag_data:
                return (
                    f"No tags found for service '{service}'. "
                    f"Try common tags: customer_id, txn_id, mobile_number, ucic, "
                    f"user_id, session_tracking_id, username, device_id, account_number"
                )
            
            # Extract tag keys from the response
            # Jaeger returns tags as a list of {key: "tagName", values: [...]} objects
            tags = []
            for item in tag_data:
                if isinstance(item, dict):
                    key = item.get("key")
                    if key:
                        tags.append(key)
                elif isinstance(item, str):
                    tags.append(item)
            
            if not tags:
                return (
                    f"No tags found for service '{service}'. "
                    f"Try common tags: customer_id, txn_id, mobile_number, ucic, "
                    f"user_id, session_tracking_id, username, device_id, account_number"
                )
            
            # Sort tags - put common ones first
            common_tags = ["customer_id", "txn_id", "mobile_number", "ucic", "user_id", 
                          "session_tracking_id", "username", "device_id", "account_number", 
                          "transaction_id", "error", "http.status_code"]
            
            sorted_tags = []
            remaining_tags = []
            
            for tag in tags:
                if tag in common_tags:
                    sorted_tags.append(tag)
                else:
                    remaining_tags.append(tag)
            
            # Add common tags first, then the rest
            final_tags = sorted_tags + sorted(remaining_tags)
            
            output = [
                f"Discovered {len(final_tags)} tags for service '{service}':",
                "",
            ]
            
            # Show common tags first
            if sorted_tags:
                output.append("Common tags:")
                for tag in sorted_tags:
                    output.append(f"  - {tag}")
                output.append("")
            
            # Show all tags (limit to 50)
            if remaining_tags:
                output.append(f"All tags ({len(final_tags)} total):")
                for tag in final_tags[:50]:
                    output.append(f"  - {tag}")
                if len(final_tags) > 50:
                    output.append(f"  ... and {len(final_tags) - 50} more tags")
            
            output.extend([
                "",
                "Note: These are ALL tags found in traces for this service in the last 1 hour.",
                "Use any of these tags with fetch_jaeger_traces tool.",
            ])
            
            return "\n".join(output)
    
    except Exception as e:
        return (
            f"Error connecting to Jaeger: {str(e)}\n\n"
            f"Try common tags: customer_id, txn_id, mobile_number, ucic, "
            f"user_id, session_tracking_id, username, device_id, account_number"
        )


try:
    from crewai.tools import BaseTool
    
    class GetServicesTool(BaseTool):
        name: str = "discover_services"
        description: str = (
            "List all available services for an application. "
            "Use this before querying Jaeger traces to discover valid service names. "
            "Returns service names and their purposes (if available)."
        )
        args_schema: type = GetServicesInput
        
        def _run(self, app: str) -> str:
            return get_services_impl(app)
    
    class GetServiceTagsTool(BaseTool):
        name: str = "discover_service_tags"
        description: str = (
            "Get valid query tags for a specific service. "
            "Use this before querying Jaeger to know which tag names are valid. "
            "Returns list of tag names like customer_id, txn_id, mobile_number, etc."
        )
        args_schema: type = GetServiceTagsInput
        
        def _run(self, app: str, service: str) -> str:
            return get_service_tags_impl(app, service)
    
    class GetELKFieldsTool(BaseTool):
        name: str = "discover_elk_fields"
        description: str = (
            "Discover all available fields in ELK index by fetching mapping. "
            "Use this to understand what fields can be queried. "
            "Returns common fields first, then full list."
        )
        args_schema: type = GetELKFieldsInput
        
        def _run(self, app: str, index_type: str = "log") -> str:
            import asyncio
            return asyncio.run(get_elk_fields_impl(app, index_type))

    class GetAppConfigTool(BaseTool):
        name: str = "get_app_config"
        description: str = (
            "Get configuration for an application including Jaeger endpoint, "
            "ELK index patterns, and service count."
        )
        args_schema: type = GetAppConfigInput

        def _run(self, app: str) -> str:
            return get_app_config_impl(app)

    class DiscoverJaegerServicesTool(BaseTool):
        name: str = "discover_jaeger_services"
        description: str = (
            "Dynamically discover all available services from Jaeger. "
            "No preconfiguration needed - queries Jaeger API directly. "
            "Returns list of all services with recent traces."
        )
        
        def _run(self, app: str = None) -> str:
            import asyncio
            return asyncio.run(discover_jaeger_services_impl(app))

    class DiscoverJaegerTagsTool(BaseTool):
        name: str = "discover_jaeger_tags"
        description: str = (
            "Dynamically discover all available tags for a specific service from Jaeger. "
            "Queries Jaeger's /tags endpoint to get all tags used in recent traces. "
            "Returns list of tags like customer_id, txn_id, mobile_number, etc."
        )
        
        def _run(self, service: str, app: str = None) -> str:
            import asyncio
            return asyncio.run(discover_jaeger_tags_impl(service, app))

    __all__ = [
        "GetServicesTool",
        "GetServiceTagsTool", 
        "GetELKFieldsTool",
        "GetAppConfigTool",
        "DiscoverJaegerServicesTool",
        "DiscoverJaegerTagsTool",
        "get_services_impl",
        "get_service_tags_impl",
        "get_elk_fields_impl",
        "get_app_config_impl",
        "discover_jaeger_services_impl",
        "discover_jaeger_tags_impl",
    ]

except ImportError:
    # CrewAI not available - provide standalone functions
    logger.warning("crewai not installed - discovery tools will be functions only")
    
    __all__ = [
        "get_services_impl",
        "get_service_tags_impl",
        "get_elk_fields_impl",
        "get_app_config_impl",
    ]
