import os
import logging
from typing import Dict, List, Optional, Any

import yaml

logger = logging.getLogger(__name__)

# Load YAML at module import
_metadata: Dict[str, Any] = {}
_metadata_loaded = False


def _load_metadata() -> Dict[str, Any]:
    """Load service metadata from YAML file."""
    global _metadata, _metadata_loaded
    
    if _metadata_loaded:
        return _metadata
    
    # Find the YAML file - check multiple locations
    possible_paths = [
        os.path.join(os.path.dirname(__file__), "service_metadata.yaml"),
        os.path.join(os.path.dirname(__file__), "..", "tools", "service_metadata.yaml"),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "tools", "service_metadata.yaml"),
    ]
    
    yaml_path = None
    for path in possible_paths:
        if os.path.exists(path):
            yaml_path = path
            break
    
    if not yaml_path:
        logger.warning("service_metadata.yaml not found - running without predefined tags")
        _metadata_loaded = True
        return {}
    
    try:
        with open(yaml_path, 'r') as f:
            _metadata = yaml.safe_load(f) or {}
        logger.info(f"Loaded service metadata for apps: {list(_metadata.keys())}")
    except Exception as e:
        logger.warning(f"Failed to load service_metadata.yaml: {e}")
        _metadata = {}
    
    _metadata_loaded = True
    return _metadata


def get_app_config(app: str) -> Optional[Dict[str, Any]]:
    metadata = _load_metadata()
    app_key = app.lower().strip()
    return metadata.get(app_key)


def get_services(app: str) -> List[str]:
    config = get_app_config(app)
    if not config:
        return []
    services = config.get("services", {})
    return list(services.keys())


def get_service_tags(app: str, service: str) -> Optional[List[str]]:
    config = get_app_config(app)
    if not config:
        return None
    
    services = config.get("services", {})
    service_key = service.lower().strip()
    
    if service_key not in services:
        return None
    
    return services[service_key].get("tags")


def get_service_purpose(app: str, service: str) -> Optional[str]:
    config = get_app_config(app)
    if not config:
        return None
    
    services = config.get("services", {})
    service_key = service.lower().strip()
    
    if service_key not in services:
        return None
    
    return services[service_key].get("purpose")


def is_tag_valid(app: str, service: str, tag: str) -> bool:
    """
    Check if a tag is valid for a service.
    
    Args:
        app: Application name
        service: Service name
        tag: Tag name to validate
    
    Returns:
        True if tag is valid, or if we don't have metadata (allow all)
        False only if we have metadata AND tag is not in the list
    """
    valid_tags = get_service_tags(app, service)
    
    # If we don't have metadata, allow all tags (discovery-first approach)
    if valid_tags is None:
        return True
    
    return tag.lower().strip() in [t.lower() for t in valid_tags]


# Import jaeger_endpoint from app_config (single source of truth)
# This removes YAML dependency for Jaeger endpoints
from new_flow.tools.app_config import get_jaeger_endpoint as _get_jaeger_endpoint_from_app_config


def get_jaeger_endpoint(app: str) -> Optional[str]:
    """
    Get Jaeger endpoint for an application.
    
    Args:
        app: Application name
    
    Returns:
        Jaeger API endpoint URL or None
    """
    return _get_jaeger_endpoint_from_app_config(app)


def get_elk_indexes(app: str) -> Dict[str, str]:
    """
    Get ELK index patterns for an application.
    
    Args:
        app: Application name
    
    Returns:
        Dict with keys: elk_trace_index, elk_log_index
    """
    config = get_app_config(app)
    if not config:
        return {}
    return {
        "elk_trace_index": config.get("elk_trace_index"),
        "elk_log_index": config.get("elk_log_index"),
    }


def validate_tag_with_warning(app: str, service: str, tag: str) -> str:
    valid_tags = get_service_tags(app, service)
    
    if valid_tags is None:
        # No metadata - allow with no warning
        return ""
    
    if tag.lower() not in [t.lower() for t in valid_tags]:
        return (
            f"⚠️ WARNING: '{tag}' is NOT in known tags for {app}/{service}. "
            f"Known tags: {valid_tags}. "
            f"Proceeding anyway with your requested tag..."
        )
    
    return ""


# Load metadata on module import
_load_metadata()
