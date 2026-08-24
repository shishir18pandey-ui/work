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










this is sericedata.yaml
# Service Metadata Configuration
# This provides optional predefined tags for services (if available)
# Discovery will work even without this - this is just for validation/guidance
# Zero dependency on support teams for discovery - all discovered via APIs

optimus:
  jaeger_endpoint: "https://tracing.uat-opt.idfcfirstbank.com/api"
  jaeger_prod_endpoint: "https://tracing.uat-opt.idfcfirstbank.com/api"
  elk_trace_index: "prod-jaeger-span-*"
  elk_log_index: "elk-opt-*"
  
  services:
    idp-api:
      tags: 
        - j_id
        - device_id
        - session_tracking_id
        - mobile_number
        - username
        - user_id
        - customer_id
      purpose: "Login/authentication/registration"
    PAYMENTS-API:
      tags:
        - txn_id
        - txn_request_id
        - customer_id
      purpose: "Fund transfer problems"
    bulk-payments-api:
      tags:
        - bulk_ref_id
        - customer_id
      purpose: "Bulk transfer problems"
    IPO-API:
      tags:
        - customer_id
      purpose: "IPO-related issues"
    deposits-api:
      tags:
        - customer_id
        - txn_id
      purpose: "Document download, deposit, FD issues"
    CREDIT-CARD-API:
      tags:
        - customer_id
        - user_id
      purpose: "Credit card operations"
    ecom-api:
      tags:
        - customer_id
        - txn_id
        - txn_request_id
      purpose: "E-commerce transactions"
    WEALTH-API:
      tags:
        - customer_id
        - user_id
        - session_tracking_id
      purpose: "Mutual fund portfolio"
    DEBITCARD-API:
      tags:
        - customer_id
        - user_id
      purpose: "Debit card operations"
    upi-api:
      tags:
        - customer_id
        - txn_id
        - mobile_number
      purpose: "UPI transactions"
    FX-API:
      tags:
        - customer_id
        - txn_id
      purpose: "Pay Abroad transactions"
    kyc-service:
      tags:
        - customer_id
        - user_id
      purpose: "KYC operations"
    beneficiary-api:
      tags:
        - customer_id
        - beneficiary_id
      purpose: "Beneficiary management"
    EMANDATE:
      tags:
        - customer_id
        - mandate_id
      purpose: "Aadhaar mandate operations"
    CX-API:
      tags:
        - mobile_number
        - customer_id
        - user_id
      purpose: "Mobile/email validation"
    INSURANCE-API:
      tags:
        - customer_id
        - policy_id
      purpose: "Insurance operations"
    LAS-API:
      tags:
        - customer_id
        - lai_id
      purpose: "LAS operations"
    REMITTANCE-API:
      tags:
        - customer_id
        - remittance_id
      purpose: "Remittance operations"
    CAS-API:
      tags:
        - customer_id
        - mf_account_id
      purpose: "Mutual fund linking"
    E-CHEQUE-API:
      tags:
        - customer_id
        - cheque_id
      purpose: "Cheque operations"
    LOANS-API:
      tags:
        - customer_id
        - loan_id
      purpose: "Loan operations"
    billpay-api:
      tags:
        - customer_id
        - user_id
        - session_tracking_id
        - txn_id
        - mobile_number
      purpose: "Bill payments"
    APPLY-FM-API:
      tags:
        - applicationId
        - ucic
        - mobile_number
        - customer_id
        - user_id
      purpose: "Loan applications"

cbs:
  jaeger_endpoint: "https://tracing.uat-cbs.idfcfirstbank.com/api"
  jaeger_prod_endpoint: "https://tracing.prod-cbs.idfcfirstbank.com/api"
  elk_trace_index: "prod-jaeger-span-cbs-*"
  elk_log_index: "elk-cbs-prod*"
  
  services:
    cbs-backend:
      tags:
        - account_number
        - customer_id
        - txn_id
        - ucic
      purpose: "Core banking backend"
    cbs-api:
      tags:
        - account_number
        - customer_id
        - txn_id
      purpose: "Core banking API"
    cbs-core:
      tags:
        - account_number
        - customer_id
        - txn_id
      purpose: "Core banking core processing"

idp:
  jaeger_endpoint: "https://tracing.uat-idp.idfcfirstbank.com/api"
  elk_trace_index: "prod-jaeger-span-idp-*"
  elk_log_index: "elk-idp-prod*"
  
  services:
    idp-auth:
      tags:
        - j_id
        - user_id
        - session_tracking_id
        - mobile_number
      purpose: "Authentication service"
    idp-oauth:
      tags:
        - user_id
        - access_token
        - client_id
      purpose: "OAuth operations"
    idp-mfa:
      tags:
        - user_id
        - mobile_number
        - otp
      purpose: "MFA operations"
