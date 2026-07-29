import os
import re
import logging
from typing import Dict, Any

os.environ["OTEL_SDK_DISABLED"] = "true"

logger = logging.getLogger(__name__)

# Keywords that are always blocked
BLOCKED_KEYWORDS = [
    "DROP",
    "TRUNCATE",
    "GRANT",
    "REVOKE",
    "ALTER",  # Block ALTER for safety (can add more granular control later)
]

# DELETE is allowed only with WHERE clause
DELETE_PATTERN = re.compile(r'\bDELETE\s+FROM\b', re.IGNORECASE)
WHERE_PATTERN = re.compile(r'\bWHERE\b', re.IGNORECASE)

# Patterns that indicate dangerous operations
DANGEROUS_PATTERNS = [
    (r'\bDROP\s+TABLE\b', "DROP TABLE is not allowed"),
    (r'\bDROP\s+DATABASE\b', "DROP DATABASE is not allowed"),
    (r'\bDROP\s+INDEX\b', "DROP INDEX is not allowed"),
    (r'\bTRUNCATE\s+TABLE\b', "TRUNCATE TABLE is not allowed"),
    (r'\bTRUNCATE\b', "TRUNCATE is not allowed"),
    (r'\bGRANT\s+\w+\s+TO\b', "GRANT is not allowed"),
    (r'\bREVOKE\s+\w+\s+FROM\b', "REVOKE is not allowed"),
    (r'\bDELETE\s+FROM\s+\w+\s*;?\s*$', "DELETE without WHERE clause is not allowed"),
    (r'\bDROP\s+USER\b', "DROP USER is not allowed"),
    (r'\bDROP\s+VIEW\b', "DROP VIEW is not allowed"),
    (r'\bDROP\s+SEQUENCE\b', "DROP SEQUENCE is not allowed"),
    (r'\bDROP\s+PROCEDURE\b', "DROP PROCEDURE is not allowed"),
    (r'\bDROP\s+FUNCTION\b', "DROP FUNCTION is not allowed"),
    (r'\bDROP\s+TRIGGER\b', "DROP TRIGGER is not allowed"),
    (r'\bSHUTDOWN\b', "SHUTDOWN is not allowed"),
    (r'\bKILL\b', "KILL is not allowed"),
]


def validate_sql_query(query: str) -> Dict[str, Any]:
    """
    Validate a SQL query for safety.
    
    Args:
        query: The SQL query to validate
        
    Returns:
        Dict with keys:
            - is_safe: bool - True if query is safe to execute
            - reason: str - Explanation if blocked, empty if safe
    """
    if not query or not query.strip():
        return {
            "is_safe": False,
            "reason": "Empty query provided"
        }
    
    # Normalize query - remove extra whitespace and convert to uppercase for checking
    query_normalized = ' '.join(query.split()).upper()
    
    # Check for blocked keywords
    for keyword in BLOCKED_KEYWORDS:
        if keyword in query_normalized:
            # Special handling for DELETE - allow if WHERE is present
            if keyword == "DELETE":
                # Check if DELETE has WHERE
                if WHERE_PATTERN.search(query):
                    continue  # Allow DELETE with WHERE
                else:
                    return {
                        "is_safe": False,
                        "reason": "DELETE without WHERE clause is not allowed for safety"
                    }
            
            return {
                "is_safe": False,
                "reason": f"Query contains blocked keyword: {keyword}"
            }
    
    # Check for dangerous patterns
    for pattern, reason in DANGEROUS_PATTERNS:
        if re.search(pattern, query, re.IGNORECASE):
            return {
                "is_safe": False,
                "reason": reason
            }
    
    # Additional check: Ensure DELETE has WHERE
    if DELETE_PATTERN.search(query):
        if not WHERE_PATTERN.search(query):
            return {
                "is_safe": False,
                "reason": "DELETE queries must include a WHERE clause"
            }
    
    # Query passed all checks
    return {
        "is_safe": True,
        "reason": ""
    }


def check_dangerous_keywords(query: str) -> bool:
    """
    Check if query contains dangerous keywords.
    
    Args:
        query: The SQL query to check
        
    Returns:
        True if dangerous keywords found, False otherwise
    """
    if not query:
        return False
    
    query_upper = query.upper()
    
    for keyword in BLOCKED_KEYWORDS:
        if keyword in query_upper:
            # Allow DELETE if WHERE is present
            if keyword == "DELETE":
                if WHERE_PATTERN.search(query):
                    continue
            return True
    
    return False


def get_query_risk_level(query: str) -> str:
    """
    Get the risk level of a query.
    
    Args:
        query: The SQL query to check
        
    Returns:
        One of: "safe", "low", "medium", "high", "blocked"
    """
    validation = validate_sql_query(query)
    
    if not validation["is_safe"]:
        return "blocked"
    
    # Check for potentially risky but allowed operations
    query_upper = query.upper()
    
    # UPDATE without WHERE is risky but not blocked
    if "UPDATE" in query_upper and not WHERE_PATTERN.search(query):
        return "medium"
    
    # INSERT is low risk
    if "INSERT" in query_upper:
        return "low"
    
    # SELECT is generally safe
    if query_upper.strip().startswith("SELECT"):
        return "safe"
    
    return "low"


# ─────────────────────────────────────────────────────────────────────────────
# Integration helper
# ─────────────────────────────────────────────────────────────────────────────

async def validate_and_execute(
    query: str,
    execute_func,
    *args,
    **kwargs
):
    """
    Validate a query before execution.
    
    Args:
        query: The SQL query to validate
        execute_func: Async function to execute the query if valid
        *args, **kwargs: Arguments to pass to execute_func
        
    Returns:
        Result from execute_func if valid, error message if blocked
    """
    validation = validate_sql_query(query)
    
    if not validation["is_safe"]:
        logger.warning(f"[QueryValidator] Blocked: {validation['reason']}")
        return {
            "success": False,
            "error": validation["reason"],
            "blocked": True
        }
    
    # Execute the query
    return await execute_func(*args, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Test functions
# ─────────────────────────────────────────────────────────────────────────────

def test_validator():
    """Test the query validator with various queries."""
    test_queries = [
        # Should be blocked
        ("DROP TABLE users", False, "DROP TABLE"),
        ("TRUNCATE TABLE accounts", False, "TRUNCATE"),
        ("DELETE FROM users", False, "DELETE without WHERE"),
        ("GRANT SELECT ON users TO admin", False, "GRANT"),
        ("REVOKE INSERT FROM users", False, "REVOKE"),
        
        # Should be allowed
        ("SELECT * FROM users WHERE id = 1", True, "SELECT with WHERE"),
        ("SELECT account_no, balance FROM gldm WHERE ucic = '123'", True, "SELECT"),
        ("INSERT INTO logs (msg) VALUES ('test')", True, "INSERT"),
        ("UPDATE users SET name = 'test' WHERE id = 1", True, "UPDATE with WHERE"),
        ("DELETE FROM users WHERE id = 1", True, "DELETE with WHERE"),
    ]
    
    print("Testing Query Validator:")
    print("-" * 60)
    
    for query, expected_safe, description in test_queries:
        result = validate_sql_query(query)
        status = "✓" if result["is_safe"] == expected_safe else "✗"
        print(f"{status} {description}")
        print(f"  Query: {query}")
        print(f"  Result: {result}")
        print()
    
    print("-" * 60)
    print("Risk level tests:")
    
    risk_tests = [
        "SELECT * FROM users",
        "INSERT INTO logs VALUES (1)",
        "UPDATE users SET name = 'x'",
        "DELETE FROM users WHERE id = 1",
    ]
    
    for query in risk_tests:
        level = get_query_risk_level(query)
        print(f"  {query}: {level}")


if __name__ == "__main__":
    test_validator()






schema_discovery.py

"""
Dynamic Schema Discovery for Incident Manager.

Queries Oracle metadata to discover table schemas dynamically.
Used by the LLM to understand available columns before writing queries.
"""

import os
import asyncio
from typing import Dict, List, Optional, Any
from utils.oracle_connection import execute_oracle_query_async

os.environ["OTEL_SDK_DISABLED"] = "true"

import logging
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Oracle Metadata Queries
# ─────────────────────────────────────────────────────────────────────────────

async def get_table_schema(
    table_name: str,
    app: str = "cbs",
    db_instance: str = "main"
) -> Dict[str, Any]:
    """
    Get column information for a table from Oracle ALL_TAB_COLUMNS.
    
    Args:
        table_name: Name of the table
        app: Application name (cbs, optimus, idp)
        db_instance: Database instance
        
    Returns:
        Dict with columns and their properties
    """
    # Different owners for different apps
    owner_map = {
        "cbs": "FISONLPD",
        "optimus": "IDP",
        "idp": "IDP",
    }
    owner = owner_map.get(app.lower(), "FISONLPD")
    
    query = """
        SELECT 
            column_name, 
            data_type, 
            data_length, 
            data_precision,
            data_scale,
            nullable,
            column_id
        FROM all_tab_columns 
        WHERE owner = :owner 
          AND table_name = UPPER(:table_name)
        ORDER BY column_id
    """
    
    params = {
        "owner": owner,
        "table_name": table_name
    }
    
    try:
        result = await execute_oracle_query_async(
            query,
            app=app,
            db_instance=db_instance,
            params=params
        )
        
        if result.get("success"):
            data = result.get("data", [])
            columns = []
            for row in data:
                columns.append({
                    "name": row[0],
                    "type": row[1],
                    "length": row[2],
                    "precision": row[3],
                    "scale": row[4],
                    "nullable": row[5],
                    "position": row[6]
                })
            
            return {
                "table_name": table_name.upper(),
                "owner": owner,
                "columns": columns,
                "column_count": len(columns)
            }
        else:
            return {"error": result.get("error")}
    
    except Exception as e:
        logger.error(f"[get_table_schema] Error: {e}")
        return {"error": str(e)}


async def list_tables(
    pattern: str = "%",
    app: str = "cbs",
    db_instance: str = "main"
) -> List[str]:
    """
    List tables matching a pattern from USER_TABLES.
    
    Args:
        pattern: SQL LIKE pattern (e.g., '%GLD%', '%INF%')
        app: Application name
        db_instance: Database instance
        
    Returns:
        List of table names
    """
    query = """
        SELECT table_name 
        FROM user_tables 
        WHERE table_name LIKE UPPER(:pattern)
        ORDER BY table_name
    """
    
    params = {"pattern": pattern}
    
    try:
        result = await execute_oracle_query_async(
            query,
            app=app,
            db_instance=db_instance,
            params=params
        )
        
        if result.get("success"):
            return [row[0] for row in result.get("data", [])]
        else:
            logger.error(f"[list_tables] Error: {result.get('error')}")
            return []
    
    except Exception as e:
        logger.error(f"[list_tables] Error: {e}")
        return []


async def get_table_comments(
    table_name: str,
    app: str = "cbs",
    db_instance: str = "main"
) -> Optional[str]:
    """
    Get comments on a table from USER_TAB_COMMENTS.
    
    Args:
        table_name: Table name
        app: Application
        db_instance: Database instance
        
    Returns:
        Table comments or None
    """
    owner_map = {
        "cbs": "FISONLPD",
        "optimus": "IDP",
        "idp": "IDP",
    }
    owner = owner_map.get(app.lower(), "FISONLPD")
    
    query = """
        SELECT comments
        FROM all_tab_comments
        WHERE owner = :owner AND table_name = UPPER(:table_name)
    """
    
    params = {"owner": owner, "table_name": table_name}
    
    try:
        result = await execute_oracle_query_async(
            query,
            app=app,
            db_instance=db_instance,
            params=params
        )
        
        if result.get("success"):
            data = result.get("data", [])
            if data and data[0]:
                return data[0][0]
        return None
    
    except Exception as e:
        logger.error(f"[get_table_comments] Error: {e}")
        return None


async def get_column_comments(
    table_name: str,
    app: str = "cbs",
    db_instance: str = "main"
) -> Dict[str, str]:
    """
    Get comments on columns from USER_COL_COMMENTS.
    
    Args:
        table_name: Table name
        app: Application
        db_instance: Database instance
        
    Returns:
        Dict mapping column_name to comment
    """
    owner_map = {
        "cbs": "FISONLPD",
        "optimus": "IDP",
        "idp": "IDP",
    }
    owner = owner_map.get(app.lower(), "FISONLPD")
    
    query = """
        SELECT column_name, comments
        FROM all_col_comments
        WHERE owner = :owner AND table_name = UPPER(:table_name)
          AND comments IS NOT NULL
    """
    
    params = {"owner": owner, "table_name": table_name}
    
    try:
        result = await execute_oracle_query_async(
            query,
            app=app,
            db_instance=db_instance,
            params=params
        )
        
        if result.get("success"):
            return {row[0]: row[1] for row in result.get("data", [])}
        return {}
    
    except Exception as e:
        logger.error(f"[get_column_comments] Error: {e}")
        return {}


async def get_primary_keys(
    table_name: str,
    app: str = "cbs",
    db_instance: str = "main"
) -> List[str]:
    """
    Get primary key columns for a table.
    
    Args:
        table_name: Table name
        app: Application
        db_instance: Database instance
        
    Returns:
        List of primary key column names
    """
    owner_map = {
        "cbs": "FISONLPD",
        "optimus": "IDP",
        "idp": "IDP",
    }
    owner = owner_map.get(app.lower(), "FISONLPD")
    
    query = """
        SELECT a.column_name
        FROM all_cons_columns a
        JOIN all_constraints c ON a.constraint_name = c.constraint_name
        WHERE c.owner = :owner 
          AND c.table_name = UPPER(:table_name)
          AND c.constraint_type = 'P'
        ORDER BY a.position
    """
    
    params = {"owner": owner, "table_name": table_name}
    
    try:
        result = await execute_oracle_query_async(
            query,
            app=app,
            db_instance=db_instance,
            params=params
        )
        
        if result.get("success"):
            return [row[0] for row in result.get("data", [])]
        return []
    
    except Exception as e:
        logger.error(f"[get_primary_keys] Error: {e}")
        return []


async def get_foreign_keys(
    table_name: str,
    app: str = "cbs",
    db_instance: str = "main"
) -> List[Dict[str, str]]:
    """
    Get foreign key relationships for a table.
    
    Args:
        table_name: Table name
        app: Application
        db_instance: Database instance
        
    Returns:
        List of dicts with FK details
    """
    owner_map = {
        "cbs": "FISONLPD",
        "optimus": "IDP",
        "idp": "IDP",
    }
    owner = owner_map.get(app.lower(), "FISONLPD")
    
    query = """
        SELECT
            a.column_name,
            c_r.table_name AS referenced_table,
            b.column_name AS referenced_column
        FROM all_cons_columns a
        JOIN all_constraints c ON a.constraint_name = c.constraint_name
        JOIN all_constraints c_r ON c.r_constraint_name = c_r.constraint_name
        JOIN all_cons_columns b ON c_r.constraint_name = b.constraint_name AND b.position = a.position
        WHERE c.owner = :owner 
          AND c.table_name = UPPER(:table_name)
          AND c.constraint_type = 'R'
        ORDER BY a.position
    """
    
    params = {"owner": owner, "table_name": table_name}
    
    try:
        result = await execute_oracle_query_async(
            query,
            app=app,
            db_instance=db_instance,
            params=params
        )
        
        if result.get("success"):
            fks = []
            for row in result.get("data", []):
                fks.append({
                    "column": row[0],
                    "referenced_table": row[1],
                    "referenced_column": row[2]
                })
            return fks
        return []
    
    except Exception as e:
        logger.error(f"[get_foreign_keys] Error: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# High-level discovery functions
# ─────────────────────────────────────────────────────────────────────────────

async def discover_table(
    table_name: str,
    app: str = "cbs",
    db_instance: str = "main",
    include_comments: bool = True,
    include_relations: bool = True
) -> Dict[str, Any]:
    """
    Comprehensive table discovery.
    
    Returns schema, comments, primary keys, and foreign keys.
    
    Args:
        table_name: Table name
        app: Application
        db_instance: Database instance
        include_comments: Include table/column comments
        include_relations: Include FK relationships
        
    Returns:
        Complete table information
    """
    schema = await get_table_schema(table_name, app, db_instance)
    
    if "error" in schema:
        return schema
    
    if include_comments:
        schema["table_comment"] = await get_table_comments(table_name, app, db_instance)
        schema["column_comments"] = await get_column_comments(table_name, app, db_instance)
    
    if include_relations:
        schema["primary_keys"] = await get_primary_keys(table_name, app, db_instance)
        schema["foreign_keys"] = await get_foreign_keys(table_name, app, db_instance)
    
    return schema


async def search_tables_by_keyword(
    keyword: str,
    app: str = "cbs",
    db_instance: str = "main"
) -> List[Dict[str, Any]]:
    """
    Search tables by keyword in table name or comments.
    
    Args:
        keyword: Search keyword
        app: Application
        db_instance: Database instance
        
    Returns:
        List of matching tables
    """
    pattern = f"%{keyword}%"
    
    # Get tables matching pattern
    tables = await list_tables(pattern, app, db_instance)
    
    results = []
    for table_name in tables:
        table_comment = await get_table_comments(table_name, app, db_instance)
        results.append({
            "table_name": table_name,
            "comment": table_comment
        })
    
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Common table references for quick lookup
# ─────────────────────────────────────────────────────────────────────────────

# Common CBS tables with their purposes (fallback when metadata unavailable)
CBS_COMMON_TABLES = {
    "GLDM": "General Ledger Daily Master - account balances, posting masks",
    "INFZ": "Transaction freeze records - manual freezes",
    "NPAR": "NPA debit freeze records",
    "ACD3": "Account preferences - email/mobile flags",
    "STTM": "Static tables master",
    "DPD": "Days Past Due - loan delinquency",
    "ACCOUNT": "Account master table",
}

OPTIMUS_COMMON_TABLES = {
    "USERS": "User accounts",
    "DEVICES": "Registered devices",
    "SESSIONS": "Active sessions",
    "OAUTH_TOKENS": "OAuth tokens",
    "MFA_RECORDS": "MFA registration records",
}

IDP_COMMON_TABLES = {
    "USERS": "User identities",
    "OAUTH_TOKENS": "OAuth tokens",
    "MFA_RECORDS": "MFA records",
    "DEVICES": "Trusted devices",
    "SESSIONS": "Auth sessions",
}


def get_common_tables(app: str) -> Dict[str, str]:
    """Get common tables for an app (fallback reference)."""
    app_lower = app.lower()
    if app_lower == "cbs":
        return CBS_COMMON_TABLES
    elif app_lower == "optimus":
        return OPTIMUS_COMMON_TABLES
    elif app_lower == "idp":
        return IDP_COMMON_TABLES
    return {}



this is service_metadta.py
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
from tools.app_config import get_jaeger_endpoint as _get_jaeger_endpoint_from_app_config


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






this is service _metadta.yaml


# Service Metadata Configuration
# This provides optional predefined tags for services (if available)
# Discovery will work even without this - this is just for validation/guidance
# Zero dependency on support teams for discovery - all discovered via APIs

optimus:
  jaeger_endpoint: "https://tracing.uat-opt.idfcfirstbank.com/api"
  jaeger_prod_endpoint: "https://tracing.prod-opt.idfcfirstbank.com/api"
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





this is tool.py

from glob import glob
import json
import asyncio
from utils.logs import execute_elk_query
from utils.oracle_connection import execute_oracle_query_async
from crewai.tools import BaseTool
import os

from utils.http_calls import http_client_post_async, http_client_get_async

os.environ["OTEL_SDK_DISABLED"] = "true"

from glob import glob

import traceback
import logging
import threading

logger = logging.getLogger(__name__)


import json as _json
import time as _time
import base64 as _base64
import datetime as _datetime
import httpx
from collections import defaultdict

JAEGER_API_BASE = os.getenv("JAEGER_API_BASE", "https://tracing.uat-opt.idfcfirstbank.com/api")
JAEGER_AUTH_TOKEN = os.getenv("JAEGER_AUTH_TOKEN")

_JAEGER_EXCLUDED_OPS = {
    'get', 'set', 'sql:prepare', 'sql:query', 'sql-conn-query', 'sql-rows-next',
    'sql-tx-begin', 'sql-tx-commit', 'sql:exec', 'sql-prepare', 'sql-conn-exec',
    'sql-stmt-close',
    'sql.conn.prepare', 'sql.stmt.query', 'sql.stmt.exec', 'sql.stmt.close',
    'sql.tx.begin', 'sql.tx.commit', 'sql.conn.exec', 'sql.rows.next',
    'persistence.sql.findSessionBySignature', 'persistence.sql.toRequest',
    'persistence.sql.GetConcreteClient',
}
_JAEGER_EXCLUDED_SERVICES = {
    'RISK-CONTROL-REQUEST-METADATA-PROCESSOR',
    'RISK-CONTROL-USER-METADATA-PROCESSOR',
    'OAUTH-SERVER',
}
_JAEGER_MAX_PAYLOAD_CHARS = 450


def _jaeger_auth_headers():
    if JAEGER_AUTH_TOKEN:
        return {"Authorization": f"Basic {JAEGER_AUTH_TOKEN}"}
    return {}


def _jaeger_us_to_ist(microseconds_ts):
    from datetime import timezone, timedelta
    seconds = microseconds_ts / 1_000_000
    dt_utc = _datetime.datetime.fromtimestamp(seconds, tz=timezone.utc)
    ist_tz = timezone(timedelta(hours=5, minutes=30))
    return dt_utc.astimezone(ist_tz).strftime('%Y-%m-%d %H:%M:%S IST')


def _jaeger_trace_has_error(trace) -> bool:
    for span in trace.get("spans", []):
        for tag in span.get("tags", []):
            if tag.get("key") == "http.status_code":
                code = str(tag.get("value", ""))
                if code and not code.startswith("2"):
                    return True
            if tag.get("key") == "error" and tag.get("value") in (True, "true", "True"):
                return True
        for log in span.get("logs", []):
            for f in log.get("fields", []):
                if f.get("key") == "level" and str(f.get("value", "")).lower() in ("error", "fatal", "critical"):
                    return True
    return False


def _jaeger_process_trace(trace):
    spans = trace.get("spans", [])
    processes = trace.get("processes", {})

    id_to_span = {s["spanID"]: s for s in spans}
    pid_to_service = {pid: p["serviceName"] for pid, p in processes.items()}
    children_map = defaultdict(list)

    all_ids = set(id_to_span.keys())
    child_ids = set()

    for span in spans:
        for ref in span.get("references", []):
            parent_id = ref.get("spanID")
            if parent_id in id_to_span:
                children_map[parent_id].append(span["spanID"])
                child_ids.add(span["spanID"])

    roots = all_ids - child_ids
    if not roots:
        return []

    result = []

    def process_span(span_id, depth=0):
        span = id_to_span[span_id]
        indent = "    " * depth
        op_name = span.get("operationName", "")
        service_name = pid_to_service.get(span.get("processID"), "")

        if depth == 0:
            result.append("Time: " + _jaeger_us_to_ist(span["startTime"]))

        is_noise = op_name in _JAEGER_EXCLUDED_OPS or service_name in _JAEGER_EXCLUDED_SERVICES

        if not is_noise:
            result.append(f"{indent}{span_id} - {service_name} - {op_name}")
            for tag in span.get("tags", []):
                if tag.get("key") == "http.status_code":
                    result.append(f"{indent}HTTP Status: {tag.get('value')}")

            for log in span.get("logs", []):
                fields = {f.get("key"): f.get("value") for f in log.get("fields", [])}

                level = str(fields.get("level", "")).lower()
                event = fields.get("event", "")
                if level in ("error", "fatal", "critical") and event:
                    cls = fields.get("Class") or fields.get("class") or ""
                    method = fields.get("Method") or fields.get("method") or ""
                    ctx = f" [{cls}.{method}]" if (cls or method) else ""
                    result.append(f"{indent}  ERROR{ctx}: {event}")

                for payload_key in ("request", "response"):
                    if fields.get(payload_key):
                        result.append(f"{indent}  {payload_key}: {str(fields[payload_key])[:_JAEGER_MAX_PAYLOAD_CHARS]}")

        for child_id in children_map.get(span_id, []):
            process_span(child_id, depth + 1)

    for root_id in roots:
        process_span(root_id)

    return result


async def _jaeger_fetch(service: str, tag_name: str, tag_value: str):
    now_us = int(_time.time() * 1_000_000)
    start_us = now_us - 72 * 3600 * 1_000_000
    end_us = now_us

    if tag_name == "mobile_number" and not tag_value.startswith("+"):
        if len(tag_value) == 10:
            tag_value = f"+91{tag_value}"

    tags = _json.dumps({tag_name: tag_value})
    headers = _jaeger_auth_headers()
    params = {"service": service, "start": start_us, "end": end_us, "limit": 100, "tags": tags}

    logger.info(f"[JAEGER] service={service} {tag_name}={tag_value}")

    try:
        async with httpx.AsyncClient(timeout=45) as client:
            response = await client.get(f"{JAEGER_API_BASE}/traces", params=params, headers=headers)
            if response.status_code != 200:
                logger.error(f"[JAEGER] API returned {response.status_code}: {response.text}")
                return {"service": service, "tag_name": tag_name, "tag_value": tag_value,
                        "total_traces_scanned": 0, "failed_traces": [],
                        "error": f"Jaeger API returned {response.status_code}"}
            data = response.json().get("data", [])
    except Exception as e:
        logger.error(f"[JAEGER] Request failed: {e}")
        return {"service": service, "tag_name": tag_name, "tag_value": tag_value,
                "total_traces_scanned": 0, "failed_traces": [], "error": str(e)}

    if not data:
        return {"service": service, "tag_name": tag_name, "tag_value": tag_value,
                "total_traces_scanned": 0, "failed_traces": [],
                "message": f"No traces found for {tag_name}={tag_value} in {service}"}

    failed_traces_raw = [t for t in data if _jaeger_trace_has_error(t)]

    failed_traces_text = []
    for trace in failed_traces_raw:
        lines = _jaeger_process_trace(trace)
        if lines:
            failed_traces_text.append("\n".join(lines))

    logger.info(f"[JAEGER] Done: scanned={len(data)} failed={len(failed_traces_text)}")

    return {
        "service": service,
        "tag_name": tag_name,
        "tag_value": tag_value,
        "total_traces_scanned": len(data),
        "total_failed": len(failed_traces_text),
        "failed_traces": failed_traces_text
    }


# ─────────────────────────────────────────
# TOKEN MANAGER — CBS only, unchanged
# ─────────────────────────────────────────

class GiftWrappingTokenManager:

    def __init__(self):
        self.token = None
        self.token_expiry = None
        self._token_lock = threading.Lock()

    async def _get_token(self):
        client_id = os.getenv('GIFTWRAP_ENT_AUTH_CLIENT_ID')
        client_secret = os.getenv('GIFTWRAP_ENT_AUTH_CLIENT_SECRET')
        oauth_url = os.getenv('GIFTWRAP_ENT_AUTH_TOKEN_URL')

        if not client_id or not client_secret:
            raise ValueError("GIFTWRAP_ENT_AUTH_CLIENT_ID and GIFTWRAP_ENT_AUTH_CLIENT_SECRET must be set")

        form_data = {
            'client_id': client_id,
            'client_secret': client_secret,
            'grant_type': 'client_credentials'
        }

        headers = {
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        response = await http_client_post_async(url=oauth_url, headers=headers, data=None, form_data=form_data)
        json_resp = response.json
        return json_resp['access_token']

    async def get_valid_token(self):
        import time
        with self._token_lock:
            if self.token is None or self.token_expiry is None or time.time() >= self.token_expiry:
                self.token = await self._get_token()
                self.token_expiry = time.time() + (30 * 60) - 60
            return self.token

giftwrap_token_manager = GiftWrappingTokenManager()

async def make_giftwrap_api_call_async(endpoint: str, method: str, parameters: dict, headers: dict = None, query_params: dict = None):
    import uuid

    base_url = os.getenv('GIFTWRAP_API_BASE_URL')

    token = await giftwrap_token_manager.get_valid_token()

    request_headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json',
        'User-Agent': 'incident-agent/1.0',
        'X-B3-Sampled': '1',
        'X-B3-SpanId': f'{uuid.uuid4()}',
        'X-B3-TraceId': f'{uuid.uuid4()}'
    }

    if headers:
        request_headers.update(headers)

    url = f"{base_url}{endpoint}"
    if query_params:
        query_string = '&'.join([f"{k}={v}" for k, v in query_params.items() if v is not None])
        if query_string:
            url = f"{url}?{query_string}"

    logger.info(f"[GiftWrapping API Async] Calling {method} {url}")
    logger.info(f"[GiftWrapping API Async] Parameters: {parameters}")
    logger.info(f"[GiftWrapping API Async] Headers: {request_headers}")

    if method == 'GET':
        response = await http_client_get_async(url=url, headers=request_headers)
    elif method == 'POST':
        response = await http_client_post_async(url, headers=request_headers, json=parameters)
    else:
        raise ValueError(f"Unsupported HTTP method: {method}")

    return response.json


# ─────────────────────────────────────────
# ENTRY POINT — app-based routing
# ─────────────────────────────────────────

def get_tool_list(app: str = "cbs"):
    app_key = app.lower().strip()
    logger.info(f"Loading tools for app: {app_key}")

    if app_key == "cbs":
        return _load_tools_from_folder("./tools/cbs", app="cbs", include_normalize=True)
    elif app_key == "optimus":
        return _load_tools_from_folder("./tools/optimus", app="optimus", include_normalize=False)
    else:
        logger.warning(f"Unknown app '{app}' — defaulting to CBS tools")
        return _load_tools_from_folder("./tools/cbs", app="cbs", include_normalize=True)


# ─────────────────────────────────────────
# SHARED TOOL LOADER
# ─────────────────────────────────────────

def _load_tools_from_folder(folder: str, app: str, include_normalize: bool = False):
    json_tool_list = glob(f'{folder}/*.json')

    tool_parameter_list = []
    for path in json_tool_list:
        with open(path, 'r') as f:
            d = json.loads(f.read())
            tool_parameter_list += d

    tool_list = []
    for tool in tool_parameter_list:
        logger.info(f"Adding tool: {tool['name']} (type={tool.get('type')}, db_instance={tool.get('db_instance', 'main')})")
        tool_list.append(_make_tool(tool, app))

    if include_normalize:
        tool_list.append(NormalizeAccountTool())

    return tool_list


def _make_tool(tool, app: str):
    tool_name = tool['name']
    tool_type = tool.get('type', 'UNKNOWN')
    tool_query = tool.get('query', '')

    # Capture app and db_instance in closure for SQL tools
    _app = app
    _db_instance = tool.get('db_instance', 'main')

    if not tool.get('parameters'):
        tool['parameters'] = [{'name': 'key', 'description': 'Input value'}]

    param_name = tool['parameters'][0]['name']
    param_description = tool['parameters'][0]['description']

    # ── SQL ──
    if tool_type == 'SQL':
        class SQLTool(BaseTool):
            name: str = tool_name
            description: str = (
                tool['purpose'] + '\nEnter the ' + param_name + ' in key, it should contain ' + param_description
            )

            def regularize_id(self, incoming_id: str) -> str:
                raw = str(incoming_id).strip()
                raw = raw.removeprefix('003')
                return raw.lstrip('0')

            def _run(self, key: str):
                return asyncio.run(self._arun(key))

            async def _arun(self, key: str):
                clean_value = self.regularize_id(str(key).strip())

                # CBS-specific account number formatting
                if self.name in ["verify_sms_required_flag", "get_account_level_mobile",
                                 "get_email_preference_flag", "check_posting_restriction_mask",
                                 "find_ucic_from_account"]:
                    processed_value = f"003{clean_value.zfill(16)}"
                elif self.name in ["get_account_level_email", "get_ucic_email",
                                   "check_transaction_freeze", "check_npa_debit_freeze"]:
                    processed_value = f"{clean_value.zfill(16)}"
                elif self.name in ["find_account_from_ucic"]:
                    clean_value = clean_value[:-1]
                    processed_value = f"{clean_value.zfill(16)}"
                else:
                    # Optimus SQL tools and any future apps — no special formatting
                    processed_value = clean_value

                final_query = tool_query.replace(f":{param_name}", f"'{processed_value}'")
                logger.info(f"[{self.name}] app={_app} db={_db_instance} query={final_query}")
                try:
                    return await execute_oracle_query_async(
                        final_query,
                        app=_app,
                        db_instance=_db_instance
                    )
                except Exception as e:
                    return f"Query Error: {str(e)}"

        return SQLTool()

    # ── ELK ──
    elif tool_type == 'ELK':
        class ELKTool(BaseTool):
            name: str = tool['name']
            description: str = (
                tool['purpose'] + '\nEnter the ' + tool['parameters'][0]['name'] +
                ' in key, it should contain ' + tool['parameters'][0]['description']
            )

            def _run(self, key: str):
                return asyncio.run(self._arun(key))

            async def _arun(self, key: str):
                try:
                    pname = tool['parameters'][0]['name']
                    query_json = tool['query']
                    result = await execute_elk_query(query_json, parameter_name=pname, parameter_value=key)
                    return result
                except Exception as e:
                    return f"ELK query failed: {str(e)}"

        return ELKTool()

    # ── JAEGER ──
    elif tool_type == 'JAEGER':
        from pydantic import BaseModel

        jaeger_service = tool.get('service', '')

        class JaegerToolSchema(BaseModel):
            tag_name: str
            tag_value: str

        class JaegerTool(BaseTool):
            name: str = tool_name
            description: str = (
                tool['purpose']
                + f"\nThis queries the '{jaeger_service}' service."
                + "\ntag_name: identifier type (ucic, customer_id, mobile_number, username, user_id)"
                + "\ntag_value: the actual identifier value"
                + "\nNote: if ucic/customer_id is provided, the tool automatically tries "
                  "both tag keys internally — you do not need to retry manually."
                + "\nOnly failed/error traces are returned (non-2xx HTTP status, error tags, "
                  "or error-level log events). If no failures are found, the issue may not be "
                  "reproducible in logs, or login is working fine for this identifier."
            )

            args_schema: type = JaegerToolSchema

            def _run(self, tag_name: str, tag_value: str):
                return asyncio.run(self._arun(tag_name, tag_value))

            async def _arun(self, tag_name: str, tag_value: str):
                tag_name = tag_name.strip().lower()
                tag_value = str(tag_value).strip()

                tried = []
                attempt_order = [tag_name]

                if tag_name in ("ucic", "customer_id", "individualucic", "entityucic"):
                    for candidate in ("ucic", "customer_id"):
                        if candidate not in attempt_order:
                            attempt_order.append(candidate)

                last_result = None

                for candidate_tag in attempt_order:
                    tried.append(candidate_tag)
                    try:
                        result = await _jaeger_fetch(
                            service=jaeger_service,
                            tag_name=candidate_tag,
                            tag_value=tag_value
                        )
                    except Exception as e:
                        logger.error(f"[JaegerTool {self.name}] failed on tag_name={candidate_tag}: {e}")
                        result = {"total_traces_scanned": 0, "failed_traces": [], "error": str(e)}

                    last_result = result

                    scanned = result.get("total_traces_scanned", 0)
                    if scanned and scanned > 0:
                        result["tag_name_used"] = candidate_tag
                        result["tag_names_tried"] = tried
                        logger.info(f"[JaegerTool {self.name}] found {scanned} traces using tag_name={candidate_tag}")
                        return result

                logger.info(f"[JaegerTool {self.name}] no traces found after trying tag_names={tried}")
                if last_result is not None:
                    last_result["tag_names_tried"] = tried
                    return last_result
                return {"total_traces_scanned": 0, "failed_traces": [], "tag_names_tried": tried}

        return JaegerTool()

    # ── UNKNOWN ──
    else:
        class TempTool(BaseTool):
            name: str = tool_name
            description: str = tool['purpose']

            def _run(self, *args, **kwargs):
                return "NOT AVAILABLE"

            async def _arun(self, key: str):
                return "NOT AVAILABLE"

        return TempTool()




class NormalizeAccountTool(BaseTool):
    name: str = "normalize_account_number"
    description: str = (
        "Normalizes account numbers to a standard format for comparison. "
        "Input: Account number in any format. Output: Normalized account number. "
        "Use this to compare account numbers from different sources."
    )

    def _run(self, account_number: str):
        return asyncio.run(self._arun(account_number))

    async def _arun(self, account_number: str):
        raw = str(account_number).strip()
        raw = raw.removeprefix('003')
        return raw.lstrip('0')
