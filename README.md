{
    "cbs": {
        "name": "Core Banking System",
        "db_instance": "main",
        "services": [
            "cbs-backend",
            "cbs-api",
            "cbs-core"
        ],
        "tables": [
            "GLDM",
            "INFZ",
            "NPAR",
            "ACD3",
            "STTM",
            "DPD",
            "ACCOUNT"
        ],
        "elk_endpoint": "",
        "elk_index": "elk-cbs-prod*",
        "jaeger_endpoint": "https://tracing.uat-cbs.idfcfirstbank.com/api",
        "jaeger_prod_endpoint": "https://tracing.prod-cbs.idfcfirstbank.com/api",
        "default_jaeger_service": "cbs-backend",
        "problem_categories": [
            "account_freeze",
            "transaction_failure",
            "balance_issue",
            "payment_failure",
            "loan_issue",
            "kyc_issue"
        ]
    },
    "optimus": {
        "name": "Mobile/Net Banking",
        "db_instance": "idp",
        "services": [
            "optimus-api",
            "optimus-login",
            "optimus-web",
            "optimus-mobile"
        ],
        "tables": [
            "users",
            "devices",
            "sessions",
            "oauth_tokens",
            "mfa_records"
        ],
        "elk_endpoint": "",
        "elk_index": "elk-optimus-prod*",
        "jaeger_endpoint": "https://tracing.uat-opt.idfcfirstbank.com/api",
        "jaeger_prod_endpoint": "https://tracing.uat-opt.idfcfirstbank.com/api",
        "default_jaeger_service": "optimus-api",
        "problem_categories": [
            "login_failure",
            "session_timeout",
            "transaction_failure",
            "app_crash",
            "password_reset_failure",
            "mfa_issue"
        ]
    },
    "idp": {
        "name": "Identity Provider",
        "db_instance": "idp",
        "services": [
            "idp-auth",
            "idp-oauth",
            "idp-mfa"
        ],
        "tables": [
            "users",
            "oauth_tokens",
            "mfa_records",
            "devices",
            "sessions"
        ],
        "elk_index": "elk-idp-prod*",
        "jaeger_endpoint": "https://tracing.uat-idp.idfcfirstbank.com/api",
        "jaeger_prod_endpoint": "https://tracing.prod-idp.idfcfirstbank.com/api",
        "default_jaeger_service": "idp-auth",
        "problem_categories": [
            "authentication_failure",
            "oauth_error",
            "mfa_failure",
            "token_expired",
            "device_registration_failure"
        ]
    },
    "phi-upi": {
        "name": "PHI-UPI",
        "db_instance": "",
        "services": [],
        "tables": [],
        "elk_endpoint": "https://dckibaappprr01-prod.logging.devops.idfcbank.com:9200",
        "elk_index": "",
        "jaeger_endpoint": "",
        "jaeger_prod_endpoint": "https://prod-jaeger-ui.obs.idfcfirstbank.com/jaeger/api",
        "default_jaeger_service": "",
        "problem_categories": []
    },
    "ntbsa": {
        "name": "New To Bank",
        "db_instance": "",
        "services": [],
        "tables": [],
        "elk_endpoint": "",
        "elk_index": "",
        "jaeger_endpoint": "",
        "jaeger_prod_endpoint": "https://aws-tracing.ntbsa.idfcfirstbank.com/api",
        "default_jaeger_service": "",
        "problem_categories": []
    },
    "finnone": {
        "name": "FinnOne",
        "db_instance": "",
        "services": [],
        "tables": [],
        "elk_endpoint": "https://dckibaappprr01-prod.logging.devops.idfcbank.com:9200",
        "elk_index": "",
        "jaeger_endpoint": "",
        "jaeger_prod_endpoint": "",
        "default_jaeger_service": "",
        "problem_categories": []
    },
    "etoll": {
        "name": "FinnOne",
        "db_instance": "",
        "services": [],
        "tables": [],
        "elk_endpoint": "https://aws-obs-kibana.logging.devops.idfcbank.com:9200",
        "elk_index": "",
        "jaeger_endpoint": "https://tracing.prod-etollissuer.idfcfirstbank.com/api",
        "jaeger_prod_endpoint": "",
        "default_jaeger_service": "",
        "problem_categories": []
    },
    "finesse": {
        "name": "Contact Center (Finesse, IVR)",
        "db_instance": "",
        "services": [],
        "tables": [],
        "elk_endpoint": "https://dckibaappprr01-prod.logging.devops.idfcbank.com:9200;https://kibanat3-prod.logging.devops.idfcbank.com:9200",
        "elk_index": "",
        "jaeger_endpoint": "",
        "jaeger_prod_endpoint": "",
        "default_jaeger_service": "",
        "problem_categories": []
    },
    "prime": {
        "name": "Credit Card - PRIME",
        "db_instance": "",
        "services": [],
        "tables": [],
        "elk_endpoint": "https://dckibaappprr01-prod.logging.devops.idfcbank.com:9200",
        "elk_index": "",
        "jaeger_endpoint": "",
        "jaeger_prod_endpoint": "",
        "default_jaeger_service": "",
        "problem_categories": []
    },
    "brnet": {
        "name": "BRNET",
        "db_instance": "",
        "services": [],
        "tables": [],
        "elk_endpoint": "https://dckibaappprr01-prod.logging.devops.idfcbank.com:9200",
        "elk_index": "",
        "jaeger_endpoint": "",
        "jaeger_prod_endpoint": "",
        "default_jaeger_service": "",
        "problem_categories": []
    },
    "cdp": {
        "name": "Customer Data Platform",
        "db_instance": "",
        "services": [],
        "tables": [],
        "elk_endpoint": "https://aws-obs-kibana.logging.devops.idfcbank.com:9200",
        "elk_index": "",
        "jaeger_endpoint": "https://tracing.cdp.idfcfirstbank.com/api",
        "jaeger_prod_endpoint": "",
        "default_jaeger_service": "",
        "problem_categories": []
    },
    "api": {
        "name": "API Integrations",
        "db_instance": "",
        "services": [],
        "tables": [],
        "elk_endpoint": "https://dcawsmonitelasticcoord0.logging.devops.idfcbank.com:9200;https://dcawsmonitelasticcoord1.logging.devops.idfcbank.com:9200;https://dcawsmonitelasticcoord2.logging.devops.idfcbank.com:9200",
        "elk_index": "",
        "jaeger_endpoint": "https://prod-jaeger-ui.obs.idfcfirstbank.com/jaeger/api",
        "jaeger_prod_endpoint": "",
        "default_jaeger_service": "",
        "problem_categories": []
    },
    "bxp": {
        "name": "CMS BXP",
        "db_instance": "",
        "services": [],
        "tables": [],
        "elk_endpoint": "https://DCELASDBSPRR07-prod.logging.devops.idfcbank.com:9200/elk-cms-applog-prod*;https://DCELASDBSPRR07-prod.logging.devops.idfcbank.com:9200/elk-cash-management-services-cms-prod*;https://DCELASDBSPRR07-prod.logging.devops.idfcbank.com:9200/elk-cms-bxp-prod*",
        "elk_index": "",
        "jaeger_endpoint": "",
        "jaeger_prod_endpoint": "",
        "default_jaeger_service": "",
        "problem_categories": []
    },
    "wms": {
        "name": "Wealth Management System(WMS)",
        "db_instance": "",
        "services": [],
        "tables": [],
        "elk_endpoint": "",
        "elk_index": "",
        "jaeger_endpoint": "https://prod-jaeger-ui.obs.idfcfirstbank.com/jaeger/api",
        "jaeger_prod_endpoint": "",
        "default_jaeger_service": "",
        "problem_categories": []
    },
    "imps": {
        "name": "IMPS",
        "db_instance": "",
        "services": [],
        "tables": [],
        "elk_endpoint": "https://dckibaappprr01-prod.logging.devops.idfcbank.com:9200",
        "elk_index": "",
        "jaeger_endpoint": "",
        "jaeger_prod_endpoint": "",
        "default_jaeger_service": "",
        "problem_categories": []
    },
    "newgen": {
        "name": "Newgen (IBPS, OmniDocs, Zapin)",
        "db_instance": "",
        "services": [],
        "tables": [],
        "elk_endpoint": "https://dckibaappprr01-prod.logging.devops.idfcbank.com:9200",
        "elk_index": "",
        "jaeger_endpoint": "",
        "jaeger_prod_endpoint": "",
        "default_jaeger_service": "",
        "problem_categories": []
    },
    "commhub": {
        "name": "Communication Hub",
        "db_instance": "",
        "services": [],
        "tables": [],
        "elk_endpoint": "https://aws-obs-kibana.logging.devops.idfcbank.com:9200",
        "elk_index": "",
        "jaeger_endpoint": "https://tracing.commhub.idfcbank.com/api",
        "jaeger_prod_endpoint": "",
        "default_jaeger_service": "",
        "problem_categories": []
    },
    "fmconv": {
        "name": "FM Converge",
        "db_instance": "",
        "services": [],
        "tables": [],
        "elk_endpoint": "https://dckibaappprr01-prod.logging.devops.idfcbank.com:9200",
        "elk_index": "",
        "jaeger_endpoint": "",
        "jaeger_prod_endpoint": "",
        "default_jaeger_service": "",
        "problem_categories": []
    },
    "phi-neft": {
        "name": "",
        "db_instance": "",
        "services": [],
        "tables": [],
        "elk_endpoint": "https://dckibaappprr01-prod.logging.devops.idfcbank.com:9200",
        "elk_index": "",
        "jaeger_endpoint": "https://prod-jaeger-ui.obs.idfcfirstbank.com/jaeger/api",
        "jaeger_prod_endpoint": "",
        "default_jaeger_service": "",
        "problem_categories": []
    },
    "ogl": {
        "name": "Oracle GL (OGL)",
        "db_instance": "",
        "services": [],
        "tables": [],
        "elk_endpoint": "https://kibanat3-prod.logging.devops.idfcbank.com:9200",
        "elk_index": "",
        "jaeger_endpoint": "",
        "jaeger_prod_endpoint": "",
        "default_jaeger_service": "",
        "problem_categories": []
    },
    "figateway": {
        "name": "FI Gateway",
        "db_instance": "",
        "services": [],
        "tables": [],
        "elk_endpoint": "https://dckibaappprr01-prod.logging.devops.idfcbank.com:9200",
        "elk_index": "",
        "jaeger_endpoint": "ttps://prod-jaeger-ui.obs.idfcfirstbank.com/jaeger/api",
        "jaeger_prod_endpoint": "",
        "default_jaeger_service": "",
        "problem_categories": []
    }
}
above one is app_config.json
belwo is app_config.py
from typing import Dict, List, Optional
from pydantic import BaseModel
from pathlib import Path
import json


class AppConfig(BaseModel):
    name: str
    db_instance: str
    services: List[str]
    tables: List[str]
    elk_index: str
    elk_endpoint: Optional[str] = None
    jaeger_endpoint: Optional[str] = None
    jaeger_prod_endpoint: Optional[str] = None
    default_jaeger_service: Optional[str] = None
    problem_categories: List[str]

def generate_app_config_from_file(filename: str) -> Dict[str, AppConfig]:
    filepath = Path(f"{Path(__file__).resolve().parent}/{filename}")
    with filepath.open('r') as file:
        try:
            raw_confg = json.loads(file.read().strip())
            return {key: AppConfig(**val) for key, val in raw_confg.items()}
        except json.JSONDecodeError as e:
            print(f"APP CONFIG PARSE FAIL - {e}")
            return {}

APPS_CONFIG: Dict[str, AppConfig] = generate_app_config_from_file("app_config.json")

def get_app_config(app: str) -> AppConfig:
    app_key = app.lower().strip()
    if app_key not in APPS_CONFIG:
        raise ValueError(
            f"App '{app}' not supported. Available: {list(APPS_CONFIG.keys())}"
        )
    return APPS_CONFIG[app_key]

def get_supported_apps() -> List[str]:
    return list(APPS_CONFIG.keys())

DEFAULT_APP_CONFIG = AppConfig(
    name="Unknown Application",
    db_instance="main",
    services=[],
    tables=[],
    elk_index="elk-*",
    default_jaeger_service=None,
    problem_categories=[]
)

def get_app_config_safe(app: str) -> AppConfig:
    try:
        return get_app_config(app)
    except ValueError:
        return DEFAULT_APP_CONFIG

def get_jaeger_endpoint(app: str) -> Optional[str]:
    if not app:
        return None
    try:
        config = get_app_config(app)
        return config.jaeger_prod_endpoint
    except ValueError:
        return None//




        





