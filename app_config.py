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
        "elk_endpoint": "https://DCELASDBSPRR07-prod.logging.devops.idfcbank.com:9200;https://DCELASDBSPRR08-prod.logging.devops.idfcbank.com:9200;https://DCELASDBSPRR09-prod.logging.devops.idfcbank.com:9200",
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
    "sfdc asset org 3": {
        "name": "SFDC Asset Org 3",
        "db_instance": "",
        "services": [],
        "tables": [],
        "elk_endpoint": "",
        "elk_index": "",
        "jaeger_endpoint": "",
        "jaeger_prod_endpoint": "",
        "default_jaeger_service": "",
        "problem_categories": []
    },
    "finnone": {
        "name": "FinnOne",
        "db_instance": "",
        "services": [],
        "tables": [],
        "elk_endpoint": "",
        "elk_index": "",
        "jaeger_endpoint": "",
        "jaeger_prod_endpoint": "",
        "default_jaeger_service": "",
        "problem_categories": []
    },
    "sfdc jlg": {
        "name": "SFDC JLG",
        "db_instance": "",
        "services": [],
        "tables": [],
        "elk_endpoint": "",
        "elk_index": "",
        "jaeger_endpoint": "",
        "jaeger_prod_endpoint": "",
        "default_jaeger_service": "",
        "problem_categories": []
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
        "elk_endpoint": "https://dcawsmonitelasticcoord0.logging.devops.idfcbank.com:9200;https://dcawsmonitelasticcoord1.logging.devops.idfcbank.com:9200;https://dcawsmonitelasticcoord2.logging.devops.idfcbank.com:9200",
        "elk_index": "elk-optimus-prod*",
        "jaeger_endpoint": "https://tracing.uat-opt.idfcfirstbank.com/api;https://prod-jaeger-ui.obs.idfcfirstbank.com/jaeger/search",
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
        "elk_endpoint": "",
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
        "elk_endpoint": "https://DCELASDBSPRR07-prod.logging.devops.idfcbank.com:9200;https://DCELASDBSPRR08-prod.logging.devops.idfcbank.com:9200;https://DCELASDBSPRR09-prod.logging.devops.idfcbank.com:9200",
        "elk_index": "",
        "jaeger_endpoint": "",
        "jaeger_prod_endpoint": "https://prod-jaeger-ui.obs.idfcfirstbank.com/jaeger/api",
        "default_jaeger_service": "",
        "problem_categories": []
    },
    "new to bank": {
        "name": "New To Bank",
        "db_instance": "",
        "services": [],
        "tables": [],
        "elk_endpoint": "",
        "elk_index": "",
        "jaeger_endpoint": "",
        "jaeger_prod_endpoint": "https://dcawsmonitelasticcoord0.logging.devops.idfcbank.com:9200;https://dcawsmonitelasticcoord1.logging.devops.idfcbank.com:9200;https://dcawsmonitelasticcoord2.logging.devops.idfcbank.com:9200",
        "default_jaeger_service": "",
        "problem_categories": []
    },
    "electronic eoll eollection (etoll)": {
        "name": "Electronic Toll Collection (ETOLL)",
        "db_instance": "",
        "services": [],
        "tables": [],
        "elk_endpoint": "https://DCELASDBSPRR07-prod.logging.devops.idfcbank.com:9200;https://DCELASDBSPRR08-prod.logging.devops.idfcbank.com:9200;https://DCELASDBSPRR09-prod.logging.devops.idfcbank.com:9200",
        "elk_index": "",
        "jaeger_endpoint": "https://tracing.prod-etollissuer.idfcfirstbank.com/api",
        "jaeger_prod_endpoint": "",
        "default_jaeger_service": "",
        "problem_categories": []
    },
    "contact center (finesse, ivr)": {
        "name": "Contact Center (Finesse, IVR)",
        "db_instance": "",
        "services": [],
        "tables": [],
        "elk_endpoint": "https://DCELASDBSPRR07-prod.logging.devops.idfcbank.com:9200;https://DCELASDBSPRR08-prod.logging.devops.idfcbank.com:9200;https://DCELASDBSPRR09-prod.logging.devops.idfcbank.com:9200",
        "elk_index": "",
        "jaeger_endpoint": "",
        "jaeger_prod_endpoint": "",
        "default_jaeger_service": "",
        "problem_categories": []
    },
    "prime": {
        "name": "credit card - prime",
        "db_instance": "Credit Card - PRIME",
        "services": [],
        "tables": [],
        "elk_endpoint": "https://DCELASDBSPRR07-prod.logging.devops.idfcbank.com:9200;https://DCELASDBSPRR08-prod.logging.devops.idfcbank.com:9200;https://DCELASDBSPRR09-prod.logging.devops.idfcbank.com:9200",
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
        "elk_endpoint": "https://DCEST3APPPRR01-prod.logging.devops.idfcbank.com:9200;https://DCEST3APPPRR02-prod.logging.devops.idfcbank.com:9200;https://DCEST3APPPRR03-prod.logging.devops.idfcbank.com:9200",
        "elk_index": "",
        "jaeger_endpoint": "https://prod-jaeger-ui.obs.idfcfirstbank.com/jaeger/search;https://prod-jaeger-2-ui.obs.idfcbank.com/jaeger/search",
        "jaeger_prod_endpoint": "",
        "default_jaeger_service": "",
        "problem_categories": []
    },
    "customer data platform": {
        "name": "Customer Data Platform",
        "db_instance": "",
        "services": [],
        "tables": [],
        "elk_endpoint": "https://dcawsmonitelasticcoord0.logging.devops.idfcbank.com:9200;https://dcawsmonitelasticcoord1.logging.devops.idfcbank.com:9200;https://dcawsmonitelasticcoord2.logging.devops.idfcbank.com:9200",
        "elk_index": "",
        "jaeger_endpoint": "https://tracing.cdp.idfcfirstbank.com/api",
        "jaeger_prod_endpoint": "",
        "default_jaeger_service": "",
        "problem_categories": []
    },
    "api integrations": {
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
    "cms bxp": {
        "name": "CMS BXP",
        "db_instance": "",
        "services": [],
        "tables": [],
        "elk_endpoint": "https://DCELASDBSPRR07-prod.logging.devops.idfcbank.com:9200;https://DCELASDBSPRR08-prod.logging.devops.idfcbank.com:9200;https://DCELASDBSPRR09-prod.logging.devops.idfcbank.com:9200",
        "elk_index": "",
        "jaeger_endpoint": "",
        "jaeger_prod_endpoint": "",
        "default_jaeger_service": "",
        "problem_categories": []
    },
    "wealth management system (wms)": {
        "name": "Wealth Management System(WMS)",
        "db_instance": "",
        "services": [],
        "tables": [],
        "elk_endpoint": "https://DCEST3APPPRR01-prod.logging.devops.idfcbank.com:9200;https://DCEST3APPPRR02-prod.logging.devops.idfcbank.com:9200;https://DCEST3APPPRR03-prod.logging.devops.idfcbank.com:9200",
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
        "elk_endpoint": "https://DCELASDBSPRR07-prod.logging.devops.idfcbank.com:9200;https://DCELASDBSPRR08-prod.logging.devops.idfcbank.com:9200;https://DCELASDBSPRR09-prod.logging.devops.idfcbank.com:9200",
        "elk_index": "",
        "jaeger_endpoint": "",
        "jaeger_prod_endpoint": "",
        "default_jaeger_service": "",
        "problem_categories": []
    },
    "newgen (ibps, omnidocs, zapin)": {
        "name": "Newgen (IBPS, OmniDocs, Zapin)",
        "db_instance": "",
        "services": [],
        "tables": [],
        "elk_endpoint": "https://DCEST3APPPRR01-prod.logging.devops.idfcbank.com:9200;https://DCEST3APPPRR02-prod.logging.devops.idfcbank.com:9200;https://DCEST3APPPRR03-prod.logging.devops.idfcbank.com:9200",
        "elk_index": "",
        "jaeger_endpoint": "",
        "jaeger_prod_endpoint": "",
        "default_jaeger_service": "",
        "problem_categories": []
    },
    "communication hub": {
        "name": "Communication Hub",
        "db_instance": "",
        "services": [],
        "tables": [],
        "elk_endpoint": "https://dcawsmonitelasticcoord0.logging.devops.idfcbank.com:9200;https://dcawsmonitelasticcoord1.logging.devops.idfcbank.com:9200;https://dcawsmonitelasticcoord2.logging.devops.idfcbank.com:9200",
        "elk_index": "",
        "jaeger_endpoint": "https://tracing.commhub.idfcbank.com/api",
        "jaeger_prod_endpoint": "",
        "default_jaeger_service": "",
        "problem_categories": []
    },
    "fm converge": {
        "name": "FM CONVERGE",
        "db_instance": "",
        "services": [],
        "tables": [],
        "elk_endpoint": "https://DCELASDBSPRR07-prod.logging.devops.idfcbank.com:9200;https://DCELASDBSPRR08-prod.logging.devops.idfcbank.com:9200;https://DCELASDBSPRR09-prod.logging.devops.idfcbank.com:9200",
        "elk_index": "",
        "jaeger_endpoint": "",
        "jaeger_prod_endpoint": "",
        "default_jaeger_service": "",
        "problem_categories": []
    },
    "phi_neft": {
        "name": "PHI_NEFT",
        "db_instance": "",
        "services": [],
        "tables": [],
        "elk_endpoint": "https://DCEST3APPPRR01-prod.logging.devops.idfcbank.com:9200;https://DCEST3APPPRR02-prod.logging.devops.idfcbank.com:9200;https://DCEST3APPPRR03-prod.logging.devops.idfcbank.com:9200",
        "elk_index": "",
        "jaeger_endpoint": "https://prod-jaeger-ui.obs.idfcfirstbank.com/jaeger/api",
        "jaeger_prod_endpoint": "",
        "default_jaeger_service": "",
        "problem_categories": []
    },
    "oracle gl (ogl)": {
        "name": "Oracle GL (OGL)",
        "db_instance": "",
        "services": [],
        "tables": [],
        "elk_endpoint": "https://DCEST3APPPRR01-prod.logging.devops.idfcbank.com:9200;https://DCEST3APPPRR02-prod.logging.devops.idfcbank.com:9200;https://DCEST3APPPRR03-prod.logging.devops.idfcbank.com:9200",
        "elk_index": "",
        "jaeger_endpoint": "",
        "jaeger_prod_endpoint": "",
        "default_jaeger_service": "",
        "problem_categories": []
    },
    "fi gateway": {
        "name": "FI Gateway",
        "db_instance": "",
        "services": [],
        "tables": [],
        "elk_endpoint": "https://DCELASDBSPRR07-prod.logging.devops.idfcbank.com:9200;https://DCELASDBSPRR08-prod.logging.devops.idfcbank.com:9200;https://DCELASDBSPRR09-prod.logging.devops.idfcbank.com:9200",
        "elk_index": "",
        "jaeger_endpoint": "ttps://prod-jaeger-ui.obs.idfcfirstbank.com/jaeger/api",
        "jaeger_prod_endpoint": "",
        "default_jaeger_service": "",
        "problem_categories": []
    },
    "enterprise fraud risk management (falcon)": {
        "name": "Enterprise Fraud Risk Management (Falcon)",
        "db_instance": "",
        "services": [],
        "tables": [],
        "elk_endpoint": "https://DCELASDBSPRR07-prod.logging.devops.idfcbank.com:9200;https://DCELASDBSPRR08-prod.logging.devops.idfcbank.com:9200;https://DCELASDBSPRR09-prod.logging.devops.idfcbank.com:9200",
        "elk_index": "",
        "jaeger_endpoint": "ttps://prod-jaeger-ui.obs.idfcfirstbank.com/jaeger/api",
        "jaeger_prod_endpoint": "",
        "default_jaeger_service": "",
        "problem_categories": []
    },
    "valuefy": {
        "name": "Valuefy",
        "db_instance": "",
        "services": [],
        "tables": [],
        "elk_endpoint": "https://DCEST3APPPRR01-prod.logging.devops.idfcbank.com:9200;https://DCEST3APPPRR02-prod.logging.devops.idfcbank.com:9200;https://DCEST3APPPRR03-prod.logging.devops.idfcbank.com:9200",
        "elk_index": "",
        "jaeger_endpoint": "ttps://prod-jaeger-ui.obs.idfcfirstbank.com/jaeger/api",
        "jaeger_prod_endpoint": "",
        "default_jaeger_service": "",
        "problem_categories": []
    }
}
