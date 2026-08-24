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
        app_names = [a.name for a in APPS_CONFIG.values()]
        if app not in app_names:
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
        return None


def app_has_observability(app: str) -> bool:
    """True if this app has any Jaeger or ELK config to run live investigation against."""
    try:
        config = get_app_config(app)
    except ValueError:
        return False
    has_jaeger = bool(config.jaeger_endpoint or config.jaeger_prod_endpoint)
    has_elk = bool(config.elk_index)
    return has_jaeger or has_elk
