this is orcel conenciton.py 
import asyncio
import os
import json
import logging
import threading
from typing import Optional, Dict
import oracledb
from utils.observability import get_tracer

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────
# DB CONFIG LOADER
# ─────────────────────────────────────────

_db_config: Optional[dict] = None

def _load_db_config():
    global _db_config
    config_path = os.getenv("DB_CONFIG_PATH", "./db_config.json")
    with open(config_path) as f:
        _db_config = json.load(f)
    logger.info(f"Loaded DB config from {config_path}: apps={list(_db_config.keys())}")

def get_db_config() -> dict:
    if _db_config is None:
        _load_db_config()
    return _db_config

# ─────────────────────────────────────────
# MULTI-POOL MANAGEMENT
# ─────────────────────────────────────────

_pools: Dict[str, oracledb.ConnectionPool] = {}
_pool_lock = threading.Lock()


def _build_dsn(app: str, db_instance: str) -> str:
    config = get_db_config()
    app_config = config.get(app)
    if not app_config:
        raise ValueError(f"No DB config found for app '{app}'")
    
    oracle_config = app_config.get("oracle", {})
    instance_config = oracle_config.get(db_instance)
    if not instance_config:
        available = list(oracle_config.keys())
        raise ValueError(f"No DB instance '{db_instance}' for app '{app}'. Available: {available}")
    
    host = instance_config["host"]
    port = instance_config["port"]
    service = instance_config.get("service") or instance_config.get("service_name")
    use_tls = instance_config.get("tls", True)
    protocol = "TCPS" if use_tls else "TCP"
    
    dsn = (
        f"(DESCRIPTION="
        f"(ADDRESS=(PROTOCOL={protocol})(HOST={host})(PORT={port}))"
        f"(CONNECT_DATA=(SERVER=DEDICATED)(SERVICE_NAME={service})))"
    )
    return dsn


def _get_credentials(app: str) -> tuple:
    username = os.getenv(f"{app.upper()}_ORACLE_USERNAME")
    password = os.getenv(f"{app.upper()}_ORACLE_PASSWORD")
    if not username or not password:
        raise ValueError(
            f"Missing credentials: {app.upper()}_ORACLE_USERNAME and/or "
            f"{app.upper()}_ORACLE_PASSWORD not set in environment"
        )
    return username, password


def _get_ssl_dn(app: str, db_instance: str) -> Optional[str]:
    """Return ssl_server_cert_dn if configured, else None."""
    config = get_db_config()
    instance_config = config.get(app, {}).get("oracle", {}).get(db_instance, {})
    return instance_config.get("ssl_server_cert_dn")


def _get_pool(app: str, db_instance: str) -> oracledb.ConnectionPool:
    pool_key = f"{app}:{db_instance}"
    
    if pool_key in _pools:
        return _pools[pool_key]
    
    with _pool_lock:
        # Double-check after acquiring lock
        if pool_key in _pools:
            return _pools[pool_key]
        
        logger.info(f"Creating Oracle pool for {pool_key}")
        dsn = _build_dsn(app, db_instance)
        username, password = _get_credentials(app)
        ssl_dn = _get_ssl_dn(app, db_instance)
        
        pool_params = {
            "user": username,
            "password": password,
            "min": 2,
            "max": 10,
            "increment": 1,
        }
        
        if ssl_dn:
            pool_params["ssl_server_cert_dn"] = ssl_dn
            pool_params["ssl_server_dn_match"] = True
        
        pool = oracledb.create_pool(
            dsn=dsn,
            params=oracledb.PoolParams(**pool_params)
        )
        
        _pools[pool_key] = pool
        logger.info(f"Oracle pool created for {pool_key} → {dsn}")
        return pool


# ─────────────────────────────────────────
# QUERY EXECUTION
# ─────────────────────────────────────────

def _execute_query_sync(query: str, app: str, db_instance: str, params: Optional[dict] = None) -> dict:
    pool = _get_pool(app, db_instance)
    with pool.acquire() as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, params or {})
            if cursor.description:
                data = cursor.fetchall()
                return {"success": True, "data": data, "error": None}
            connection.commit()
            return {"success": True, "data": [], "error": None}


async def execute_oracle_query_async(
    query: str,
    app: str = "cbs",
    db_instance: str = "main",
    params: dict = None
) -> dict:
    tracer = get_tracer()
    with tracer.start_as_current_span("execute_oracle_query_async") as span:
        span.set_attribute("db.statement", query)
        span.set_attribute("db.app", app)
        span.set_attribute("db.instance", db_instance)
        try:
            result = await asyncio.to_thread(
                _execute_query_sync, query, app, db_instance, params
            )
            if result["success"] and result["data"]:
                span.set_attribute("db.rows_returned", len(result["data"]))
            return result
        except Exception as e:
            logger.error(f"DB Error [{app}:{db_instance}]: {e}", exc_info=True)
            span.record_exception(e)
            span.set_status(status=False, description=str(e))
            return {"success": False, "data": None, "error": str(e)}


# ─────────────────────────────────────────
# CLEANUP
# ─────────────────────────────────────────

async def close_all_oracle_pools() -> None:
    global _pools
    for pool_key, pool in _pools.items():
        try:
            logger.info(f"Closing Oracle pool: {pool_key}")
            await asyncio.to_thread(pool.close)
            logger.info(f"Closed Oracle pool: {pool_key}")
        except Exception as e:
            logger.error(f"Failed to close pool {pool_key}: {e}")
    _pools = {}
async def close_oracle_async_pool() -> None:
    await close_all_oracle_pools()

# ─────────────────────────────────────────
# STARTUP VALIDATION
# ─────────────────────────────────────────

def validate_db_config():
    """Call at app startup to fail fast if config is broken."""
    config = get_db_config()
    for app_name, app_cfg in config.items():
        for db_type, instances in app_cfg.items():
            for instance_name, instance_cfg in instances.items():
                # Check required fields
                for field in ("host", "port"):
                    if field not in instance_cfg:
                        raise ValueError(
                            f"db_config.json: {app_name}.{db_type}.{instance_name} missing '{field}'"
                        )
                if "service" not in instance_cfg and "service_name" not in instance_cfg:
                    raise ValueError(
                        f"db_config.json: {app_name}.{db_type}.{instance_name} missing 'service' or 'service_name'"
                    )
            
                username = os.getenv(f"{app_name.upper()}_ORACLE_USERNAME")
                password = os.getenv(f"{app_name.upper()}_ORACLE_PASSWORD")
                if not username or not password:
                    logger.warning(
                        f"Missing env vars: {app_name.upper()}_ORACLE_USERNAME / "
                        f"{app_name.upper()}_ORACLE_PASSWORD"
                    )
    logger.info("DB config validation passed")



    and im getting this issue 
    2026-06-30 07:04:48,264 - tools.tool - INFO - [idp_password_expiry_check] app=optimus db=platform query=SELECT * FROM IDP.customer WHERE ucic = '1055678509'
2026-06-30 07:05:08,285 - utils.oracle_connection - ERROR - DB Error [optimus:platform]: DPY-6005: cannot connect to database (CONNECTION_ID=5SW+csYb1ZjpgoU/B42WEg==).
timed out
Traceback (most recent call last):
  File "src/oracledb/impl/thin/connection.pyx", line 421, in oracledb.thin_impl.ThinConnImpl._connect_with_address
  File "src/oracledb/impl/thin/protocol.pyx", line 268, in oracledb.thin_impl.Protocol._connect_phase_one
  File "src/oracledb/impl/thin/protocol.pyx", line 402, in oracledb.thin_impl.Protocol._connect_tcp
  File "/usr/local/lib/python3.11/socket.py", line 851, in create_connection
    raise exceptions[0]
  File "/usr/local/lib/python3.11/socket.py", line 836, in create_connection
    sock.connect(sa)
TimeoutError: timed out

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/app/utils/oracle_connection.py", line 148, in execute_oracle_query_async
    result = await asyncio.to_thread(
             ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/asyncio/threads.py", line 25, in to_thread
    return await loop.run_in_executor(None, func_call)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/concurrent/futures/thread.py", line 58, in run
    result = self.fn(*self.args, **self.kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/app/utils/oracle_connection.py", line 126, in _execute_query_sync
    with pool.acquire() as connection:
         ^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/oracledb/pool.py", line 428, in acquire
    return oracledb.connect(
           ^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/oracledb/connection.py", line 1768, in connect
    return conn_class(dsn=dsn, pool=pool, params=params, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/oracledb/connection.py", line 889, in __init__
    impl = pool_impl.acquire(params_impl)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "src/oracledb/impl/thin/pool.pyx", line 637, in oracledb.thin_impl.ThinPoolImpl.acquire
  File "src/oracledb/impl/thin/pool.pyx", line 643, in oracledb.thin_impl.ThinPoolImpl.acquire
  File "src/oracledb/impl/thin/pool.pyx", line 639, in oracledb.thin_impl.ThinPoolImpl.acquire
  File "/usr/local/lib/python3.11/threading.py", line 363, in wait_for
    result = predicate()
             ^^^^^^^^^^^
  File "src/oracledb/impl/thin/pool.pyx", line 950, in oracledb.thin_impl.PooledConnRequest.fulfill
  File "src/oracledb/impl/thin/pool.pyx", line 602, in oracledb.thin_impl.ThinPoolImpl._process_request
  File "src/oracledb/impl/thin/pool.pyx", line 575, in oracledb.thin_impl.ThinPoolImpl._create_conn_impl
  File "src/oracledb/impl/thin/connection.pyx", line 544, in oracledb.thin_impl.ThinConnImpl.connect
  File "src/oracledb/impl/thin/connection.pyx", line 540, in oracledb.thin_impl.ThinConnImpl.connect
  File "src/oracledb/impl/thin/connection.pyx", line 482, in oracledb.thin_impl.ThinConnImpl._connect_with_params
  File "src/oracledb/impl/thin/connection.pyx", line 463, in oracledb.thin_impl.ThinConnImpl._connect_with_description
  File "src/oracledb/impl/thin/connection.pyx", line 425, in oracledb.thin_impl.ThinConnImpl._connect_with_address
  File "/usr/local/lib/python3.11/site-packages/oracledb/errors.py", line 199, in _raise_err
    raise error.exc_type(error) from cause
oracledb.exceptions.OperationalError: DPY-6005: cannot connect to database (CONNECTION_ID=5SW+csYb1ZjpgoU/B42WEg==).
timed out
╭───────────────────── ✅ Tool Execution Completed (#11) ──────────────────────╮
│                                                                              │
│  Tool Completed                                                              │
│  Tool: idp_password_expiry_check                                             │
│  Output: {'success': False, 'data': None, 'error': 'DPY-6005: cannot         │
│  connect to database (CONNECTION_ID=5SW+csYb1ZjpgoU/B42WEg==).\ntimed out'}  │
│                                      
