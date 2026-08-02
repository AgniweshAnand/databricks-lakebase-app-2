"""
Lakebase (Databricks-managed Postgres) connection helper.

Lakebase uses short-lived OAuth tokens as the Postgres password. This module
fetches a fresh token via the Databricks SDK and hands back a DB-API/SQLAlchemy
connection so callers never touch raw credentials.
"""

import os
import time
from contextlib import contextmanager

from databricks.sdk import WorkspaceClient
import psycopg2
from psycopg2.extras import RealDictCursor
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool

_w = WorkspaceClient()

# Cache the OAuth token briefly to avoid minting a new one on every request.
# Lakebase tokens are valid for a limited window; we refresh well before expiry.
_TOKEN_TTL_SECONDS = 55 * 60
_token_cache = {"token": None, "fetched_at": 0}


def _get_lakebase_token() -> str:
    """Return a cached or freshly-minted Postgres OAuth token for Lakebase."""
    now = time.time()
    if _token_cache["token"] and (now - _token_cache["fetched_at"]) < _TOKEN_TTL_SECONDS:
        return _token_cache["token"]

    instance_name = os.environ["LAKEBASE_INSTANCE_NAME"]
    cred = _w.database.generate_database_credential(
        instance_names=[instance_name],
        request_id=f"massive-app-{int(now)}",
    )
    _token_cache["token"] = cred.token
    _token_cache["fetched_at"] = now
    return cred.token


def _connection_params() -> dict:
    return {
        "host": os.environ["LAKEBASE_HOST"],
        "port": os.environ.get("LAKEBASE_PORT", "5432"),
        "dbname": os.environ.get("LAKEBASE_DATABASE", "databricks_postgres"),
        "user": os.environ["LAKEBASE_USER"],
        "password": _get_lakebase_token(),
        "sslmode": "require",
    }


@contextmanager
def get_connection():
    """Yield a raw psycopg2 connection with a RealDictCursor factory."""
    params = _connection_params()
    conn = psycopg2.connect(cursor_factory=RealDictCursor, **params)
    try:
        yield conn
    finally:
        conn.close()


def get_engine():
    """
    Return a SQLAlchemy engine.

    NullPool is used because Lakebase auth tokens expire; long-lived pooled
    connections would eventually fail auth. Each checkout re-resolves creds
    via creator_fn below, using a fresh token when the cache expires.
    """

    def _creator():
        params = _connection_params()
        return psycopg2.connect(**params)

    return create_engine("postgresql+psycopg2://", creator=_creator, poolclass=NullPool)


def run_query(sql: str, params: tuple | dict | None = None) -> list[dict]:
    """Run a read query against Lakebase and return rows as list[dict]."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()


def run_write(sql: str, params: tuple | dict | None = None) -> int:
    """Run an INSERT/UPDATE/DELETE against Lakebase, return affected row count."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            conn.commit()
            return cur.rowcount
