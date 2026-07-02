"""
extract.py — Data extraction layer.

Supports three source types:
  • CSV files  (default for this project)
  • SQLAlchemy-compatible relational databases
  • REST API endpoints (stub)

In production, swap the CSV reader for the appropriate connector.
"""

from __future__ import annotations
import pandas as pd
import requests
from pathlib import Path
from sqlalchemy import create_engine, text

from src.utils.config import DATA_RAW
from src.utils.helpers import get_logger, timed

log = get_logger("etl.extract")

# ─── CSV Extractors ───────────────────────────────────────────────────────────


@timed
def load_work_orders(path: str | Path | None = None) -> pd.DataFrame:
    """
    Read raw CMMS work-order export from CSV.

    `path` defaults to `None`, resolved against `DATA_RAW` inside the
    function body rather than baked into the signature at import time —
    see docs/METHODOLOGY.md §5 for why a literal `Path = DATA_RAW / "..."`
    default here would be the same stale-default-argument bug class found
    and fixed elsewhere in this codebase, applied here for consistency.
    """
    if path is None:
        path = DATA_RAW / "work_orders.csv"
    log.info("Extracting work orders from %s", path)
    df = pd.read_csv(
        path,
        dtype={
            "wo_id": "string",
            "asset_tag": "string",
            "asset_class": "string",
            "area": "string",
            "task_type": "string",
            "priority": "string",
            "mandatory": bool,
            "predecessor_wo_id": "string",
        },
        parse_dates=["install_date"],
        low_memory=False,
    )
    log.info("  → %d rows | %d columns", len(df), df.shape[1])
    return df


@timed
def load_asset_master(path: str | Path | None = None) -> pd.DataFrame:
    """Read asset-master table with Weibull parameters and replacement costs."""
    if path is None:
        path = DATA_RAW / "asset_master.csv"
    log.info("Extracting asset master from %s", path)
    df = pd.read_csv(path, dtype={"asset_tag": "string", "asset_class": "string"})
    log.info("  → %d assets", len(df))
    return df


@timed
def load_failure_history(path: str | Path | None = None) -> pd.DataFrame:
    """Read historical failure-time records for Weibull fitting."""
    if path is None:
        path = DATA_RAW / "failure_history.csv"
    log.info("Extracting failure history from %s", path)
    df = pd.read_csv(
        path,
        dtype={"asset_tag": "string", "asset_class": "string"},
        parse_dates=["failure_date"],
    )
    log.info("  → %d failure events", len(df))
    return df


# ─── Database Extractor (stub) ────────────────────────────────────────────────


def load_from_db(connection_string: str, query: str) -> pd.DataFrame:
    """
    Execute `query` against any SQLAlchemy-compatible database.

    Example::

        df = load_from_db(
            "mssql+pyodbc://user:pw@server/CMMS?driver=ODBC+Driver+17+for+SQL+Server",
            "SELECT * FROM dbo.WorkOrders WHERE Status = 'Planned'"
        )
    """
    engine = create_engine(connection_string)
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn)
    log.info("DB query returned %d rows", len(df))
    return df


# ─── REST API Extractor (stub) ────────────────────────────────────────────────


def load_from_api(endpoint: str, token: str | None = None) -> pd.DataFrame:
    """
    Fetch paginated JSON from a REST endpoint and flatten into a DataFrame.

    Swap in the real CMMS vendor API here (SAP PM, Maximo, etc.).

    Two hardening fixes applied here vs the naive implementation:
    - resp.json() is caught — a proxy returning an HTML error page (very
      common in enterprise CMMS environments) produces a JSONDecodeError,
      not an HTTP error code, so raise_for_status() alone misses it.
    - The endpoint URL is stripped of any query-string parameters before
      being logged, so API keys passed as `?api_key=...` are never written
      to the audit trail or log files in plaintext.
    """
    from urllib.parse import urlsplit

    # Redact query string (may contain api_key= or token= parameters)
    parsed = urlsplit(endpoint)
    safe_url = parsed._replace(query="", fragment="").geturl()

    headers = {"Authorization": f"Bearer {token}"} if token else {}
    resp = requests.get(endpoint, headers=headers, timeout=30)
    resp.raise_for_status()

    try:
        payload = resp.json()
    except Exception as exc:
        raise ValueError(
            f"API at {safe_url!r} returned non-JSON content "
            f"(Content-Type: {resp.headers.get('Content-Type', 'unknown')}). "
            "Ensure the endpoint returns application/json."
        ) from exc

    records = payload if isinstance(payload, list) else payload.get("data", payload)
    if not isinstance(records, (list, dict)):
        raise ValueError(
            f"API response from {safe_url!r} has unexpected structure: "
            f"expected a list or dict, got {type(records).__name__}."
        )

    df = pd.json_normalize(records)
    log.info("API returned %d records from %s", len(df), safe_url)
    return df
