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
from pathlib import Path

from src.utils.config import DATA_RAW
from src.utils.helpers import get_logger, timed

log = get_logger("etl.extract")

# ─── CSV Extractors ───────────────────────────────────────────────────────────


@timed
def load_work_orders(path: Path = DATA_RAW / "work_orders.csv") -> pd.DataFrame:
    """Read raw CMMS work-order export from CSV."""
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
def load_asset_master(path: Path = DATA_RAW / "asset_master.csv") -> pd.DataFrame:
    """Read asset-master table with Weibull parameters and replacement costs."""
    log.info("Extracting asset master from %s", path)
    df = pd.read_csv(path, dtype={"asset_tag": "string", "asset_class": "string"})
    log.info("  → %d assets", len(df))
    return df


@timed
def load_failure_history(path: Path = DATA_RAW / "failure_history.csv") -> pd.DataFrame:
    """Read historical failure-time records for Weibull fitting."""
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
    try:
        from sqlalchemy import create_engine, text

        engine = create_engine(connection_string)
        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn)
        log.info("DB query returned %d rows", len(df))
        return df
    except ImportError:
        raise RuntimeError("sqlalchemy not installed — run: pip install sqlalchemy")


# ─── REST API Extractor (stub) ────────────────────────────────────────────────


def load_from_api(endpoint: str, token: str | None = None) -> pd.DataFrame:
    """
    Fetch paginated JSON from a REST endpoint and flatten into a DataFrame.

    Swap in the real CMMS vendor API here (SAP PM, Maximo, etc.).
    """
    import requests

    headers = {"Authorization": f"Bearer {token}"} if token else {}
    resp = requests.get(endpoint, headers=headers, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    records = payload if isinstance(payload, list) else payload.get("data", payload)
    df = pd.json_normalize(records)
    log.info("API returned %d records from %s", len(df), endpoint)
    return df
