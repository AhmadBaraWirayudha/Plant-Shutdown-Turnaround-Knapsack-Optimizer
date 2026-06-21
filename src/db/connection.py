"""
connection.py — Database engine factory.

Defaults to a local SQLite file (zero configuration, fully testable without
any external server) but is swappable to any SQLAlchemy-compatible engine
via the DATABASE_URL environment variable, since the schema and writer
logic in this package are pure SQLAlchemy Core/ORM and don't assume SQLite
anywhere.

    # default — local SQLite file, no setup needed
    DATABASE_URL not set → sqlite:///database/turnaround.db

    # production examples (install the matching driver first)
    DATABASE_URL=postgresql+psycopg2://user:pass@host:5432/turnaround
    DATABASE_URL=mysql+pymysql://user:pass@host:3306/turnaround
    DATABASE_URL=mssql+pyodbc://user:pass@host/turnaround?driver=ODBC+Driver+17+for+SQL+Server

Postgres and SQL Server both have native, no-driver-install Power BI
connectors — if a live, multi-user, refreshable Power BI Service deployment
is the goal rather than local prototyping, pointing DATABASE_URL at one of
those is the more production-appropriate path than SQLite (see
power_bi/README.md).
"""

from __future__ import annotations
import os

from sqlalchemy import create_engine, Engine, event

from src.utils.config import ROOT_DIR
from src.utils.helpers import get_logger

log = get_logger("db.connection")

DATABASE_DIR = ROOT_DIR / "database"
DATABASE_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_SQLITE_PATH = DATABASE_DIR / "turnaround.db"


def get_database_url(override: str | None = None) -> str:
    """Resolve the DB connection string: explicit override > env var > local SQLite default."""
    if override:
        return override
    env_url = os.getenv("DATABASE_URL")
    if env_url:
        return env_url
    return f"sqlite:///{DEFAULT_SQLITE_PATH}"


def get_engine(database_url: str | None = None, echo: bool = False) -> Engine:
    """
    Create a SQLAlchemy engine for the resolved database URL.

    For SQLite specifically, enables foreign-key enforcement (OFF by default
    in SQLite, unlike every other engine this code supports) so that the
    referential-integrity guarantees in schema.py are actually enforced
    rather than silently accepted and ignored.
    """
    url = get_database_url(database_url)
    engine = create_engine(url, echo=echo, future=True)

    if url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def _enable_sqlite_foreign_keys(dbapi_connection, _connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    log.info("Database engine ready → %s", _redact(url))
    return engine


def _redact(url: str) -> str:
    """Never log a password embedded in a connection string."""
    if "@" in url and "://" in url:
        scheme, rest = url.split("://", 1)
        if "@" in rest:
            creds, host_part = rest.split("@", 1)
            if ":" in creds:
                user = creds.split(":", 1)[0]
                return f"{scheme}://{user}:***@{host_part}"
    return url
