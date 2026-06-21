"""
config.py — Central configuration for the Turnaround Optimizer.

All tunable parameters live here so nothing is scattered across the codebase.
Override any value via environment variables (loaded via python-dotenv).
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT_DIR / "data" / "raw"
DATA_PROC = ROOT_DIR / "data" / "processed"
DATA_EXT = ROOT_DIR / "data" / "external"
REPORTS_DIR = ROOT_DIR / "reports"
DASHBOARD_DIR = ROOT_DIR / "dashboard"

for _p in (DATA_RAW, DATA_PROC, DATA_EXT, REPORTS_DIR, DASHBOARD_DIR):
    _p.mkdir(parents=True, exist_ok=True)


# ─── Turnaround Parameters ────────────────────────────────────────────────────
@dataclass
class TurnaroundConfig:
    """Operational parameters for a single turnaround event."""

    # Turnaround window
    turnaround_date: str = os.getenv("TA_DATE", "2026-10-01")
    turnaround_days: int = int(os.getenv("TA_DAYS", "30"))

    # Budget in USD
    total_budget: float = float(os.getenv("TA_BUDGET", "5_000_000"))

    # Craft-hour capacity (union contracts, crew size)
    max_mech_hours: float = float(os.getenv("TA_MECH_HRS", "15_000"))
    max_elec_hours: float = float(os.getenv("TA_ELEC_HRS", "8_000"))
    max_inst_hours: float = float(os.getenv("TA_INST_HRS", "6_000"))
    max_civil_hours: float = float(os.getenv("TA_CIVIL_HRS", "2_500"))

    # Planning horizon for failure-probability estimate (days)
    planning_horizon_days: int = int(os.getenv("TA_HORIZON", "365"))


# ─── Optimizer Parameters ─────────────────────────────────────────────────────
@dataclass
class SolverConfig:
    """OR-Tools CP-SAT tuning knobs."""

    max_solve_seconds: float = float(os.getenv("SOLVER_TIMEOUT_S", "120"))
    num_workers: int = int(os.getenv("SOLVER_WORKERS", "4"))
    # Scaling factor: dollars → integer cents for CP-SAT (no floats allowed)
    cost_scale: int = 1  # keep values as whole dollars
    hour_scale: int = 10  # store tenths-of-hours as integers
    value_scale: int = 100  # store value in cents


# ─── Synthetic Data Parameters ────────────────────────────────────────────────
@dataclass
class DataGenConfig:
    """Controls the synthetic CMMS dataset generation."""

    random_seed: int = 42
    num_work_orders: int = 550
    mandatory_fraction: float = 0.09  # ~50 mandatory tasks
    predecessor_fraction: float = 0.10  # ~10 % tasks have a predecessor


# ─── Weibull / Risk Parameters ───────────────────────────────────────────────
@dataclass
class RiskConfig:
    """Weights for multi-attribute consequence scoring."""

    w_safety: float = 0.40
    w_environmental: float = 0.25
    w_production: float = 0.25
    w_cost: float = 0.10

    # Deferral-cost multiplier: how much of asset replacement value is at risk
    # per unit of risk-score when a task is skipped.
    deferral_cost_factor: float = 0.15


# ─── Database / Power BI Integration Parameters ──────────────────────────────
@dataclass
class DatabaseConfig:
    """
    Controls whether and where optimizer results are persisted to a
    relational database for Power BI consumption.

    `database_url` follows standard SQLAlchemy connection-string syntax.
    Leaving it unset resolves to a local SQLite file at
    `database/turnaround.db` (see src/db/connection.py) — zero setup,
    works out of the box. Point it at Postgres/MySQL/SQL Server for a
    production, multi-user, Power-BI-Service-refreshable deployment.
    """

    enabled: bool = os.getenv("ENABLE_DB_EXPORT", "true").lower() in ("1", "true", "yes")
    database_url: str | None = os.getenv("DATABASE_URL")  # None → local SQLite default
    run_label: str | None = os.getenv("RUN_LABEL")  # None → auto-generated from budget


# ─── Global singleton instances ───────────────────────────────────────────────
TA_CFG = TurnaroundConfig()
SOLVER_CFG = SolverConfig()
DGEN_CFG = DataGenConfig()
RISK_CFG = RiskConfig()
DB_CFG = DatabaseConfig()
