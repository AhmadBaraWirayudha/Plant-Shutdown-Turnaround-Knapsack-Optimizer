"""
queries.py — Read-back helpers against the star schema.

These exist for two audiences:
  1. Python callers (tests, notebooks, ad-hoc analysis) who want a
     DataFrame back without writing raw SQL.
  2. As a readable reference for the equivalent SQL/DAX a Power BI report
     would issue — see power_bi/README.md, which mirrors several of these
     as either the `latest_run_facts` view or suggested DAX measures.
"""

from __future__ import annotations
import pandas as pd
from sqlalchemy import Engine, select

from src.db.schema import DimRun, FactWorkOrderDecision
from src.utils.helpers import get_logger

log = get_logger("db.queries")


def list_runs(engine: Engine) -> pd.DataFrame:
    """Every run/scenario stored so far, most recent first — the table
    behind a Power BI 'pick a scenario' slicer."""
    query = select(DimRun).order_by(DimRun.run_id.desc())
    return pd.read_sql(query, engine)


def latest_run_id(engine: Engine) -> int | None:
    """The run_id of the most recently written scenario, or None if the
    database is empty (e.g. on a fresh clone before the first run)."""
    with engine.connect() as conn:
        result = conn.execute(select(DimRun.run_id).order_by(DimRun.run_id.desc()).limit(1))
        row = result.first()
    return row[0] if row else None


def get_run_facts(engine: Engine, run_id: int) -> pd.DataFrame:
    """Every fact row belonging to one specific run — joined out to plain
    column names rather than raw foreign keys, for direct human/Excel use."""
    query = """
        SELECT
            f.wo_id, f.description, f.asset_tag,
            a.asset_name, a.asset_class, a.area,
            tt.task_type_name AS task_type,
            p.priority_name AS priority,
            rl.risk_level_name AS risk_level,
            f.mandatory, f.estimated_cost_usd,
            f.mech_hours, f.elec_hours, f.inst_hours, f.civil_hours,
            f.failure_prob, f.rul_days, f.risk_score,
            f.deferred_cost_usd, f.net_value_usd,
            f.selected, f.decision
        FROM fact_work_order_decision f
        JOIN dim_asset a       ON a.asset_tag = f.asset_tag
        JOIN dim_task_type tt  ON tt.task_type_id = f.task_type_id
        JOIN dim_priority p    ON p.priority_id = f.priority_id
        JOIN dim_risk_level rl ON rl.risk_level_id = f.risk_level_id
        WHERE f.run_id = :run_id
    """
    return pd.read_sql(query, engine, params={"run_id": run_id})


def compare_runs_summary(engine: Engine) -> pd.DataFrame:
    """
    One row per run with the headline KPIs — exactly the table a Power BI
    'budget sensitivity' report page would slice and chart, except backed
    by real persisted history instead of a one-off notebook sweep.
    """
    query = select(
        DimRun.run_id,
        DimRun.run_label,
        DimRun.run_timestamp,
        DimRun.budget_usd,
        DimRun.tasks_selected,
        DimRun.tasks_total,
        DimRun.budget_used_usd,
        DimRun.budget_utilisation,
        DimRun.total_net_value_usd,
        DimRun.roi_ratio,
        DimRun.total_risk_score_reduced,
    ).order_by(DimRun.run_id)
    return pd.read_sql(query, engine)


def fact_row_count(engine: Engine, run_id: int | None = None) -> int:
    """Row count in the fact table, optionally scoped to one run — used by
    the round-trip integrity tests to confirm nothing was dropped or
    duplicated on write."""
    with engine.connect() as conn:
        if run_id is not None:
            stmt = select(FactWorkOrderDecision).where(FactWorkOrderDecision.run_id == run_id)
        else:
            stmt = select(FactWorkOrderDecision)
        return len(conn.execute(stmt).fetchall())


# ─── Scenario-aware queries ───────────────────────────────────────────────────


def list_scenario_runs(engine: Engine, scenario_id: int) -> pd.DataFrame:
    """
    All dim_run rows tagged with ``scenario_id``, most recent first.

    Useful for showing a scenario's full solve history — not just the latest
    (``current_run_id``) but every iteration the planner has ever triggered.
    """
    query = (
        select(DimRun)
        .where(DimRun.scenario_id == scenario_id)
        .order_by(DimRun.run_id.desc())
    )
    return pd.read_sql(query, engine)


def get_scenario_facts(engine: Engine, scenario_id: int) -> pd.DataFrame:
    """
    Per-work-order decisions for a scenario's LATEST solve (``current_run_id``).

    Returns the same joined columns as ``get_run_facts``, so callers can use
    either interchangeably once they have a scenario_id.

    Raises ``KeyError`` if the scenario has no ``current_run_id`` (not yet solved).
    """
    from sqlalchemy.orm import Session
    from src.db.schema import DimScenario

    with Session(engine) as session:
        scenario = session.get(DimScenario, scenario_id)
        if scenario is None:
            raise KeyError(f"scenario_id={scenario_id} not found")
        if scenario.current_run_id is None:
            raise KeyError(
                f"Scenario {scenario_id} ({scenario.name!r}) has not been solved yet. "
                "Call solve_scenario() first."
            )
        run_id = scenario.current_run_id

    return get_run_facts(engine, run_id)


def scenario_kpi_history(engine: Engine, scenario_id: int) -> pd.DataFrame:
    """
    KPI time-series for a scenario — one row per solve, ordered oldest first.

    Shows how the selected task count, budget used, net value, and ROI
    evolved as the planner iterated on this scenario.
    """
    query = """
        SELECT
            r.run_id,
            r.run_timestamp,
            r.run_label,
            r.budget_usd,
            r.tasks_total,
            r.tasks_selected,
            r.budget_used_usd,
            r.budget_utilisation,
            r.total_net_value_usd,
            r.roi_ratio,
            r.total_risk_score_reduced,
            r.solver_status,
            r.solve_time_s
        FROM dim_run r
        WHERE r.scenario_id = :scenario_id
        ORDER BY r.run_id ASC
    """
    return pd.read_sql(query, engine, params={"scenario_id": scenario_id})
