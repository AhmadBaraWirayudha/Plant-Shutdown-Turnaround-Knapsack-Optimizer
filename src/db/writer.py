"""
writer.py — Persist a SolverResult into the star schema.

Each call to `write_results_to_db()` represents one optimizer run/scenario:
  1. Ensure all tables exist (idempotent).
  2. Upsert the small lookup dimensions (task type, priority, risk level) —
     insert any new distinct values, never duplicate existing ones.
  3. Upsert dim_asset — insert any asset_tag not already known.
  4. Insert ONE new dim_run row describing this run's configuration and
     summary KPIs, and get back its auto-generated run_id.
  5. Bulk-insert one fact row per work order, tagged with that run_id.
  6. (Re)create the `latest_run_facts` SQL VIEW so a default Power BI import
     against that view always shows the most recent scenario without any
     DAX filtering required.

Every write is wrapped in a single transaction — either the whole run lands
in the database or none of it does, so a Power BI report can never observe
a half-written scenario.
"""

from __future__ import annotations
import pandas as pd
from sqlalchemy import Engine, select, text
from sqlalchemy.orm import Session

from src.db.schema import (
    Base,
    DimRun,
    DimAsset,
    DimTaskType,
    DimPriority,
    DimRiskLevel,
    FactWorkOrderDecision,
)
from src.optimization.solver import SolverResult
from src.utils.helpers import get_logger, timed

log = get_logger("db.writer")

RISK_LEVEL_SORT_ORDER = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}


def init_db(engine: Engine) -> None:
    """Create every table in the star schema if it doesn't already exist."""
    Base.metadata.create_all(engine)
    log.info("Schema ready: %s", ", ".join(Base.metadata.tables.keys()))


def _upsert_lookup_task_types(session: Session, schedule: pd.DataFrame) -> dict[str, int]:
    existing = {row.task_type_name: row.task_type_id for row in session.scalars(select(DimTaskType))}
    for name in schedule["task_type"].dropna().unique():
        if name not in existing:
            row = DimTaskType(task_type_name=name)
            session.add(row)
            session.flush()  # populate autoincrement id immediately
            existing[name] = row.task_type_id
    return existing


def _upsert_lookup_priorities(session: Session, schedule: pd.DataFrame) -> dict[str, int]:
    weight_map = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}
    existing = {row.priority_name: row.priority_id for row in session.scalars(select(DimPriority))}
    for name in schedule["priority"].dropna().unique():
        if name not in existing:
            row = DimPriority(priority_name=name, priority_weight=weight_map.get(name, 1))
            session.add(row)
            session.flush()
            existing[name] = row.priority_id
    return existing


def _upsert_lookup_risk_levels(session: Session, schedule: pd.DataFrame) -> dict[str, int]:
    existing = {row.risk_level_name: row.risk_level_id for row in session.scalars(select(DimRiskLevel))}
    for name in schedule["risk_level"].dropna().unique():
        if name not in existing:
            row = DimRiskLevel(
                risk_level_name=name,
                sort_order=RISK_LEVEL_SORT_ORDER.get(name, 0),
            )
            session.add(row)
            session.flush()
            existing[name] = row.risk_level_id
    return existing


def _upsert_dim_asset(session: Session, schedule: pd.DataFrame) -> None:
    """Insert any asset_tag not already present. Existing assets are left
    untouched — dim_asset holds stable identity attributes, not per-run
    measures, so there's nothing to update on a repeat sighting."""
    # NOTE: select(DimAsset.asset_tag) selects a single COLUMN, so
    # session.scalars() already unwraps each row to the bare string value.
    # (Contrast with the lookup-table upserts above, which select the full
    # ORM entity via select(DimTaskType) etc., where scalars() yields model
    # instances with attributes — a real bug here on the first pass, caught
    # only once a second run gave dim_asset existing rows to iterate over.)
    known_tags = set(session.scalars(select(DimAsset.asset_tag)))
    asset_cols = schedule.drop_duplicates(subset=["asset_tag"])

    new_count = 0
    for _, row in asset_cols.iterrows():
        if row.asset_tag in known_tags:
            continue
        session.add(
            DimAsset(
                asset_tag=row.asset_tag,
                asset_class=row.asset_class,
                asset_name=row.get("asset_name", row.asset_class),
                area=row.area,
                install_date=str(row.get("install_date", "")),
                replace_usd=float(row.replace_usd),
                c_safety=int(row.c_safety),
                c_env=int(row.c_env),
                c_prod=int(row.c_prod),
                c_cost=int(row.c_cost),
            )
        )
        known_tags.add(row.asset_tag)
        new_count += 1
    if new_count:
        session.flush()
    log.info("dim_asset: %d new assets inserted (%d total known)", new_count, len(known_tags))


def _insert_dim_run(session: Session, ta_cfg, summary: dict, run_label: str | None) -> int:
    run = DimRun(
        run_label=run_label or f"Budget ${summary['budget_usd']:,.0f}",
        turnaround_date=ta_cfg.turnaround_date,
        budget_usd=summary["budget_usd"],
        max_mech_hours=summary["max_mech_hours"],
        max_elec_hours=summary["max_elec_hours"],
        max_inst_hours=summary["max_inst_hours"],
        max_civil_hours=summary["max_civil_hours"],
        planning_horizon_days=ta_cfg.planning_horizon_days,
        solver_status=summary["solver_status"],
        solve_time_s=summary["solve_time_s"],
        tasks_total=summary["tasks_total"],
        tasks_selected=summary["tasks_selected"],
        budget_used_usd=summary["budget_used_usd"],
        budget_utilisation=summary["budget_utilisation"],
        total_net_value_usd=summary["total_net_value_usd"],
        roi_ratio=summary["roi_ratio"],
        total_risk_score_reduced=summary["total_risk_score_reduced"],
    )
    session.add(run)
    session.flush()
    return run.run_id


def _insert_fact_rows(
    session: Session,
    schedule: pd.DataFrame,
    run_id: int,
    task_type_map: dict[str, int],
    priority_map: dict[str, int],
    risk_level_map: dict[str, int],
) -> int:
    facts = []
    for _, r in schedule.iterrows():
        facts.append(
            FactWorkOrderDecision(
                run_id=run_id,
                asset_tag=r.asset_tag,
                task_type_id=task_type_map[r.task_type],
                priority_id=priority_map[r.priority],
                risk_level_id=risk_level_map[r.risk_level],
                wo_id=r.wo_id,
                description=str(r.description)[:200],
                predecessor_wo_id=(None if pd.isna(r.get("predecessor_wo_id")) else str(r.predecessor_wo_id)),
                mandatory=bool(r.mandatory),
                age_days=float(r.age_days),
                estimated_cost_usd=float(r.estimated_cost_usd),
                mech_hours=float(r.mech_hours),
                elec_hours=float(r.elec_hours),
                inst_hours=float(r.inst_hours),
                civil_hours=float(r.civil_hours),
                total_craft_hours=float(r.total_craft_hours),
                duration_days=int(r.duration_days),
                fitted_beta=float(r.fitted_beta),
                fitted_eta=float(r.fitted_eta),
                failure_prob=float(r.failure_prob),
                rul_days=float(r.rul_days),
                consequence_score=float(r.consequence_score),
                likelihood_tier=int(r.likelihood_tier),
                consequence_tier=int(r.consequence_tier),
                risk_score=int(r.risk_score),
                deferred_cost_usd=float(r.deferred_cost_usd),
                net_value_usd=float(r.net_value_usd),
                selected=bool(r.selected),
                decision=str(r.decision),
            )
        )
    session.add_all(facts)
    session.flush()
    return len(facts)


def _create_latest_run_view(engine: Engine) -> None:
    """
    (Re)create a `latest_run_facts` SQL VIEW so the simplest possible Power
    BI import — "give me the current schedule" — needs no DAX filtering at
    all. The full `fact_work_order_decision` table remains available
    separately for cross-run / scenario-comparison reports.
    """
    with engine.begin() as conn:
        conn.execute(text("DROP VIEW IF EXISTS latest_run_facts"))
        conn.execute(text("""
                CREATE VIEW latest_run_facts AS
                SELECT f.*
                FROM fact_work_order_decision f
                WHERE f.run_id = (SELECT MAX(run_id) FROM dim_run)
                """))
    log.info("View ready: latest_run_facts")


@timed
def write_results_to_db(
    engine: Engine,
    result: SolverResult,
    ta_cfg,
    run_label: str | None = None,
) -> int:
    """
    Persist one optimizer run into the star schema inside a single
    transaction. Returns the new run_id.
    """
    init_db(engine)
    schedule = result.schedule

    with Session(engine) as session:
        try:
            task_type_map = _upsert_lookup_task_types(session, schedule)
            priority_map = _upsert_lookup_priorities(session, schedule)
            risk_level_map = _upsert_lookup_risk_levels(session, schedule)
            _upsert_dim_asset(session, schedule)

            run_id = _insert_dim_run(session, ta_cfg, result.summary, run_label)
            n_facts = _insert_fact_rows(
                session, schedule, run_id, task_type_map, priority_map, risk_level_map
            )

            session.commit()
        except Exception:
            session.rollback()
            log.exception("Database write FAILED — transaction rolled back, nothing was persisted")
            raise

    _create_latest_run_view(engine)

    log.info(
        "✅  DB write complete | run_id=%d | %d fact rows | label=%r",
        run_id,
        n_facts,
        run_label or "(auto)",
    )
    return run_id
