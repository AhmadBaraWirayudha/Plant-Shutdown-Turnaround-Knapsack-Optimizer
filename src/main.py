"""
main.py — End-to-end pipeline orchestrator.

Stages
------
  1. EXTRACT     : load raw work orders, asset master, failure history
  2. TRANSFORM   : clean & validate
  3. MODEL       : fit Weibull curves, compute failure probabilities
  4. RISK        : compute criticality-matrix risk scores & net values
  5. OPTIMIZE    : build & solve the 0-1 knapsack ILP (OR-Tools CP-SAT)
  6. REPORT      : export Excel (Power BI feed) + standalone HTML dashboard
  7. PERSIST     : write the run into the star-schema database (Power BI source)
  8. AUDIT       : write a JSON run-log for traceability

Run directly:
    python -m src.main
or via the CLI wrapper:
    python run_optimizer.py --budget 5000000
"""

from __future__ import annotations
import time

from src.utils.config import TA_CFG, DATA_RAW, REPORTS_DIR, DB_CFG
from src.utils.helpers import get_logger, print_banner, write_run_log
from src.utils.data_generator import generate_all

from src.etl.extract import load_work_orders, load_asset_master, load_failure_history
from src.etl.transform import run_transforms, validate_referential_integrity, enrich_with_asset_name
from src.etl.load import save_processed

from src.modeling.weibull import run_weibull_analysis
from src.modeling.risk import compute_risk_scores

from src.optimization.solver import TurnaroundSolver

from src.reporting.export import export_to_excel
from src.reporting.dashboard import generate_dashboard

from src.db.connection import get_engine
from src.db.writer import write_results_to_db

log = get_logger("main")


def run_pipeline(
    budget: float | None = None,
    turnaround_date: str | None = None,
    regenerate_data: bool = False,
    enable_db: bool | None = None,
    database_url: str | None = None,
    run_label: str | None = None,
) -> "PipelineResult":
    """
    Execute the full ETL → Weibull → Risk → ILP → Reporting → DB pipeline.

    Parameters
    ----------
    budget          : override TA_CFG.total_budget (USD)
    turnaround_date : override TA_CFG.turnaround_date (YYYY-MM-DD)
    regenerate_data : force regeneration of synthetic CMMS data
    enable_db       : override DB_CFG.enabled (None → use config/env default)
    database_url    : override DB_CFG.database_url (None → SQLite default)
    run_label       : human-readable label for this run in dim_run
                       (None → auto-generated from budget)

    Returns
    -------
    PipelineResult  bundling the SolverResult + output file paths + db run_id
    """
    t_start = time.perf_counter()
    print_banner()

    if budget is not None:
        TA_CFG.total_budget = budget
    if turnaround_date is not None:
        TA_CFG.turnaround_date = turnaround_date

    # ── Stage 0: ensure raw data exists ─────────────────────────────────────
    if regenerate_data or not (DATA_RAW / "work_orders.csv").exists():
        log.info("STAGE 0  │ Generating synthetic CMMS dataset …")
        generate_all()
    else:
        log.info("STAGE 0  │ Using existing raw data in %s", DATA_RAW)

    # ── Stage 1: EXTRACT ─────────────────────────────────────────────────────
    log.info("STAGE 1  │ EXTRACT — loading raw CMMS data")
    raw_wos = load_work_orders()
    asset_master = load_asset_master()
    raw_failures = load_failure_history()

    # ── Stage 2: TRANSFORM ───────────────────────────────────────────────────
    log.info("STAGE 2  │ TRANSFORM — cleaning & validating")
    clean_wos, clean_failures = run_transforms(raw_wos, raw_failures)
    clean_wos = validate_referential_integrity(clean_wos, asset_master)
    clean_wos = enrich_with_asset_name(clean_wos, asset_master)
    save_processed(clean_wos, "work_orders_clean")
    save_processed(clean_failures, "failure_history_clean")

    # ── Stage 3: WEIBULL MODELING ─────────────────────────────────────────────
    log.info("STAGE 3  │ MODEL — fitting Weibull reliability curves")
    wos_with_reliability = run_weibull_analysis(
        clean_wos, clean_failures, horizon_days=TA_CFG.planning_horizon_days
    )

    # ── Stage 4: RISK SCORING ─────────────────────────────────────────────────
    log.info("STAGE 4  │ RISK — computing criticality-matrix scores")
    wos_scored = compute_risk_scores(wos_with_reliability)
    save_processed(wos_scored, "work_orders_scored")

    # ── Stage 5: OPTIMIZATION ────────────────────────────────────────────────
    log.info("STAGE 5  │ OPTIMIZE — solving 0-1 knapsack ILP (OR-Tools CP-SAT)")
    solver = TurnaroundSolver(wos_scored, config=TA_CFG)
    result = solver.solve()
    save_processed(result.schedule, "optimized_schedule")

    # ── Stage 6: REPORTING ────────────────────────────────────────────────────
    log.info("STAGE 6  │ REPORT — exporting Excel + HTML dashboard")
    excel_path = export_to_excel(result)
    dashboard_path = generate_dashboard(result)

    # ── Stage 7: DATABASE PERSISTENCE (Power BI source) ────────────────────────
    db_run_id: int | None = None
    db_enabled = DB_CFG.enabled if enable_db is None else enable_db
    if db_enabled:
        log.info("STAGE 7  │ PERSIST — writing run into the star-schema database")
        engine = get_engine(database_url or DB_CFG.database_url)
        db_run_id = write_results_to_db(engine, result, TA_CFG, run_label=run_label or DB_CFG.run_label)
    else:
        log.info("STAGE 7  │ PERSIST — skipped (enable_db=False)")

    # ── Stage 8: AUDIT TRAIL ─────────────────────────────────────────────────
    elapsed = time.perf_counter() - t_start
    audit_meta = {
        "pipeline_version": "1.1.0",
        "turnaround_date": TA_CFG.turnaround_date,
        "budget_usd": TA_CFG.total_budget,
        "n_work_orders": len(wos_scored),
        "solver_summary": result.summary,
        "excel_export": str(excel_path),
        "dashboard_export": str(dashboard_path),
        "db_run_id": db_run_id,
        "total_pipeline_s": round(elapsed, 2),
    }
    log_path = write_run_log(REPORTS_DIR / "audit_logs", audit_meta)

    log.info("=" * 60)
    log.info("PIPELINE COMPLETE in %.1f s", elapsed)
    log.info("  Excel report : %s", excel_path)
    log.info("  Dashboard    : %s", dashboard_path)
    log.info("  DB run_id    : %s", db_run_id if db_run_id is not None else "(not persisted)")
    log.info("  Audit log    : %s", log_path)
    log.info("=" * 60)

    return PipelineResult(
        solver_result=result,
        excel_path=excel_path,
        dashboard_path=dashboard_path,
        audit_log_path=log_path,
        elapsed_s=elapsed,
        db_run_id=db_run_id,
    )


class PipelineResult:
    """Bundle of all pipeline outputs."""

    def __init__(self, solver_result, excel_path, dashboard_path, audit_log_path, elapsed_s, db_run_id=None):
        self.solver_result = solver_result
        self.excel_path = excel_path
        self.dashboard_path = dashboard_path
        self.audit_log_path = audit_log_path
        self.elapsed_s = elapsed_s
        self.db_run_id = db_run_id


if __name__ == "__main__":
    run_pipeline()
