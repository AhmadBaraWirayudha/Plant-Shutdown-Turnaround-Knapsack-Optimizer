#!/usr/bin/env python3
"""
run_optimizer.py — Command-line entry point for the Turnaround Knapsack Optimizer.

Examples
--------
  # Run with all defaults (5,000 synthetic WOs, $5M budget)
  python run_optimizer.py

  # Override budget and turnaround date
  python run_optimizer.py --budget 3500000 --turnaround-date 2027-03-15

  # Force regeneration of synthetic CMMS data
  python run_optimizer.py --regenerate-data

  # Override craft-hour caps
  python run_optimizer.py --mech-hours 12000 --elec-hours 6000 --inst-hours 5000 --civil-hours 2000

  # Tighten solver timeout for CI / smoke tests
  python run_optimizer.py --timeout 15

  # Point at a production Postgres database instead of the local SQLite default
  python run_optimizer.py --database-url postgresql+psycopg2://user:pass@host:5432/turnaround

  # Skip the database entirely (Excel + dashboard only)
  python run_optimizer.py --no-db

  # Label this run for easy identification in Power BI's scenario slicer
  python run_optimizer.py --budget 3500000 --run-label "Q1 2027 reduced-scope scenario"
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

# Ensure project root is importable when run as a script (not as `-m src.main`)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils.config import TA_CFG, SOLVER_CFG  # noqa: E402
from src.main import run_pipeline  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_optimizer.py",
        description="Plant Shutdown Turnaround Knapsack Optimizer — "
        "0-1 ILP work-order scheduler using OR-Tools CP-SAT.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    g_scope = p.add_argument_group("Turnaround scope")
    g_scope.add_argument(
        "--budget",
        type=float,
        default=None,
        help=f"Total turnaround budget in USD (default: {TA_CFG.total_budget:,.0f})",
    )
    g_scope.add_argument(
        "--turnaround-date",
        type=str,
        default=None,
        help=f"Turnaround start date YYYY-MM-DD (default: {TA_CFG.turnaround_date})",
    )
    g_scope.add_argument(
        "--horizon-days",
        type=int,
        default=None,
        help=f"Planning horizon for failure probability, in days "
        f"(default: {TA_CFG.planning_horizon_days})",
    )

    g_hours = p.add_argument_group("Craft-hour capacity")
    g_hours.add_argument("--mech-hours", type=float, default=None, help="Mechanical craft-hour cap")
    g_hours.add_argument("--elec-hours", type=float, default=None, help="Electrical craft-hour cap")
    g_hours.add_argument("--inst-hours", type=float, default=None, help="Instrumentation craft-hour cap")
    g_hours.add_argument("--civil-hours", type=float, default=None, help="Civil craft-hour cap")

    g_solver = p.add_argument_group("Solver tuning")
    g_solver.add_argument(
        "--timeout",
        type=float,
        default=None,
        help=f"Max CP-SAT solve time in seconds (default: {SOLVER_CFG.max_solve_seconds})",
    )
    g_solver.add_argument(
        "--workers",
        type=int,
        default=None,
        help=f"Parallel search workers for CP-SAT (default: {SOLVER_CFG.num_workers})",
    )

    g_data = p.add_argument_group("Data")
    g_data.add_argument(
        "--regenerate-data",
        action="store_true",
        help="Force regeneration of the synthetic CMMS dataset even if cached CSVs exist",
    )
    g_data.add_argument(
        "--num-work-orders",
        type=int,
        default=None,
        help="Number of synthetic work orders to generate (only with --regenerate-data)",
    )
    g_data.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for synthetic data generation (reproducibility)",
    )

    g_db = p.add_argument_group("Database / Power BI")
    g_db.add_argument(
        "--database-url",
        type=str,
        default=None,
        help="SQLAlchemy connection string (default: local SQLite at database/turnaround.db). "
        "Examples: postgresql+psycopg2://user:pass@host:5432/turnaround, "
        "mysql+pymysql://user:pass@host/turnaround",
    )
    g_db.add_argument(
        "--run-label",
        type=str,
        default=None,
        help="Human-readable label for this run in the database (e.g. 'Q4 2026 turnaround'). "
        "Default: auto-generated from the budget.",
    )
    g_db.add_argument(
        "--no-db",
        action="store_true",
        help="Skip database persistence entirely — only produce the Excel + HTML dashboard outputs",
    )

    g_out = p.add_argument_group("Output")
    g_out.add_argument("--quiet", action="store_true", help="Suppress the ASCII banner")

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    # Apply CLI overrides to the global config singletons
    if args.mech_hours is not None:
        TA_CFG.max_mech_hours = args.mech_hours
    if args.elec_hours is not None:
        TA_CFG.max_elec_hours = args.elec_hours
    if args.inst_hours is not None:
        TA_CFG.max_inst_hours = args.inst_hours
    if args.civil_hours is not None:
        TA_CFG.max_civil_hours = args.civil_hours
    if args.horizon_days is not None:
        TA_CFG.planning_horizon_days = args.horizon_days

    if args.timeout is not None:
        SOLVER_CFG.max_solve_seconds = args.timeout
    if args.workers is not None:
        SOLVER_CFG.num_workers = args.workers

    if args.num_work_orders is not None:
        from src.utils.config import DGEN_CFG

        DGEN_CFG.num_work_orders = args.num_work_orders
    if args.seed is not None:
        from src.utils.config import DGEN_CFG

        DGEN_CFG.random_seed = args.seed

    try:
        result = run_pipeline(
            budget=args.budget,
            turnaround_date=args.turnaround_date,
            regenerate_data=args.regenerate_data,
            enable_db=not args.no_db,
            database_url=args.database_url,
            run_label=args.run_label,
        )
    except (RuntimeError, ValueError) as exc:
        print(f"\n❌  PIPELINE FAILED: {exc}\n", file=sys.stderr)
        return 1

    print(f"\n✅  Done. Open the dashboard:\n    {result.dashboard_path}\n")
    print(f"    Excel (Power BI feed):\n    {result.excel_path}\n")
    if result.db_run_id is not None:
        print(f"    Database run_id {result.db_run_id} written — see power_bi/README.md to connect\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
