"""
export.py — Generate Excel workbook consumed by the Power BI dashboard.

Flat sheets (human-browsable, one wide table per view):
  1. OptimizedSchedule  — full WO list with selection flag & risk data
  2. Selected           — only the scheduled tasks
  3. Deferred           — tasks not selected (deferred risk register)
  4. SummaryKPIs        — scalar KPIs (one row, wide format) for card visuals
  5. CapacityUtilization — craft-hour utilisation by trade
  6. RiskMatrix         — 5×5 criticality matrix pivot
  7. ByArea             — spend & hours aggregated per plant area
  8. ByEquipmentClass   — spend & risk aggregated per equipment class

Star-schema sheets (for Power BI users who want relationship-ready tables
without setting up the database — see power_bi/README.md Option C):
  9.  Dim_Asset         — one row per asset_tag, mirrors src/db/schema.py
  10. Dim_TaskType       — distinct task types
  11. Dim_Priority       — distinct priority levels
  12. Dim_RiskLevel      — distinct risk levels, with sort order
  13. FactWorkOrderDecision — same grain as the database fact table, but
      scoped to this single run only (the database accumulates history
      across runs; this sheet reflects only "right now")

These mirror src/db/schema.py exactly so the relationship-building steps
in power_bi/README.md work identically whether the data source is this
workbook or the live database.

DAX measures to build in Power BI — see power_bi/measures.dax for the full,
copy-pasteable set built against the database/star-schema sheets above.
"""

from __future__ import annotations
import pandas as pd
from pathlib import Path

from src.optimization.solver import SolverResult
from src.modeling.risk import build_criticality_matrix
from src.utils.config import REPORTS_DIR
from src.utils.helpers import get_logger, timed

log = get_logger("reporting.export")

RISK_LEVEL_SORT_ORDER = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}


def _build_dimension_sheets(sched: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Build star-schema-shaped tables (Dim_Asset, Dim_TaskType, Dim_Priority,
    Dim_RiskLevel, FactWorkOrderDecision) from the flat schedule DataFrame,
    mirroring src/db/schema.py exactly. Lets an Excel-only Power BI user
    build the same relationship model described in power_bi/README.md
    without ever touching the database.
    """
    asset_cols = [
        "asset_tag",
        "asset_class",
        "asset_name",
        "area",
        "replace_usd",
        "c_safety",
        "c_env",
        "c_prod",
        "c_cost",
    ]
    present_asset_cols = [c for c in asset_cols if c in sched.columns]
    dim_asset = sched[present_asset_cols].drop_duplicates(subset=["asset_tag"]).reset_index(drop=True)

    dim_task_type = (
        pd.DataFrame({"task_type_name": sorted(sched["task_type"].dropna().unique())})
        .reset_index()
        .rename(columns={"index": "task_type_id"})
    )
    dim_task_type["task_type_id"] += 1

    priority_weights = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}
    priorities = sorted(sched["priority"].dropna().unique(), key=lambda p: -priority_weights.get(p, 0))
    dim_priority = pd.DataFrame(
        {
            "priority_id": range(1, len(priorities) + 1),
            "priority_name": priorities,
            "priority_weight": [priority_weights.get(p, 0) for p in priorities],
        }
    )

    risk_levels = sorted(sched["risk_level"].dropna().unique(), key=lambda r: RISK_LEVEL_SORT_ORDER.get(r, 0))
    dim_risk_level = pd.DataFrame(
        {
            "risk_level_id": range(1, len(risk_levels) + 1),
            "risk_level_name": risk_levels,
            "sort_order": [RISK_LEVEL_SORT_ORDER.get(r, 0) for r in risk_levels],
        }
    )

    fact = sched.merge(
        dim_task_type[["task_type_name", "task_type_id"]],
        left_on="task_type",
        right_on="task_type_name",
        how="left",
    )
    fact = fact.merge(
        dim_priority[["priority_name", "priority_id"]],
        left_on="priority",
        right_on="priority_name",
        how="left",
    )
    fact = fact.merge(
        dim_risk_level[["risk_level_name", "risk_level_id"]],
        left_on="risk_level",
        right_on="risk_level_name",
        how="left",
    )

    fact_cols = [
        "wo_id",
        "description",
        "asset_tag",
        "task_type_id",
        "priority_id",
        "risk_level_id",
        "predecessor_wo_id",
        "mandatory",
        "age_days",
        "estimated_cost_usd",
        "mech_hours",
        "elec_hours",
        "inst_hours",
        "civil_hours",
        "total_craft_hours",
        "duration_days",
        "fitted_beta",
        "fitted_eta",
        "failure_prob",
        "rul_days",
        "consequence_score",
        "likelihood_tier",
        "consequence_tier",
        "risk_score",
        "deferred_cost_usd",
        "net_value_usd",
        "selected",
        "decision",
    ]
    present_fact_cols = [c for c in fact_cols if c in fact.columns]

    return {
        "Dim_Asset": dim_asset,
        "Dim_TaskType": dim_task_type,
        "Dim_Priority": dim_priority,
        "Dim_RiskLevel": dim_risk_level,
        "FactWorkOrderDecision": fact[present_fact_cols],
    }


@timed
def export_to_excel(result: SolverResult, out_path: Path = REPORTS_DIR / "power_bi_export.xlsx") -> Path:
    """Write multi-sheet Excel workbook; return the output path."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sched = result.schedule
    sel = result.selected_schedule
    deferred = result.deferred_schedule
    kpis = result.summary

    # ── Core columns to keep ──────────────────────────────────────────────
    KEEP_COLS = [
        "wo_id",
        "description",
        "asset_tag",
        "asset_class",
        "area",
        "task_type",
        "priority",
        "mandatory",
        "selected",
        "decision",
        "estimated_cost_usd",
        "mech_hours",
        "elec_hours",
        "inst_hours",
        "civil_hours",
        "total_craft_hours",
        "duration_days",
        "failure_prob",
        "rul_days",
        "consequence_score",
        "likelihood_tier",
        "consequence_tier",
        "risk_score",
        "risk_level",
        "deferred_cost_usd",
        "net_value_usd",
        "predecessor_wo_id",
    ]
    keep = [c for c in KEEP_COLS if c in sched.columns]

    # ── KPI scalar row ────────────────────────────────────────────────────
    kpi_row = {
        "Total WOs": kpis["tasks_total"],
        "WOs Selected": kpis["tasks_selected"],
        "WOs Deferred": kpis["tasks_deferred"],
        "Budget ($)": kpis["budget_usd"],
        "Budget Used ($)": kpis["budget_used_usd"],
        "Budget Utilisation (%)": round(kpis["budget_utilisation"] * 100, 1),
        "Mech Hours Capacity": kpis["max_mech_hours"],
        "Mech Hours Used": kpis["mech_hours_used"],
        "Mech Utilisation (%)": round(kpis["mech_utilisation"] * 100, 1),
        "Elec Hours Capacity": kpis["max_elec_hours"],
        "Elec Hours Used": kpis["elec_hours_used"],
        "Elec Utilisation (%)": round(kpis["elec_utilisation"] * 100, 1),
        "Inst Hours Capacity": kpis["max_inst_hours"],
        "Inst Hours Used": kpis["inst_hours_used"],
        "Inst Utilisation (%)": round(kpis["inst_utilisation"] * 100, 1),
        "Net Value ($)": kpis["total_net_value_usd"],
        "ROI (×)": round(kpis["roi_ratio"], 2),
        "Risk Score Reduced": kpis["total_risk_score_reduced"],
        "Solver Status": kpis["solver_status"],
        "Solve Time (s)": kpis["solve_time_s"],
    }
    df_kpi = pd.DataFrame([kpi_row])

    # ── Capacity utilisation table ────────────────────────────────────────
    trades = ["Mechanical", "Electrical", "Instrumentation", "Civil"]
    caps = [
        kpis["max_mech_hours"],
        kpis["max_elec_hours"],
        kpis["max_inst_hours"],
        kpis["max_civil_hours"],
    ]
    used = [
        kpis["mech_hours_used"],
        kpis["elec_hours_used"],
        kpis["inst_hours_used"],
        kpis["civil_hours_used"],
    ]
    df_cap = pd.DataFrame(
        {
            "Trade": trades,
            "Capacity (h)": caps,
            "Used (h)": used,
            "Available (h)": [c - u for c, u in zip(caps, used)],
            "Utilisation (%)": [round(u / c * 100, 1) for u, c in zip(used, caps)],
        }
    )

    # ── By area ───────────────────────────────────────────────────────────
    df_area = (
        sel.groupby("area", observed=True)
        .agg(
            tasks=("wo_id", "count"),
            total_cost=("estimated_cost_usd", "sum"),
            total_mech=("mech_hours", "sum"),
            avg_risk_score=("risk_score", "mean"),
            risk_score_total=("risk_score", "sum"),
        )
        .reset_index()
        .round(2)
    )

    # ── By equipment class ────────────────────────────────────────────────
    df_eqp = (
        sel.groupby("asset_class", observed=True)
        .agg(
            tasks=("wo_id", "count"),
            total_cost=("estimated_cost_usd", "sum"),
            total_value=("net_value_usd", "sum"),
            avg_failure_prob=("failure_prob", "mean"),
            avg_risk_score=("risk_score", "mean"),
        )
        .reset_index()
        .round(3)
    )
    df_eqp["roi"] = (df_eqp["total_value"] / df_eqp["total_cost"].replace(0, 1)).round(2)

    # ── Criticality matrix ────────────────────────────────────────────────
    df_crit = build_criticality_matrix(sched)

    # ── Star-schema mirror sheets (for Excel-only Power BI users) ──────────
    dim_sheets = _build_dimension_sheets(sched)

    # ── Write workbook ────────────────────────────────────────────────────
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        sched[keep].to_excel(writer, sheet_name="OptimizedSchedule", index=False)
        sel[keep].to_excel(writer, sheet_name="Selected", index=False)
        deferred[keep].to_excel(writer, sheet_name="Deferred", index=False)
        df_kpi.to_excel(writer, sheet_name="SummaryKPIs", index=False)
        df_cap.to_excel(writer, sheet_name="CapacityUtilization", index=False)
        df_crit.to_excel(writer, sheet_name="RiskMatrix")
        df_area.to_excel(writer, sheet_name="ByArea", index=False)
        df_eqp.to_excel(writer, sheet_name="ByEquipmentClass", index=False)
        for sheet_name, df in dim_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    log.info("✅  Excel export saved → %s", out_path)
    return out_path
