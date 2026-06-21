"""
solver.py — OR-Tools CP-SAT solver wrapper.

Wraps `TurnaroundModel`, runs the solve, and returns structured solution
DataFrames that the reporting layer can directly consume.
"""

from __future__ import annotations
import time
import numpy as np
import pandas as pd
from ortools.sat.python import cp_model

from src.optimization.knapsack_model import TurnaroundModel
from src.utils.config import TA_CFG, SOLVER_CFG
from src.utils.helpers import get_logger, fmt_usd, fmt_pct

log = get_logger("optimization.solver")


# ─── Status helper ────────────────────────────────────────────────────────────

STATUS_NAMES = {
    cp_model.OPTIMAL: "OPTIMAL",
    cp_model.FEASIBLE: "FEASIBLE",
    cp_model.INFEASIBLE: "INFEASIBLE",
    cp_model.UNKNOWN: "UNKNOWN",
    cp_model.MODEL_INVALID: "MODEL_INVALID",
}


# ─── Solver ───────────────────────────────────────────────────────────────────


class TurnaroundSolver:
    """
    Orchestrates building, warm-starting, and solving the turnaround ILP.

    Quick-start::

        solver = TurnaroundSolver(work_orders_with_risk_df)
        results = solver.solve()
        print(results.solution_summary)
    """

    def __init__(self, wos: pd.DataFrame, config=TA_CFG, solver_cfg=SOLVER_CFG):
        self.wos = wos
        self.config = config
        self.solver_cfg = solver_cfg
        self.ta_model: TurnaroundModel | None = None
        self.cp_solver: cp_model.CpSolver | None = None
        self.status: int | None = None
        self._solved_at: float = 0.0

    # ── Public API ─────────────────────────────────────────────────────────────

    def solve(self) -> "SolverResult":
        """Build model, add warm start, solve, return structured result."""
        log.info("=" * 60)
        log.info("TURNAROUND ILP SOLVER  — OR-Tools CP-SAT")
        log.info(
            "  Tasks: %d | Budget: %s | Timeout: %ss",
            len(self.wos),
            fmt_usd(self.config.total_budget),
            self.solver_cfg.max_solve_seconds,
        )
        log.info("=" * 60)

        # 1. Build model
        self.ta_model = TurnaroundModel(self.wos, self.config)
        self.ta_model.build()
        self.ta_model.add_greedy_hint()

        # 2. Configure CP-SAT solver
        self.cp_solver = cp_model.CpSolver()
        self.cp_solver.parameters.max_time_in_seconds = self.solver_cfg.max_solve_seconds
        self.cp_solver.parameters.num_search_workers = self.solver_cfg.num_workers
        self.cp_solver.parameters.log_search_progress = False

        # 3. Solve
        t0 = time.perf_counter()
        self.status = self.cp_solver.solve(self.ta_model.model)
        elapsed = time.perf_counter() - t0
        self._solved_at = elapsed

        status_str = STATUS_NAMES.get(self.status, f"CODE_{self.status}")
        log.info("Solver status: %s  (%.2f s)", status_str, elapsed)

        if self.status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            if self.status == cp_model.UNKNOWN:
                # UNKNOWN means the solver ran out of time before reaching
                # ANY conclusion — it says nothing about whether a feasible
                # solution exists. Blaming "mandatory tasks exceed
                # budget/hours" here would point the user at the wrong
                # fix; the actual lever is --timeout, not --budget.
                raise RuntimeError(
                    f"CP-SAT returned UNKNOWN after {elapsed:.1f}s — the solver ran out of "
                    "time before determining whether a feasible solution even exists. "
                    "This does NOT mean your constraints are infeasible. "
                    "Try increasing --timeout (current limit: "
                    f"{self.solver_cfg.max_solve_seconds:.0f}s)."
                )
            raise RuntimeError(
                f"CP-SAT returned {status_str}. " "Check that mandatory tasks don't exceed budget/hours."
            )

        return self._extract_results()

    # ── Solution extraction ────────────────────────────────────────────────────

    def _extract_results(self) -> "SolverResult":
        model = self.ta_model
        cp = self.cp_solver
        wos = model.wos

        selected_mask = np.array([bool(cp.value(model.x[i])) for i in range(model.n)])

        df_out = wos.copy()
        df_out["selected"] = selected_mask
        df_out["decision"] = np.where(selected_mask, "INCLUDE", "DEFER")

        # ── Utilisation numbers ────────────────────────────────────────────
        sel = df_out[df_out["selected"]]

        budget_used = sel["estimated_cost_usd"].sum()
        mech_used = sel["mech_hours"].sum()
        elec_used = sel["elec_hours"].sum()
        inst_used = sel["inst_hours"].sum()
        civil_used = sel["civil_hours"].sum()
        total_value = sel["net_value_usd"].sum()

        # "risk_score" and "mandatory" are produced by the risk-scoring module
        # (src.modeling.risk), not by the core ILP itself. The solver must
        # remain solvable/testable on a bare-bones WO table (budget + hours +
        # value columns only), so we degrade gracefully when they're absent
        # rather than raising a KeyError deep in reporting plumbing.
        risk_reduced = sel["risk_score"].sum() if "risk_score" in sel.columns else 0.0
        mandatory_count = int(sel["mandatory"].sum()) if "mandatory" in sel.columns else 0

        cfg = self.config

        summary = {
            "solver_status": STATUS_NAMES[self.status],
            "solve_time_s": round(self._solved_at, 3),
            "tasks_total": len(df_out),
            "tasks_selected": int(selected_mask.sum()),
            "tasks_deferred": int((~selected_mask).sum()),
            "mandatory_selected": mandatory_count,
            "budget_usd": cfg.total_budget,
            "budget_used_usd": round(float(budget_used), 2),
            "budget_utilisation": round(float(budget_used) / cfg.total_budget, 4),
            "max_mech_hours": cfg.max_mech_hours,
            "mech_hours_used": round(float(mech_used), 1),
            "mech_utilisation": round(float(mech_used) / cfg.max_mech_hours, 4),
            "max_elec_hours": cfg.max_elec_hours,
            "elec_hours_used": round(float(elec_used), 1),
            "elec_utilisation": round(float(elec_used) / cfg.max_elec_hours, 4),
            "max_inst_hours": cfg.max_inst_hours,
            "inst_hours_used": round(float(inst_used), 1),
            "inst_utilisation": round(float(inst_used) / cfg.max_inst_hours, 4),
            "max_civil_hours": cfg.max_civil_hours,
            "civil_hours_used": round(float(civil_used), 1),
            "civil_utilisation": round(float(civil_used) / cfg.max_civil_hours, 4),
            "total_net_value_usd": round(float(total_value), 2),
            "roi_ratio": round(float(total_value) / max(float(budget_used), 1), 4),
            "total_risk_score_reduced": int(risk_reduced),
            "objective_value": cp.objective_value,
        }

        self._log_summary(summary)
        return SolverResult(schedule=df_out, summary=summary)

    @staticmethod
    def _log_summary(s: dict):
        log.info("─" * 60)
        log.info("  SOLUTION SUMMARY")
        log.info(
            "  Tasks selected  : %d / %d  (%.1f %%)",
            s["tasks_selected"],
            s["tasks_total"],
            s["tasks_selected"] / s["tasks_total"] * 100,
        )
        log.info(
            "  Budget used     : %s  (%s)",
            fmt_usd(s["budget_used_usd"]),
            fmt_pct(s["budget_utilisation"]),
        )
        log.info(
            "  Mech hours      : %s h  (%s of capacity)",
            f"{s['mech_hours_used']:,.0f}",
            fmt_pct(s["mech_utilisation"]),
        )
        log.info(
            "  Elec hours      : %s h  (%s of capacity)",
            f"{s['elec_hours_used']:,.0f}",
            fmt_pct(s["elec_utilisation"]),
        )
        log.info(
            "  Inst hours      : %s h  (%s of capacity)",
            f"{s['inst_hours_used']:,.0f}",
            fmt_pct(s["inst_utilisation"]),
        )
        log.info(
            "  Net value       : %s  (ROI %.1f×)",
            fmt_usd(s["total_net_value_usd"]),
            s["roi_ratio"],
        )
        log.info("  Risk score Δ    : %d units", s["total_risk_score_reduced"])
        log.info("─" * 60)


# ─── Result container ─────────────────────────────────────────────────────────


class SolverResult:
    """Immutable result bundle returned by TurnaroundSolver.solve()."""

    def __init__(self, schedule: pd.DataFrame, summary: dict):
        self.schedule = schedule
        self.summary = summary
        self.selected_schedule = schedule[schedule["selected"]].copy()
        self.deferred_schedule = schedule[~schedule["selected"]].copy()

    def __repr__(self) -> str:
        return (
            f"SolverResult("
            f"selected={self.summary['tasks_selected']}, "
            f"budget_used=${self.summary['budget_used_usd']:,.0f}, "
            f"roi={self.summary['roi_ratio']:.2f}x)"
        )
