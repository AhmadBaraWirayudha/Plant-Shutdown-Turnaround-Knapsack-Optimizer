"""
knapsack_model.py — ILP formulation for the shutdown turnaround selection problem.

Decision variable:  x_i ∈ {0, 1}   (include task i in the turnaround plan)

Objective:
    Maximise  Σ  value_i · x_i         (total net risk-adjusted value)

Constraints:
    (C1) Budget:            Σ cost_i · x_i  ≤ Budget
    (C2) Mech craft-hours:  Σ mech_i · x_i  ≤ MaxMechHours
    (C3) Elec craft-hours:  Σ elec_i · x_i  ≤ MaxElecHours
    (C4) Inst craft-hours:  Σ inst_i · x_i  ≤ MaxInstHours
    (C5) Civil craft-hours: Σ civil_i· x_i  ≤ MaxCivilHours
    (C6) Mandatory tasks:   x_i = 1          for all mandatory i
    (C7) Precedence:        x_j ≤ x_i        if task j requires task i first

CP-SAT requires integer coefficients.
Scaling strategy:
  • Costs / values → integer dollars (no scaling needed)
  • Hours          → multiply by HOUR_SCALE=10 (store tenths-of-hours)
"""

from __future__ import annotations
import pandas as pd
from ortools.sat.python import cp_model

from src.utils.config import TA_CFG, SOLVER_CFG
from src.utils.helpers import get_logger

log = get_logger("optimization.model")

HOUR_SCALE = SOLVER_CFG.hour_scale  # 10 → integers in tenths of hours


class TurnaroundModel:
    """
    Builds and exposes the CP-SAT model for the turnaround selection ILP.

    Usage::

        model_obj = TurnaroundModel(work_orders_df)
        model_obj.build()
        # Hand off to solver.py
    """

    def __init__(self, wos: pd.DataFrame, config=TA_CFG):
        self.wos = wos.reset_index(drop=True)
        self.n = len(wos)
        self.config = config
        self.model = cp_model.CpModel()
        self.x: list[cp_model.IntVar] = []

        # Integer-scaled arrays (CP-SAT requires int64)
        self._costs: list[int] = []
        self._values: list[int] = []
        self._mech: list[int] = []
        self._elec: list[int] = []
        self._inst: list[int] = []
        self._civil: list[int] = []

        # Index maps
        self._id_to_idx: dict[str, int] = {r.wo_id: i for i, r in self.wos.iterrows()}

    # ── Build ─────────────────────────────────────────────────────────────────

    def build(self) -> "TurnaroundModel":
        """Assemble the full CP-SAT model and return self."""
        self._prepare_arrays()
        self._add_variables()
        self._add_objective()
        self._add_budget_constraint()
        self._add_hour_constraints()
        self._add_mandatory_constraints()
        self._add_precedence_constraints()

        log.info(
            "ILP model built | n=%d tasks | %d variables | budget=$%s",
            self.n,
            self.n,
            f"{self.config.total_budget:,.0f}",
        )
        return self

    # ── Private helpers ────────────────────────────────────────────────────────

    def _prepare_arrays(self):
        """Convert DataFrame columns to integer arrays for CP-SAT."""
        df = self.wos
        self._costs = df["estimated_cost_usd"].round().astype(int).tolist()
        self._values = df["net_value_usd"].round().astype(int).clip(lower=0).tolist()
        self._mech = (df["mech_hours"] * HOUR_SCALE).round().astype(int).tolist()
        self._elec = (df["elec_hours"] * HOUR_SCALE).round().astype(int).tolist()
        self._inst = (df["inst_hours"] * HOUR_SCALE).round().astype(int).tolist()
        self._civil = (df["civil_hours"] * HOUR_SCALE).round().astype(int).tolist()

    def _add_variables(self):
        """Create one Boolean decision variable per task."""
        self.x = [self.model.new_bool_var(f"x_{i}_{row.wo_id}") for i, row in self.wos.iterrows()]

    def _add_objective(self):
        """Maximise total net value."""
        self.model.maximize(sum(self._values[i] * self.x[i] for i in range(self.n)))

    def _add_budget_constraint(self):
        """Total spend ≤ budget."""
        budget_int = int(self.config.total_budget)
        self.model.add(sum(self._costs[i] * self.x[i] for i in range(self.n)) <= budget_int)
        log.debug("  C1 Budget ≤ $%d", budget_int)

    def _add_hour_constraints(self):
        """Craft-hour constraints for each trade."""
        cfg = self.config
        trades = [
            ("Mechanical", self._mech, cfg.max_mech_hours),
            ("Electrical", self._elec, cfg.max_elec_hours),
            ("Instrumentation", self._inst, cfg.max_inst_hours),
            ("Civil", self._civil, cfg.max_civil_hours),
        ]
        for name, hrs, cap in trades:
            cap_int = int(cap * HOUR_SCALE)
            self.model.add(sum(hrs[i] * self.x[i] for i in range(self.n)) <= cap_int)
            log.debug("  C[%s] ≤ %d.%d h", name, cap_int // HOUR_SCALE, cap_int % HOUR_SCALE)

    def _add_mandatory_constraints(self):
        """Force all mandatory tasks to be selected."""
        n_mandatory = 0
        for i, row in self.wos.iterrows():
            if row.get("mandatory", False):
                self.model.add(self.x[i] == 1)
                n_mandatory += 1
        log.debug("  C6 Mandatory tasks forced: %d", n_mandatory)

    def _add_precedence_constraints(self):
        """
        If task j has predecessor p, then  x_j ≤ x_p
        (can't do j unless p is also selected).
        """
        n_prec = 0
        if "predecessor_wo_id" not in self.wos.columns:
            return
        for i, row in self.wos.iterrows():
            pred_id = row.get("predecessor_wo_id")
            if pd.isna(pred_id):
                continue
            pred_id = str(pred_id)
            if pred_id and pred_id in self._id_to_idx:
                p = self._id_to_idx[pred_id]
                self.model.add(self.x[i] <= self.x[p])
                n_prec += 1
        log.debug("  C7 Precedence constraints added: %d", n_prec)

    # ── Greedy warm-start hint ─────────────────────────────────────────────────

    def add_greedy_hint(self):
        """
        Provide a greedy feasible solution as a warm-start hint to CP-SAT.

        Strategy: Sort tasks by value/cost ratio (bang-per-buck) descending;
        select greedily while budget and mandatory constraints allow.
        This dramatically speeds convergence on large instances.
        """
        remaining_budget = int(self.config.total_budget)
        selected = set()

        # Force mandatory tasks first
        for i, row in self.wos.iterrows():
            if row.get("mandatory", False):
                selected.add(i)
                remaining_budget -= self._costs[i]

        # Greedy on ratio for optionals
        ratios = [(i, self._values[i] / max(self._costs[i], 1)) for i in range(self.n) if i not in selected]
        for i, _ in sorted(ratios, key=lambda t: t[1], reverse=True):
            if self._costs[i] <= remaining_budget:
                selected.add(i)
                remaining_budget -= self._costs[i]

        # Apply hints
        for i in range(self.n):
            self.model.add_hint(self.x[i], 1 if i in selected else 0)

        log.info("Greedy warm-start hint: %d tasks pre-selected", len(selected))
        return self
