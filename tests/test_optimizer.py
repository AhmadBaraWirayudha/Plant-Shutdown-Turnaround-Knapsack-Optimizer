"""
test_optimizer.py — Constraint-satisfaction tests for the 0-1 knapsack ILP.

These are the most important tests in the suite: they prove the solver
NEVER violates budget, craft-hour, mandatory, or precedence constraints,
regardless of problem size or instance. A scheduling tool that quietly
overspends or skips a mandatory safety task is worse than no tool at all.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from dataclasses import dataclass

from src.optimization.knapsack_model import TurnaroundModel, HOUR_SCALE
from src.optimization.solver import TurnaroundSolver


@dataclass
class _Cfg:
    """Lightweight stand-in for TurnaroundConfig in tests."""

    total_budget: float = 100_000.0
    max_mech_hours: float = 500.0
    max_elec_hours: float = 200.0
    max_inst_hours: float = 200.0
    max_civil_hours: float = 100.0
    turnaround_date: str = "2026-10-01"
    turnaround_days: int = 30
    planning_horizon_days: int = 365


@dataclass
class _SolverCfg:
    """Lightweight stand-in for SolverConfig in tests."""

    max_solve_seconds: float = 30.0
    num_workers: int = 4
    cost_scale: int = 1
    hour_scale: int = 10
    value_scale: int = 100


def _make_wos(n=20, seed=0, mandatory_idx=(), predecessor_map=None) -> pd.DataFrame:
    """Build a small synthetic work-order table for controlled testing."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "wo_id": [f"WO-{i:03d}" for i in range(n)],
            "estimated_cost_usd": rng.uniform(500, 8000, n).round(2),
            "mech_hours": rng.uniform(1, 40, n).round(1),
            "elec_hours": rng.uniform(0, 20, n).round(1),
            "inst_hours": rng.uniform(0, 20, n).round(1),
            "civil_hours": rng.uniform(0, 10, n).round(1),
            "net_value_usd": rng.uniform(-2000, 20000, n).round(2),
            "mandatory": [i in mandatory_idx for i in range(n)],
        }
    )
    df["predecessor_wo_id"] = None
    if predecessor_map:
        for succ_i, pred_i in predecessor_map.items():
            df.loc[succ_i, "predecessor_wo_id"] = df.loc[pred_i, "wo_id"]
    return df


class TestBudgetConstraint:
    def test_never_exceeds_budget(self):
        """Across many random seeds, the optimal solution must respect budget."""
        cfg = _Cfg(total_budget=50_000.0)
        for seed in range(10):
            wos = _make_wos(n=40, seed=seed)
            result = TurnaroundSolver(wos, config=cfg).solve()
            total_cost = result.selected_schedule["estimated_cost_usd"].sum()
            assert (
                total_cost <= cfg.total_budget + 1e-6
            ), f"seed={seed}: spent {total_cost} > budget {cfg.total_budget}"

    def test_tight_budget_selects_fewer_tasks(self):
        wos = _make_wos(n=30, seed=1)
        loose = TurnaroundSolver(wos, config=_Cfg(total_budget=200_000.0)).solve()
        tight = TurnaroundSolver(wos, config=_Cfg(total_budget=5_000.0)).solve()
        assert tight.summary["tasks_selected"] <= loose.summary["tasks_selected"]


class TestCraftHourConstraints:
    def test_never_exceeds_any_trade_capacity(self):
        cfg = _Cfg(
            total_budget=1_000_000.0,  # budget not binding
            max_mech_hours=100.0,
            max_elec_hours=50.0,
            max_inst_hours=50.0,
            max_civil_hours=25.0,
        )
        for seed in range(10):
            wos = _make_wos(n=50, seed=seed)
            result = TurnaroundSolver(wos, config=cfg).solve()
            sel = result.selected_schedule
            assert sel["mech_hours"].sum() <= cfg.max_mech_hours + 1e-6
            assert sel["elec_hours"].sum() <= cfg.max_elec_hours + 1e-6
            assert sel["inst_hours"].sum() <= cfg.max_inst_hours + 1e-6
            assert sel["civil_hours"].sum() <= cfg.max_civil_hours + 1e-6


class TestMandatoryConstraint:
    def test_all_mandatory_tasks_are_selected(self):
        wos = _make_wos(n=30, seed=2, mandatory_idx={0, 5, 10, 15, 20})
        cfg = _Cfg(total_budget=500_000.0)  # generous, mandatory shouldn't be starved
        result = TurnaroundSolver(wos, config=cfg).solve()
        mandatory_ids = wos.loc[[0, 5, 10, 15, 20], "wo_id"].tolist()
        selected_ids = set(result.selected_schedule["wo_id"])
        for mid in mandatory_ids:
            assert mid in selected_ids, f"Mandatory task {mid} was NOT selected!"

    def test_mandatory_selected_even_with_negative_net_value(self):
        """A safety-critical task that costs more than its computed risk value
        must still be executed — mandatory means mandatory."""
        wos = _make_wos(n=10, seed=3, mandatory_idx={0})
        wos.loc[0, "net_value_usd"] = -50_000.0  # deliberately unprofitable
        wos.loc[0, "estimated_cost_usd"] = 1_000.0
        cfg = _Cfg(total_budget=50_000.0)
        result = TurnaroundSolver(wos, config=cfg).solve()
        assert wos.loc[0, "wo_id"] in set(result.selected_schedule["wo_id"])

    def test_infeasible_when_mandatory_exceeds_budget(self):
        """If mandatory tasks alone exceed budget, solver must raise — never
        silently drop a mandatory task to 'fit'."""
        wos = _make_wos(n=5, seed=4, mandatory_idx={0, 1, 2, 3, 4})
        wos["estimated_cost_usd"] = 100_000.0  # 5 × $100k = $500k mandatory spend
        cfg = _Cfg(total_budget=10_000.0)  # budget far too small
        with pytest.raises(RuntimeError):
            TurnaroundSolver(wos, config=cfg).solve()

    def test_infeasible_error_message_blames_constraints_not_timeout(self):
        """INFEASIBLE is a definitive proof no solution exists — the error
        message should point at constraints (budget/hours/mandatory), not
        timeout, since more time would never help an infeasible problem."""
        wos = _make_wos(n=5, seed=4, mandatory_idx={0, 1, 2, 3, 4})
        wos["estimated_cost_usd"] = 100_000.0
        cfg = _Cfg(total_budget=10_000.0)
        with pytest.raises(RuntimeError, match="mandatory tasks don't exceed budget"):
            TurnaroundSolver(wos, config=cfg).solve()


class TestUnknownVsInfeasibleErrorMessages:
    """
    Regression tests for a real bug: CP-SAT's INFEASIBLE status (a
    DEFINITIVE proof no feasible solution exists) and UNKNOWN status (the
    solver ran out of time before reaching ANY conclusion — says nothing
    about whether a solution exists) used to share one identical error
    message that always blamed "mandatory tasks exceed budget/hours."
    That's actively misleading for UNKNOWN: a too-short --timeout produces
    UNKNOWN even on a perfectly feasible, easy problem, and the old message
    would send a confused user to increase budget when the real fix was
    --timeout.
    """

    def test_unknown_status_blames_timeout_not_budget(self):
        """A trivially easy, generously-budgeted problem given ZERO solve
        time should return UNKNOWN (not enough time to even start), and the
        error message must say so — not blame budget/hours, which are
        deliberately generous here."""
        wos = _make_wos(n=50, seed=9)  # generous, easily feasible problem
        cfg = _Cfg(total_budget=1_000_000.0)
        solver_cfg = _SolverCfg(max_solve_seconds=0.0)
        with pytest.raises(RuntimeError) as exc_info:
            TurnaroundSolver(wos, config=cfg, solver_cfg=solver_cfg).solve()
        msg = str(exc_info.value)
        assert "UNKNOWN" in msg
        assert "timeout" in msg.lower() or "--timeout" in msg
        assert "does not mean" in msg.lower() or "does NOT mean" in msg
        assert "exceed budget" not in msg.lower()

    def test_unknown_message_includes_actual_configured_timeout(self):
        wos = _make_wos(n=50, seed=9)
        cfg = _Cfg(total_budget=1_000_000.0)
        solver_cfg = _SolverCfg(max_solve_seconds=0.0)
        with pytest.raises(RuntimeError, match=r"0s"):
            TurnaroundSolver(wos, config=cfg, solver_cfg=solver_cfg).solve()


class TestPrecedenceConstraint:
    def test_successor_never_selected_without_predecessor(self):
        """If task 5 requires task 2 first, x_5 <= x_2 must hold in every
        solution — we force this by making task 2 unattractive (low value,
        high cost) so the solver would prefer to skip it absent the constraint."""
        wos = _make_wos(n=15, seed=5, predecessor_map={5: 2})
        wos.loc[2, "net_value_usd"] = -10_000.0  # unattractive predecessor
        wos.loc[2, "estimated_cost_usd"] = 9_000.0
        wos.loc[5, "net_value_usd"] = 50_000.0  # very attractive successor
        cfg = _Cfg(total_budget=30_000.0)
        result = TurnaroundSolver(wos, config=cfg).solve()
        sched = result.schedule.set_index("wo_id")
        pred_id = wos.loc[2, "wo_id"]
        succ_id = wos.loc[5, "wo_id"]
        if sched.loc[succ_id, "selected"]:
            assert sched.loc[pred_id, "selected"], (
                "Successor was selected without its predecessor — " "precedence constraint violated!"
            )

    def test_handles_null_predecessor_gracefully(self):
        """Regression test: pandas <NA> in predecessor_wo_id must not crash
        the model builder (previously raised TypeError: boolean value of
        NA is ambiguous)."""
        wos = _make_wos(n=10, seed=6)
        wos["predecessor_wo_id"] = pd.array([None] * 10, dtype="string")
        cfg = _Cfg(total_budget=20_000.0)
        # Should not raise
        result = TurnaroundSolver(wos, config=cfg).solve()
        assert result.summary["tasks_total"] == 10

    def test_handles_missing_predecessor_column_entirely(self):
        """A WO table that never tracked dependencies at all (column absent,
        not just empty) must hit the early-return guard, not crash."""
        wos = _make_wos(n=8, seed=7)
        wos = wos.drop(columns=["predecessor_wo_id"])
        assert "predecessor_wo_id" not in wos.columns
        cfg = _Cfg(total_budget=15_000.0)
        result = TurnaroundSolver(wos, config=cfg).solve()
        assert result.summary["tasks_total"] == 8


class TestObjectiveOptimality:
    def test_solver_reports_optimal_status_on_small_instances(self):
        wos = _make_wos(n=25, seed=8)
        cfg = _Cfg(total_budget=40_000.0)
        result = TurnaroundSolver(wos, config=cfg).solve()
        assert result.summary["solver_status"] in ("OPTIMAL", "FEASIBLE")

    def test_more_budget_never_decreases_objective(self):
        """Monotonicity sanity check: relaxing the budget constraint can only
        help (or tie), never hurt, the achievable objective value."""
        wos = _make_wos(n=30, seed=9)
        low = TurnaroundSolver(wos, config=_Cfg(total_budget=10_000.0)).solve()
        high = TurnaroundSolver(wos, config=_Cfg(total_budget=100_000.0)).solve()
        assert high.summary["total_net_value_usd"] >= low.summary["total_net_value_usd"] - 1e-6

    def test_scales_to_500_plus_tasks_within_timeout(self):
        """Mirrors the production scale called out in the project brief
        (500+ CMMS work orders) and asserts a real OPTIMAL/FEASIBLE solve."""
        wos = _make_wos(n=600, seed=42, mandatory_idx=set(range(0, 600, 11)))
        cfg = _Cfg(
            total_budget=400_000.0,
            max_mech_hours=8000.0,
            max_elec_hours=4000.0,
            max_inst_hours=4000.0,
            max_civil_hours=2000.0,
        )
        result = TurnaroundSolver(wos, config=cfg).solve()
        assert result.summary["solver_status"] in ("OPTIMAL", "FEASIBLE")
        assert result.summary["solve_time_s"] < 30.0


class TestKnapsackModelDirectly:
    """White-box tests against TurnaroundModel internals."""

    def test_variable_count_matches_task_count(self):
        wos = _make_wos(n=17, seed=10)
        m = TurnaroundModel(wos, _Cfg()).build()
        assert len(m.x) == 17

    def test_hour_scaling_is_reversible(self):
        wos = _make_wos(n=5, seed=11)
        m = TurnaroundModel(wos, _Cfg()).build()
        for i in range(5):
            original = wos.loc[i, "mech_hours"]
            scaled = m._mech[i]
            assert abs(scaled / HOUR_SCALE - original) < 0.05


class TestSolverResultRepr:
    def test_repr_is_human_readable_and_contains_key_numbers(self):
        wos = _make_wos(n=10, seed=12)
        cfg = _Cfg(total_budget=20_000.0)
        result = TurnaroundSolver(wos, config=cfg).solve()
        r = repr(result)
        assert "SolverResult(" in r
        assert "selected=" in r
        assert "budget_used=$" in r
        assert "roi=" in r
