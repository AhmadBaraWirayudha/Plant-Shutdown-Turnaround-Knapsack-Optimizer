"""
test_export.py — Tests for the star-schema mirror sheets built for the
Excel/Power BI export (the pure-logic part of export.py; the actual
openpyxl file-writing is covered by the pipeline smoke test instead).
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from src.reporting.export import _build_dimension_sheets, export_to_excel


def _make_schedule(n=12) -> pd.DataFrame:
    task_types = ["Inspection", "Repair", "Overhaul"]
    priorities = ["Critical", "High", "Medium", "Low"]
    risk_levels = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    rows = []
    for i in range(n):
        rows.append(
            {
                "wo_id": f"WO-{i:05d}",
                "description": f"Task {i}",
                "asset_tag": f"PMP-{i % 5:04d}",  # 5 distinct assets
                "asset_class": "PMP",
                "asset_name": "Centrifugal Pump",
                "area": "Unit-100",
                "replace_usd": 50000.0,
                "c_safety": 3,
                "c_env": 2,
                "c_prod": 4,
                "c_cost": 2,
                "task_type": task_types[i % len(task_types)],
                "priority": priorities[i % len(priorities)],
                "risk_level": risk_levels[i % len(risk_levels)],
                "predecessor_wo_id": None,
                "mandatory": i == 0,
                "age_days": 500.0,
                "estimated_cost_usd": 10000.0,
                "mech_hours": 10.0,
                "elec_hours": 2.0,
                "inst_hours": 1.0,
                "civil_hours": 0.0,
                "total_craft_hours": 13.0,
                "duration_days": 2,
                "fitted_beta": 2.0,
                "fitted_eta": 1500.0,
                "failure_prob": 0.3,
                "rul_days": 800.0,
                "consequence_score": 3.0,
                "likelihood_tier": 3,
                "consequence_tier": 3,
                "risk_score": 9,
                "deferred_cost_usd": 5000.0,
                "net_value_usd": 1000.0,
                "selected": i % 2 == 0,
                "decision": "INCLUDE" if i % 2 == 0 else "DEFER",
            }
        )
    return pd.DataFrame(rows)


class TestExportToExcelEndToEnd:
    """
    Runs export_to_excel against a REAL SolverResult (from an actual small
    TurnaroundSolver.solve() call, not a hand-built mock) and inspects the
    written workbook directly. This is what catches a column-name mismatch
    between the solver's actual output and what export.py expects to find
    — a hand-rolled fixture, built with the "correct" column names by
    construction, could never catch that class of bug.
    """

    @staticmethod
    def _make_real_result(n=12, budget=200_000.0):
        from src.optimization.solver import TurnaroundSolver

        rng = np.random.default_rng(0)
        task_types = ["Inspection", "Repair", "Overhaul"]
        priorities = ["Critical", "High", "Medium", "Low"]
        risk_levels = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

        rows = []
        for i in range(n):
            rows.append(
                {
                    "wo_id": f"WO-{i:05d}",
                    "description": f"Task {i}",
                    "asset_tag": f"PMP-{i % 4:04d}",
                    "asset_class": "PMP",
                    "asset_name": "Centrifugal Pump",
                    "area": "Unit-100",
                    "replace_usd": 50000.0,
                    "c_safety": 3,
                    "c_env": 2,
                    "c_prod": 4,
                    "c_cost": 2,
                    "task_type": task_types[i % len(task_types)],
                    "priority": priorities[i % len(priorities)],
                    "risk_level": risk_levels[i % len(risk_levels)],
                    "predecessor_wo_id": None,
                    "mandatory": i == 0,
                    "age_days": float(rng.uniform(100, 2000)),
                    "estimated_cost_usd": float(rng.uniform(5000, 30000)),
                    "mech_hours": float(rng.uniform(1, 20)),
                    "elec_hours": float(rng.uniform(0, 5)),
                    "inst_hours": float(rng.uniform(0, 5)),
                    "civil_hours": float(rng.uniform(0, 2)),
                    "total_craft_hours": float(rng.uniform(5, 30)),
                    "duration_days": int(rng.integers(1, 5)),
                    "fitted_beta": 2.0,
                    "fitted_eta": 1500.0,
                    "failure_prob": float(rng.uniform(0, 1)),
                    "rul_days": float(rng.uniform(0, 2000)),
                    "consequence_score": float(rng.uniform(1, 5)),
                    "likelihood_tier": int(rng.integers(1, 6)),
                    "consequence_tier": int(rng.integers(1, 6)),
                    "risk_score": int(rng.integers(1, 26)),
                    "deferred_cost_usd": float(rng.uniform(0, 40000)),
                    "net_value_usd": float(rng.uniform(-5000, 40000)),
                }
            )
        wos = pd.DataFrame(rows)

        class _Cfg:
            total_budget = budget
            max_mech_hours = 1000.0
            max_elec_hours = 500.0
            max_inst_hours = 500.0
            max_civil_hours = 200.0

        return TurnaroundSolver(wos, config=_Cfg()).solve()

    def test_writes_a_real_xlsx_file(self, tmp_path):
        result = self._make_real_result()
        out_path = tmp_path / "test_export.xlsx"
        returned_path = export_to_excel(result, out_path=out_path)

        assert returned_path == out_path
        assert out_path.exists()
        assert out_path.stat().st_size > 0

    def test_workbook_contains_every_expected_sheet(self, tmp_path):
        import openpyxl

        result = self._make_real_result()
        out_path = tmp_path / "test_export.xlsx"
        export_to_excel(result, out_path=out_path)

        wb = openpyxl.load_workbook(out_path, read_only=True)
        expected = {
            "OptimizedSchedule",
            "Selected",
            "Deferred",
            "SummaryKPIs",
            "CapacityUtilization",
            "RiskMatrix",
            "ByArea",
            "ByEquipmentClass",
            "Dim_Asset",
            "Dim_TaskType",
            "Dim_Priority",
            "Dim_RiskLevel",
            "FactWorkOrderDecision",
        }
        assert expected.issubset(set(wb.sheetnames))

    def test_optimized_schedule_row_count_matches_total_tasks(self, tmp_path):
        import openpyxl

        result = self._make_real_result(n=12)
        out_path = tmp_path / "test_export.xlsx"
        export_to_excel(result, out_path=out_path)

        wb = openpyxl.load_workbook(out_path, read_only=True)
        ws = wb["OptimizedSchedule"]
        assert ws.max_row - 1 == 12  # minus header row

    def test_creates_parent_directory_if_missing(self, tmp_path):
        result = self._make_real_result(n=5)
        nested_path = tmp_path / "does" / "not" / "exist" / "export.xlsx"
        assert not nested_path.parent.exists()
        export_to_excel(result, out_path=nested_path)
        assert nested_path.exists()


class TestBuildDimensionSheets:

    def test_returns_all_five_expected_sheets(self):
        sched = _make_schedule()
        sheets = _build_dimension_sheets(sched)
        assert set(sheets.keys()) == {
            "Dim_Asset",
            "Dim_TaskType",
            "Dim_Priority",
            "Dim_RiskLevel",
            "FactWorkOrderDecision",
        }

    def test_dim_asset_has_one_row_per_distinct_asset(self):
        sched = _make_schedule(n=20)  # uses 5 distinct asset tags (i % 5)
        sheets = _build_dimension_sheets(sched)
        assert len(sheets["Dim_Asset"]) == 5
        assert sheets["Dim_Asset"]["asset_tag"].is_unique

    def test_dim_task_type_has_no_duplicates(self):
        sched = _make_schedule(n=30)
        sheets = _build_dimension_sheets(sched)
        dim = sheets["Dim_TaskType"]
        assert dim["task_type_name"].is_unique
        assert dim["task_type_id"].is_unique
        assert set(dim["task_type_name"]) == set(sched["task_type"].unique())

    def test_dim_priority_weights_match_convention(self):
        sched = _make_schedule()
        sheets = _build_dimension_sheets(sched)
        dim = sheets["Dim_Priority"].set_index("priority_name")
        assert dim.loc["Critical", "priority_weight"] == 4
        assert dim.loc["Low", "priority_weight"] == 1

    def test_dim_risk_level_sort_order_matches_severity_convention(self):
        sched = _make_schedule()
        sheets = _build_dimension_sheets(sched)
        dim = sheets["Dim_RiskLevel"].set_index("risk_level_name")
        assert dim.loc["LOW", "sort_order"] < dim.loc["CRITICAL", "sort_order"]
        assert dim.loc["MEDIUM", "sort_order"] < dim.loc["HIGH", "sort_order"]

    def test_fact_table_has_no_unresolved_foreign_keys(self):
        """Every fact row must successfully resolve to a task_type_id,
        priority_id, and risk_level_id — a merge failure here would mean
        the Power BI relationships silently lose rows."""
        sched = _make_schedule(n=25)
        sheets = _build_dimension_sheets(sched)
        fact = sheets["FactWorkOrderDecision"]
        assert fact["task_type_id"].notna().all()
        assert fact["priority_id"].notna().all()
        assert fact["risk_level_id"].notna().all()

    def test_fact_table_row_count_matches_schedule(self):
        sched = _make_schedule(n=17)
        sheets = _build_dimension_sheets(sched)
        assert len(sheets["FactWorkOrderDecision"]) == 17

    def test_fact_table_does_not_carry_raw_categorical_columns(self):
        """The fact table should reference dimensions via _id columns, not
        duplicate the raw task_type/priority/risk_level strings — that's
        the whole point of normalizing into a star schema."""
        sched = _make_schedule()
        sheets = _build_dimension_sheets(sched)
        fact = sheets["FactWorkOrderDecision"]
        assert "task_type" not in fact.columns
        assert "priority" not in fact.columns
        assert "risk_level" not in fact.columns
        assert "task_type_id" in fact.columns

    def test_handles_single_distinct_value_per_dimension(self):
        """Edge case: a schedule where every row shares the same task_type,
        priority, and risk_level must still produce a valid 1-row dimension,
        not crash on a degenerate unique() call."""
        sched = _make_schedule(n=5)
        sched["task_type"] = "Inspection"
        sched["priority"] = "Medium"
        sched["risk_level"] = "LOW"
        sheets = _build_dimension_sheets(sched)
        assert len(sheets["Dim_TaskType"]) == 1
        assert len(sheets["Dim_Priority"]) == 1
        assert len(sheets["Dim_RiskLevel"]) == 1
        assert sheets["FactWorkOrderDecision"]["task_type_id"].notna().all()
