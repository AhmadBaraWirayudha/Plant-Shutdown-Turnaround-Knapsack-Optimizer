"""
test_dashboard.py — Regression tests for generate_dashboard()'s path
handling, mirroring the same bug class fixed in export_to_excel() (see
tests/test_export.py and docs/METHODOLOGY.md §5).

This is deliberately a small, focused file rather than a full dashboard
test suite — chart-rendering correctness (Plotly figure construction) is
validated by the end-to-end pipeline smoke test in CI, since dashboard.py
is primarily I/O/rendering code rather than decision logic. These tests
target the two real bugs found: a plain string out_path crashing on
.parent.mkdir()/.write_text(), and the stale-default-argument pattern on
DASHBOARD_DIR.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from src.optimization.solver import TurnaroundSolver
from src.reporting.dashboard import generate_dashboard


def _make_real_result(n=10, budget=200_000.0):
    """Build a real SolverResult from an actual small solve — not a mock —
    so these tests exercise generate_dashboard() against genuine solver
    output, matching the pattern in tests/test_export.py."""
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
        turnaround_date = "2026-10-01"

    return TurnaroundSolver(wos, config=_Cfg()).solve()


class TestGenerateDashboardPathHandling:
    def test_writes_a_real_html_file(self, tmp_path):
        result = _make_real_result()
        out_path = tmp_path / "test_dashboard.html"
        returned = generate_dashboard(result, out_path=out_path)
        assert returned == out_path
        assert out_path.exists()
        assert out_path.stat().st_size > 0

    def test_accepts_plain_string_path_not_just_path_object(self, tmp_path):
        """
        Regression test for a real bug: generate_dashboard(out_path=...)
        used to crash with 'AttributeError: str object has no attribute
        parent' if given a bare string instead of a pathlib.Path, since
        .parent.mkdir(...) and .write_text(...) both require a true Path.
        """
        str_path = str(tmp_path / "string_path_dashboard.html")
        result = _make_real_result()
        returned = generate_dashboard(result, out_path=str_path)
        assert returned == Path(str_path)
        assert Path(str_path).exists()

    def test_omitted_out_path_resolves_live_dashboard_dir(self, monkeypatch, tmp_path):
        """The out_path=None sentinel must resolve DASHBOARD_DIR at CALL
        time, not whatever it was when this module was first imported —
        same bug class as docs/METHODOLOGY.md §5."""
        import src.reporting.dashboard as dashboard_mod

        monkeypatch.setattr(dashboard_mod, "DASHBOARD_DIR", tmp_path)
        result = _make_real_result()
        returned = generate_dashboard(result)
        assert returned == tmp_path / "turnaround_dashboard.html"
        assert returned.exists()

    def test_creates_parent_directory_if_missing(self, tmp_path):
        result = _make_real_result(n=5)
        nested_path = tmp_path / "does" / "not" / "exist" / "dashboard.html"
        assert not nested_path.parent.exists()
        generate_dashboard(result, out_path=nested_path)
        assert nested_path.exists()

    def test_html_contains_expected_structure(self, tmp_path):
        result = _make_real_result()
        out_path = tmp_path / "test_dashboard.html"
        generate_dashboard(result, out_path=out_path)
        html = out_path.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in html
        assert "<html" in html.lower()

    def test_xss_in_field_values_is_escaped(self, tmp_path):
        """
        Regression test for a real XSS vulnerability: user-controlled string
        fields (wo_id, asset_tag, area, task_type, priority, decision,
        risk_level) were interpolated directly into the HTML table without
        html.escape(). A real CMMS description or area field containing
        '<script>' would execute as JavaScript in any browser rendering the
        dashboard. Every string column is now escaped before interpolation.
        """
        import html as html_lib

        result = _make_real_result(n=3)
        # Inject a script tag into the wo_id and area fields
        result.schedule.loc[0, "wo_id"] = '<script>alert("xss")</script>'
        result.schedule.loc[0, "area"] = 'Unit-<b onmouseover="evil()">100</b>'
        result.selected_schedule = result.schedule[result.schedule["selected"]].copy()
        result.deferred_schedule = result.schedule[~result.schedule["selected"]].copy()

        out_path = tmp_path / "xss_test.html"
        generate_dashboard(result, out_path=out_path)
        content = out_path.read_text(encoding="utf-8")

        # The literal script tags must NOT appear in the output
        assert "<script>alert" not in content
        assert 'onmouseover="evil()"' not in content

        # The HTML-escaped equivalents SHOULD appear (data is preserved, just safe)
        assert "&lt;script&gt;" in content or html_lib.escape('<script>alert("xss")</script>') in content
