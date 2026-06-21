"""
test_db.py — Tests for the star-schema database layer.

Two tiers, deliberately:
  1. Unit tests against a small, hand-built DataFrame fixture — fast,
     isolated, exercise writer.py's upsert/transaction logic directly.
  2. One integration test that runs the REAL pipeline (data generator →
     transform → Weibull → risk) and feeds its actual output into
     write_results_to_db. Hand-built fixtures can't catch a column-name
     mismatch between modules if the fixture happens to use the "right"
     names by construction — only running the real chain does.

A real bug was caught by the multi-run test below during development: a
single-column SQLAlchemy select via session.scalars() returns bare scalar
values, not ORM row objects, so `row.asset_tag` raised AttributeError on
any call after the first (the first call's dim_asset table was empty, so
the buggy attribute access was never actually executed — the loop body
never ran). That's exactly the kind of bug that only a *second* write
exposes, which is why every idempotency test here writes at least twice.
"""

from __future__ import annotations
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import select, text

from src.db.connection import get_engine, get_database_url, _redact
from src.db.schema import DimAsset
from src.db.writer import write_results_to_db, init_db
from src.db.queries import (
    list_runs,
    latest_run_id,
    get_run_facts,
    compare_runs_summary,
    fact_row_count,
)

# ─── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def temp_engine():
    """A fresh SQLite database in a temp directory, torn down after the test."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test_turnaround.db"
        engine = get_engine(f"sqlite:///{db_path}")
        yield engine
        engine.dispose()


class _FakeSummary(dict):
    """SolverResult.summary is a plain dict — this just documents the
    minimal keys write_results_to_db actually reads."""


class _FakeResult:
    """Minimal stand-in for SolverResult, carrying only `.schedule` and
    `.summary`, exactly what write_results_to_db consumes."""

    def __init__(self, schedule: pd.DataFrame, summary: dict):
        self.schedule = schedule
        self.summary = summary


class _FakeTaCfg:
    turnaround_date = "2026-10-01"
    planning_horizon_days = 365


def _make_schedule(n=10, seed=0) -> pd.DataFrame:
    """A small, fully-specified schedule covering every column the fact
    table needs — built by hand for fast, isolated writer tests."""
    rng = np.random.default_rng(seed)
    task_types = ["Inspection", "Repair", "Overhaul"]
    priorities = ["Critical", "High", "Medium", "Low"]
    risk_levels = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

    rows = []
    for i in range(n):
        rows.append(
            {
                "wo_id": f"WO-{i:05d}",
                "description": f"Test task {i}",
                "asset_tag": f"PMP-{i % 4:04d}",  # only 4 distinct assets across n rows
                "asset_class": "PMP",
                "area": "Unit-100",
                "task_type": task_types[i % len(task_types)],
                "priority": priorities[i % len(priorities)],
                "mandatory": i == 0,
                "predecessor_wo_id": None,
                "age_days": float(rng.uniform(100, 2000)),
                "estimated_cost_usd": float(rng.uniform(1000, 20000)),
                "mech_hours": float(rng.uniform(1, 40)),
                "elec_hours": float(rng.uniform(0, 10)),
                "inst_hours": float(rng.uniform(0, 10)),
                "civil_hours": float(rng.uniform(0, 5)),
                "total_craft_hours": float(rng.uniform(10, 60)),
                "duration_days": int(rng.integers(1, 10)),
                "fitted_beta": 2.2,
                "fitted_eta": 1500.0,
                "failure_prob": float(rng.uniform(0, 1)),
                "rul_days": float(rng.uniform(0, 2000)),
                "consequence_score": float(rng.uniform(1, 5)),
                "likelihood_tier": int(rng.integers(1, 6)),
                "consequence_tier": int(rng.integers(1, 6)),
                "risk_score": int(rng.integers(1, 26)),
                "risk_level": risk_levels[i % len(risk_levels)],
                "deferred_cost_usd": float(rng.uniform(0, 50000)),
                "net_value_usd": float(rng.uniform(-5000, 50000)),
                "replace_usd": 50000.0,
                "c_safety": 3,
                "c_env": 2,
                "c_prod": 4,
                "c_cost": 2,
                "selected": i % 2 == 0,
                "decision": "INCLUDE" if i % 2 == 0 else "DEFER",
            }
        )
    return pd.DataFrame(rows)


def _make_summary(schedule: pd.DataFrame, budget: float = 100_000.0) -> dict:
    sel = schedule[schedule["selected"]]
    return {
        "solver_status": "OPTIMAL",
        "solve_time_s": 0.05,
        "tasks_total": len(schedule),
        "tasks_selected": int(sel.shape[0]),
        "budget_usd": budget,
        "budget_used_usd": float(sel.estimated_cost_usd.sum()),
        "budget_utilisation": float(sel.estimated_cost_usd.sum()) / budget,
        "max_mech_hours": 1000.0,
        "max_elec_hours": 500.0,
        "max_inst_hours": 500.0,
        "max_civil_hours": 200.0,
        "total_net_value_usd": float(sel.net_value_usd.sum()),
        "roi_ratio": float(sel.net_value_usd.sum()) / max(float(sel.estimated_cost_usd.sum()), 1),
        "total_risk_score_reduced": int(sel.risk_score.sum()),
    }


# ─── Schema tests ──────────────────────────────────────────────────────────────


class TestConnectionUrlResolution:
    def test_explicit_override_wins_over_everything(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql://envuser@envhost/envdb")
        url = get_database_url(override="sqlite:///explicit.db")
        assert url == "sqlite:///explicit.db"

    def test_env_var_used_when_no_override(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql://envuser@envhost/envdb")
        url = get_database_url(override=None)
        assert url == "postgresql://envuser@envhost/envdb"

    def test_falls_back_to_local_sqlite_when_nothing_set(self, monkeypatch):
        monkeypatch.delenv("DATABASE_URL", raising=False)
        url = get_database_url(override=None)
        assert url.startswith("sqlite:///")
        assert "turnaround.db" in url

    def test_get_engine_works_with_env_var_only(self, monkeypatch, tmp_path):
        db_file = tmp_path / "env_test.db"
        monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_file}")
        engine = get_engine()  # no explicit override — must resolve via env var
        init_db(engine)
        assert db_file.exists()


class TestRedactPassword:
    def test_masks_password_in_postgres_url(self):
        url = "postgresql+psycopg2://myuser:supersecret@dbhost:5432/turnaround"
        redacted = _redact(url)
        assert "supersecret" not in redacted
        assert "myuser" in redacted
        assert "***" in redacted

    def test_leaves_credential_free_url_unchanged(self):
        url = "sqlite:////home/claude/database/turnaround.db"
        assert _redact(url) == url

    def test_leaves_url_without_colon_in_creds_unchanged(self):
        """A username with no password (no colon before @) should pass through
        rather than crash on an unexpected split."""
        url = "postgresql://myuser@dbhost/turnaround"
        # Should not raise, and should not invent a password
        result = _redact(url)
        assert "myuser" in result


class TestSchemaCreation:
    def test_creates_all_six_tables(self, temp_engine):
        init_db(temp_engine)
        with temp_engine.connect() as conn:
            tables = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'")).scalars().all()
        expected = {
            "dim_run",
            "dim_asset",
            "dim_task_type",
            "dim_priority",
            "dim_risk_level",
            "fact_work_order_decision",
        }
        assert expected.issubset(set(tables))

    def test_idempotent_to_call_twice(self, temp_engine):
        init_db(temp_engine)
        init_db(temp_engine)  # must not raise on a table that already exists

    def test_foreign_keys_enforced_on_sqlite(self, temp_engine):
        """SQLite has FK enforcement OFF by default — connection.py must
        turn it on, or invalid fact rows would be silently accepted."""
        init_db(temp_engine)
        with temp_engine.connect() as conn:
            with pytest.raises(Exception):
                conn.execute(
                    text(
                        "INSERT INTO fact_work_order_decision "
                        "(run_id, asset_tag, task_type_id, priority_id, risk_level_id, "
                        "wo_id, description, mandatory, age_days, estimated_cost_usd, "
                        "mech_hours, elec_hours, inst_hours, civil_hours, total_craft_hours, "
                        "duration_days, fitted_beta, fitted_eta, failure_prob, rul_days, "
                        "consequence_score, likelihood_tier, consequence_tier, risk_score, "
                        "deferred_cost_usd, net_value_usd, selected, decision) "
                        "VALUES (999, 'NONEXISTENT', 1, 1, 1, 'WO-X', 'x', 0, 1, 1, "
                        "1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 'DEFER')"
                    )
                )
                conn.commit()


# ─── Writer round-trip tests ───────────────────────────────────────────────────


class TestWriteResultsToDb:
    def test_returns_incrementing_run_ids(self, temp_engine):
        schedule = _make_schedule(n=10)
        summary = _make_summary(schedule)
        result = _FakeResult(schedule, summary)

        run_id_1 = write_results_to_db(temp_engine, result, _FakeTaCfg())
        run_id_2 = write_results_to_db(temp_engine, result, _FakeTaCfg())
        assert run_id_2 == run_id_1 + 1

    def test_fact_row_count_matches_schedule_length(self, temp_engine):
        schedule = _make_schedule(n=15)
        summary = _make_summary(schedule)
        result = _FakeResult(schedule, summary)
        run_id = write_results_to_db(temp_engine, result, _FakeTaCfg())
        assert fact_row_count(temp_engine, run_id=run_id) == 15

    def test_dim_asset_does_not_duplicate_across_runs(self, temp_engine):
        """Regression test for the session.scalars() AttributeError bug:
        writing the SAME schedule twice must leave dim_asset with exactly
        the distinct asset count, not double it, and must not crash."""
        schedule = _make_schedule(n=20)  # uses asset tags PMP-0000..PMP-0003 (4 distinct)
        summary = _make_summary(schedule)
        result = _FakeResult(schedule, summary)

        write_results_to_db(temp_engine, result, _FakeTaCfg())
        write_results_to_db(temp_engine, result, _FakeTaCfg())  # must NOT raise

        with temp_engine.connect() as conn:
            count = conn.execute(select(DimAsset)).fetchall()
        assert len(count) == 4

    def test_three_runs_produce_three_times_the_fact_rows(self, temp_engine):
        schedule = _make_schedule(n=12)
        summary = _make_summary(schedule)
        result = _FakeResult(schedule, summary)

        for _ in range(3):
            write_results_to_db(temp_engine, result, _FakeTaCfg())

        assert fact_row_count(temp_engine) == 36

    def test_round_trip_cost_sum_matches_summary_exactly(self, temp_engine):
        """The number a planner sees in the dashboard must be EXACTLY the
        number that lands in the database — no silent rounding drift."""
        schedule = _make_schedule(n=25)
        summary = _make_summary(schedule)
        result = _FakeResult(schedule, summary)
        run_id = write_results_to_db(temp_engine, result, _FakeTaCfg())

        facts = get_run_facts(temp_engine, run_id)
        db_selected_cost = facts.loc[facts["selected"].astype(bool), "estimated_cost_usd"].sum()
        assert db_selected_cost == pytest.approx(summary["budget_used_usd"], abs=0.01)

    def test_transaction_rolls_back_completely_on_failure(self, temp_engine):
        """If anything fails mid-write, NOTHING should be persisted —
        a half-written run is worse than no run, since a Power BI report
        could silently show partial data as if it were complete.

        Note: corrupting `risk_level` to an unrecognized string would NOT
        actually trigger a failure here, because _upsert_lookup_risk_levels
        runs first and would simply add it as a new, valid lookup entry —
        the upsert pattern is self-healing for unrecognized categoricals by
        design. To force a genuine, unrecoverable failure we corrupt a
        numeric field instead, simulating a CMMS export with a malformed
        cost value that can't be cast to float."""
        schedule = _make_schedule(n=10)
        summary = _make_summary(schedule)

        bad_schedule = schedule.copy()
        # Force object dtype first — newer pandas refuses to silently
        # coerce a float64 column to hold a string via .loc assignment.
        bad_schedule["estimated_cost_usd"] = bad_schedule["estimated_cost_usd"].astype(object)
        bad_schedule.loc[0, "estimated_cost_usd"] = "not_a_number"
        bad_result = _FakeResult(bad_schedule, summary)

        with pytest.raises(ValueError):
            write_results_to_db(temp_engine, bad_result, _FakeTaCfg())

        # Nothing should have been committed — zero runs, zero facts
        assert latest_run_id(temp_engine) is None
        assert fact_row_count(temp_engine) == 0

    def test_uses_custom_run_label(self, temp_engine):
        schedule = _make_schedule(n=5)
        summary = _make_summary(schedule)
        result = _FakeResult(schedule, summary)
        write_results_to_db(temp_engine, result, _FakeTaCfg(), run_label="My custom scenario")

        runs = list_runs(temp_engine)
        assert runs.iloc[0]["run_label"] == "My custom scenario"

    def test_auto_generates_label_when_none_given(self, temp_engine):
        schedule = _make_schedule(n=5)
        summary = _make_summary(schedule, budget=250_000.0)
        result = _FakeResult(schedule, summary)
        write_results_to_db(temp_engine, result, _FakeTaCfg(), run_label=None)

        runs = list_runs(temp_engine)
        assert "250,000" in runs.iloc[0]["run_label"]


# ─── Query helper tests ─────────────────────────────────────────────────────────


class TestQueryHelpers:
    def test_latest_run_id_is_none_on_empty_db(self, temp_engine):
        init_db(temp_engine)
        assert latest_run_id(temp_engine) is None

    def test_latest_run_id_tracks_most_recent_write(self, temp_engine):
        schedule = _make_schedule(n=5)
        summary = _make_summary(schedule)
        result = _FakeResult(schedule, summary)

        first = write_results_to_db(temp_engine, result, _FakeTaCfg())
        second = write_results_to_db(temp_engine, result, _FakeTaCfg())
        assert latest_run_id(temp_engine) == second
        assert second != first

    def test_compare_runs_summary_has_one_row_per_run(self, temp_engine):
        schedule = _make_schedule(n=5)
        summary = _make_summary(schedule)
        result = _FakeResult(schedule, summary)
        for _ in range(3):
            write_results_to_db(temp_engine, result, _FakeTaCfg())

        comparison = compare_runs_summary(temp_engine)
        assert len(comparison) == 3
        assert list(comparison["run_id"]) == [1, 2, 3]

    def test_get_run_facts_joins_dimension_names_not_raw_ids(self, temp_engine):
        schedule = _make_schedule(n=5)
        summary = _make_summary(schedule)
        result = _FakeResult(schedule, summary)
        run_id = write_results_to_db(temp_engine, result, _FakeTaCfg())

        facts = get_run_facts(temp_engine, run_id)
        # Human-readable names should appear, not opaque integer FK columns
        assert "asset_class" in facts.columns
        assert "risk_level" in facts.columns
        assert facts["risk_level"].isin(["LOW", "MEDIUM", "HIGH", "CRITICAL"]).all()


# ─── Integration test: real pipeline → real DB write ──────────────────────────


class TestRealPipelineIntegration:
    """
    Runs the ACTUAL data-generation → transform → Weibull → risk chain (not
    a hand-built fixture) and writes its real output into the database.
    This is what would catch a column-name mismatch between risk.py's
    output and writer.py's expectations that a hand-rolled fixture,
    constructed with the "correct" names by definition, never could.
    """

    def test_real_pipeline_output_writes_cleanly(self, temp_engine):
        import numpy as np
        from src.utils.data_generator import (
            generate_asset_master,
            generate_failure_history,
            generate_work_orders,
        )
        from src.etl.transform import run_transforms, validate_referential_integrity, enrich_with_asset_name
        from src.modeling.weibull import run_weibull_analysis
        from src.modeling.risk import compute_risk_scores
        from src.optimization.solver import TurnaroundSolver

        # Call the sub-functions directly (not generate_all()) so this test
        # never writes to the real data/raw/ directory as a side effect —
        # generate_all() always persists CSVs there with no path override.
        # An explicit rng is constructed here rather than relying on any
        # module/global default, exercising the same explicit-dependency
        # pattern the bug-fix introduced (see tests/test_data_generator.py
        # for the dedicated regression tests on that fix itself).
        rng = np.random.default_rng(99)
        assets = generate_asset_master(rng)
        failures = generate_failure_history(assets, rng)
        wos = generate_work_orders(assets, rng, n=40)

        clean_wos, clean_failures = run_transforms(wos, failures)
        clean_wos = validate_referential_integrity(clean_wos, assets)
        clean_wos = enrich_with_asset_name(clean_wos, assets)
        wos_reliability = run_weibull_analysis(clean_wos, clean_failures)
        wos_scored = compute_risk_scores(wos_reliability)

        class _Cfg:
            total_budget = 500_000.0
            max_mech_hours = 2000.0
            max_elec_hours = 1000.0
            max_inst_hours = 1000.0
            max_civil_hours = 500.0
            turnaround_date = "2026-10-01"
            planning_horizon_days = 365

        result = TurnaroundSolver(wos_scored, config=_Cfg()).solve()

        run_id = write_results_to_db(temp_engine, result, _Cfg())

        assert fact_row_count(temp_engine, run_id=run_id) == len(wos_scored)
        facts = get_run_facts(temp_engine, run_id)
        db_selected_cost = facts.loc[facts["selected"].astype(bool), "estimated_cost_usd"].sum()
        assert db_selected_cost == pytest.approx(result.summary["budget_used_usd"], abs=0.01)
