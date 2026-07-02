"""
tests/test_scenarios.py — Tests for the concurrent scenario management system.

Coverage
--------
* create_scenario: validation, uniqueness, defaults
* list_scenarios: filtering by status, owner, shared
* get_scenario / get_scenario_params
* lock_scenario / unlock_scenario: happy path, conflict, wrong owner
* update_scenario: field changes, Ellipsis sentinel, locked-guard
* archive_scenario: soft delete, blocks further operations
* clone_scenario: parameter inheritance, budget adjustment, parent linkage
* set_current_run: idempotent, wrong scenario raises
* _compare_and_swap_update: version mismatch → ScenarioConflictError
* Concurrent lock race: two threads, only one wins
* queries.py: list_scenario_runs, get_scenario_facts, scenario_kpi_history
* comparison.py: compare_scenarios KPIs and WO diffs
* build_config_for_scenario: None fields inherit from base_cfg
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from src.db.schema import Base, DimScenario, DimRun, ScenarioStatus, FactWorkOrderDecision
from src.db.schema import DimTaskType, DimPriority, DimRiskLevel, DimAsset
from src.scenarios.manager import (
    create_scenario,
    list_scenarios,
    get_scenario,
    get_scenario_params,
    lock_scenario,
    unlock_scenario,
    update_scenario,
    archive_scenario,
    clone_scenario,
    set_current_run,
    _compare_and_swap_update,
    ScenarioNotFoundError,
    ScenarioLockedError,
    ScenarioConflictError,
    ScenarioArchivedError,
)
from src.scenarios.comparison import (
    compare_scenarios,
    compare_many_scenarios,
    ScenarioNotSolvedError,
    ScenarioComparison,
)
from src.scenarios.runner import build_config_for_scenario
from src.db.queries import list_scenario_runs, get_scenario_facts, scenario_kpi_history
from src.utils.config import TurnaroundConfig


# ─── Test database fixture ────────────────────────────────────────────────────


@pytest.fixture
def engine():
    """
    In-memory SQLite engine with full star schema including dim_scenario.

    StaticPool is required for multi-threaded tests: the default NullPool
    (and even QueuePool) assigns a NEW in-memory database to each connection,
    so background threads see empty schema.  StaticPool routes every
    connection to the same underlying sqlite3.Connection object, which keeps
    all threads working against the same in-memory database.

    ``check_same_thread=False`` suppresses SQLite's thread-safety check;
    SQLAlchemy's own connection-pool serialises access correctly.
    """
    eng = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(eng)
    yield eng
    eng.dispose()


# ─── Minimal dim_run + fact seeding helper ────────────────────────────────────

_TASK_TYPES = {"Inspection": 1, "Replacement": 2}
_PRIORITIES = {"High": 2, "Critical": 1}
_RISK_LEVELS = {"MEDIUM": 2, "HIGH": 3}


def _seed_lookups(engine):
    """Insert lookup-dimension rows if they don't already exist (idempotent)."""
    with Session(engine) as s:
        # Only insert if not already present (idempotent across repeated calls
        # within the same StaticPool in-memory database).
        if not s.get(DimTaskType, 1):
            s.add_all([
                DimTaskType(task_type_id=1, task_type_name="Inspection"),
                DimTaskType(task_type_id=2, task_type_name="Replacement"),
                DimPriority(priority_id=1, priority_name="Critical", priority_weight=4),
                DimPriority(priority_id=2, priority_name="High", priority_weight=3),
                DimRiskLevel(risk_level_id=2, risk_level_name="MEDIUM", sort_order=2),
                DimRiskLevel(risk_level_id=3, risk_level_name="HIGH", sort_order=3),
            ])
        if not s.get(DimAsset, "PMP-0001"):
            s.add_all([
                DimAsset(
                    asset_tag="PMP-0001", asset_class="PMP", asset_name="Pump 1",
                    area="Unit-100", install_date="2015-01-01", replace_usd=150_000.0,
                    c_safety=4, c_env=3, c_prod=4, c_cost=2,
                ),
                DimAsset(
                    asset_tag="HX-0001", asset_class="HX", asset_name="Exchanger 1",
                    area="Unit-200", install_date="2012-06-01", replace_usd=250_000.0,
                    c_safety=3, c_env=2, c_prod=3, c_cost=3,
                ),
            ])
        s.commit()


def _insert_run(engine, scenario_id=None, budget=5_000_000.0, label="Test Run"):
    """Insert one dim_run row and return its run_id."""
    with Session(engine) as s:
        run = DimRun(
            run_label=label,
            scenario_id=scenario_id,
            turnaround_date="2026-10-01",
            budget_usd=budget,
            max_mech_hours=15_000.0,
            max_elec_hours=8_000.0,
            max_inst_hours=6_000.0,
            max_civil_hours=2_500.0,
            planning_horizon_days=365,
            solver_status="OPTIMAL",
            solve_time_s=1.2,
            tasks_total=100,
            tasks_selected=60,
            budget_used_usd=4_500_000.0,
            budget_utilisation=0.90,
            total_net_value_usd=9_000_000.0,
            roi_ratio=2.0,
            total_risk_score_reduced=800,
        )
        s.add(run)
        s.commit()
        return run.run_id


def _insert_facts(engine, run_id, selections: dict[str, bool]):
    """
    Insert minimal fact rows for the given {wo_id: selected} mapping.
    All non-keyed fields are filled with harmless defaults.
    """
    _seed_lookups(engine)
    with Session(engine) as s:
        for i, (wo_id, selected) in enumerate(selections.items()):
            asset_tag = "PMP-0001" if i % 2 == 0 else "HX-0001"
            s.add(FactWorkOrderDecision(
                run_id=run_id,
                asset_tag=asset_tag,
                task_type_id=1,
                priority_id=1,
                risk_level_id=2,
                wo_id=wo_id,
                description=f"Task {wo_id}",
                mandatory=False,
                age_days=1000.0,
                estimated_cost_usd=50_000.0,
                mech_hours=8.0, elec_hours=2.0, inst_hours=1.0, civil_hours=0.0,
                total_craft_hours=11.0,
                duration_days=2,
                fitted_beta=2.0, fitted_eta=2000.0,
                failure_prob=0.3, rul_days=500.0,
                consequence_score=3.5,
                likelihood_tier=3, consequence_tier=3,
                risk_score=9,
                deferred_cost_usd=100_000.0,
                net_value_usd=80_000.0 if selected else 20_000.0,
                selected=selected,
                decision="INCLUDE" if selected else "DEFER",
            ))
        s.commit()


# ─── create_scenario ──────────────────────────────────────────────────────────


class TestCreateScenario:
    def test_returns_integer_id(self, engine):
        sid = create_scenario(engine, "Baseline", "alice")
        assert isinstance(sid, int)
        assert sid > 0

    def test_default_status_is_draft(self, engine):
        sid = create_scenario(engine, "Draft Test", "bob")
        s = get_scenario(engine, sid)
        assert s.status == ScenarioStatus.DRAFT

    def test_name_is_stored_exactly(self, engine):
        sid = create_scenario(engine, "15% Budget Cut", "carol")
        s = get_scenario(engine, sid)
        assert s.name == "15% Budget Cut"

    def test_optional_params_default_to_none(self, engine):
        sid = create_scenario(engine, "No Overrides", "dave")
        p = get_scenario_params(engine, sid)
        assert all(v is None for v in p.values())

    def test_params_stored_correctly(self, engine):
        sid = create_scenario(
            engine, "With Budget", "eve",
            budget_usd=4_250_000.0,
            turnaround_date="2026-10-01",
            max_mech_hours=12_000.0,
        )
        p = get_scenario_params(engine, sid)
        assert p["budget_usd"] == 4_250_000.0
        assert p["turnaround_date"] == "2026-10-01"
        assert p["max_mech_hours"] == 12_000.0
        assert p["max_elec_hours"] is None  # not overridden

    def test_description_stored(self, engine):
        sid = create_scenario(engine, "Desc Test", "frank", description="A detailed note.")
        s = get_scenario(engine, sid)
        assert s.description == "A detailed note."

    def test_shared_defaults_true(self, engine):
        sid = create_scenario(engine, "Shared", "grace")
        s = get_scenario(engine, sid)
        assert s.is_shared is True

    def test_private_scenario(self, engine):
        sid = create_scenario(engine, "Private", "hank", is_shared=False)
        s = get_scenario(engine, sid)
        assert s.is_shared is False

    def test_empty_name_raises(self, engine):
        with pytest.raises(ValueError, match="cannot be empty"):
            create_scenario(engine, "   ", "ivan")

    def test_name_too_long_raises(self, engine):
        with pytest.raises(ValueError, match="120 characters"):
            create_scenario(engine, "x" * 121, "ivan")

    def test_name_exactly_120_chars_accepted(self, engine):
        sid = create_scenario(engine, "y" * 120, "ivan")
        assert sid > 0

    def test_duplicate_name_raises_integrity_error(self, engine):
        from sqlalchemy.exc import IntegrityError
        create_scenario(engine, "DupName", "alice")
        with pytest.raises(IntegrityError):
            create_scenario(engine, "DupName", "bob")

    def test_version_starts_at_one(self, engine):
        sid = create_scenario(engine, "v1 test", "alice")
        s = get_scenario(engine, sid)
        assert s.version == 1

    def test_created_at_is_recent(self, engine):
        before = datetime.now(timezone.utc)
        sid = create_scenario(engine, "Timestamp Test", "alice")
        after = datetime.now(timezone.utc)
        s = get_scenario(engine, sid)
        # created_at may be stored as naive UTC — compare without tz if needed
        created = s.created_at
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        assert before <= created <= after


# ─── list_scenarios ───────────────────────────────────────────────────────────


class TestListScenarios:
    def test_returns_dataframe(self, engine):
        df = list_scenarios(engine)
        assert isinstance(df, pd.DataFrame)

    def test_empty_when_no_scenarios(self, engine):
        df = list_scenarios(engine)
        assert len(df) == 0

    def test_lists_created_scenario(self, engine):
        create_scenario(engine, "Scenario A", "alice")
        df = list_scenarios(engine)
        assert len(df) == 1
        assert df.iloc[0]["name"] == "Scenario A"

    def test_excludes_archived_by_default(self, engine):
        sid = create_scenario(engine, "To Archive", "alice")
        archive_scenario(engine, sid, "alice")
        df = list_scenarios(engine)
        assert len(df) == 0

    def test_includes_archived_when_requested(self, engine):
        sid = create_scenario(engine, "Archived One", "alice")
        archive_scenario(engine, sid, "alice")
        df = list_scenarios(engine, include_archived=True)
        assert len(df) == 1

    def test_filter_by_created_by(self, engine):
        create_scenario(engine, "Alice's", "alice")
        create_scenario(engine, "Bob's", "bob")
        df = list_scenarios(engine, created_by="alice")
        assert len(df) == 1
        assert df.iloc[0]["created_by"] == "alice"

    def test_shared_only_filter(self, engine):
        create_scenario(engine, "Public", "alice", is_shared=True)
        create_scenario(engine, "Private", "alice", is_shared=False)
        df = list_scenarios(engine, shared_only=True)
        assert len(df) == 1
        assert df.iloc[0]["name"] == "Public"

    def test_ordered_by_updated_at_desc(self, engine):
        create_scenario(engine, "First", "alice")
        create_scenario(engine, "Second", "alice")
        df = list_scenarios(engine)
        # Most recently updated first
        assert df.iloc[0]["name"] == "Second"

    def test_has_expected_columns(self, engine):
        create_scenario(engine, "ColTest", "alice")
        df = list_scenarios(engine)
        expected = {"scenario_id", "name", "status", "created_by", "budget_usd", "current_run_id"}
        assert expected.issubset(set(df.columns))


# ─── get_scenario ─────────────────────────────────────────────────────────────


class TestGetScenario:
    def test_returns_dim_scenario_instance(self, engine):
        sid = create_scenario(engine, "Getter Test", "alice")
        s = get_scenario(engine, sid)
        assert isinstance(s, DimScenario)

    def test_unknown_id_raises(self, engine):
        with pytest.raises(ScenarioNotFoundError):
            get_scenario(engine, 99999)

    def test_returned_instance_is_detached(self, engine):
        """Attributes must be readable after the session closes."""
        sid = create_scenario(engine, "Detach Test", "alice", budget_usd=3_000_000.0)
        s = get_scenario(engine, sid)
        # If the instance were still bound to a closed session this would fail.
        assert s.budget_usd == 3_000_000.0
        assert s.name == "Detach Test"


# ─── lock_scenario / unlock_scenario ─────────────────────────────────────────


class TestLockUnlock:
    def test_lock_sets_status_and_locked_by(self, engine):
        sid = create_scenario(engine, "Lock Target", "alice")
        lock_scenario(engine, sid, "alice")
        s = get_scenario(engine, sid)
        assert s.status == ScenarioStatus.LOCKED
        assert s.locked_by == "alice"

    def test_lock_increments_version(self, engine):
        sid = create_scenario(engine, "Version Lock", "alice")
        new_v = lock_scenario(engine, sid, "alice")
        assert new_v == 2

    def test_unlock_sets_status_back_to_draft(self, engine):
        sid = create_scenario(engine, "Unlock Target", "alice")
        lock_scenario(engine, sid, "alice")
        unlock_scenario(engine, sid, "alice")
        s = get_scenario(engine, sid)
        assert s.status == ScenarioStatus.DRAFT
        assert s.locked_by is None

    def test_unlock_increments_version(self, engine):
        sid = create_scenario(engine, "Version Unlock", "alice")
        lock_scenario(engine, sid, "alice")
        new_v = unlock_scenario(engine, sid, "alice")
        assert new_v == 3  # create=v1, lock=v2, unlock=v3

    def test_lock_by_different_user_raises(self, engine):
        sid = create_scenario(engine, "Multi-Lock", "alice")
        lock_scenario(engine, sid, "alice")
        with pytest.raises(ScenarioLockedError, match="already locked by"):
            lock_scenario(engine, sid, "bob")

    def test_unlock_by_wrong_user_raises(self, engine):
        sid = create_scenario(engine, "Wrong Unlock", "alice")
        lock_scenario(engine, sid, "alice")
        with pytest.raises(ScenarioLockedError):
            unlock_scenario(engine, sid, "bob")

    def test_unlock_when_not_locked_raises(self, engine):
        sid = create_scenario(engine, "Not Locked", "alice")
        with pytest.raises(ScenarioLockedError, match="not locked"):
            unlock_scenario(engine, sid, "alice")

    def test_lock_archived_scenario_raises(self, engine):
        sid = create_scenario(engine, "Archived Lock", "alice")
        archive_scenario(engine, sid, "alice")
        with pytest.raises(ScenarioArchivedError):
            lock_scenario(engine, sid, "alice")

    def test_same_user_can_relock(self, engine):
        """A user who already holds the lock can call lock again (idempotent from their view)."""
        sid = create_scenario(engine, "Re-lock", "alice")
        lock_scenario(engine, sid, "alice")
        # Same user locking again should succeed (not raise)
        lock_scenario(engine, sid, "alice")

    def test_lock_unknown_scenario_raises(self, engine):
        with pytest.raises(ScenarioNotFoundError):
            lock_scenario(engine, 99999, "alice")

    def test_unlock_unknown_scenario_raises(self, engine):
        """Covers manager.py line 357 — ScenarioNotFoundError in unlock_scenario."""
        with pytest.raises(ScenarioNotFoundError):
            unlock_scenario(engine, 99999, "alice")


# ─── update_scenario ──────────────────────────────────────────────────────────


class TestUpdateScenario:
    def test_update_name(self, engine):
        sid = create_scenario(engine, "Old Name", "alice")
        lock_scenario(engine, sid, "alice")
        update_scenario(engine, sid, "alice", name="New Name")
        s = get_scenario(engine, sid)
        assert s.name == "New Name"

    def test_update_budget(self, engine):
        sid = create_scenario(engine, "Budget Update", "alice", budget_usd=5_000_000.0)
        lock_scenario(engine, sid, "alice")
        update_scenario(engine, sid, "alice", budget_usd=4_250_000.0)
        p = get_scenario_params(engine, sid)
        assert p["budget_usd"] == 4_250_000.0

    def test_clear_field_with_none(self, engine):
        """Passing None explicitly clears a previously set field."""
        sid = create_scenario(engine, "Clear Test", "alice", budget_usd=3_000_000.0)
        lock_scenario(engine, sid, "alice")
        update_scenario(engine, sid, "alice", budget_usd=None)
        p = get_scenario_params(engine, sid)
        assert p["budget_usd"] is None

    def test_ellipsis_fields_not_changed(self, engine):
        """Fields passed as ... (default) must not be overwritten."""
        sid = create_scenario(engine, "Ellipsis Test", "alice",
                              budget_usd=3_000_000.0, turnaround_date="2026-10-01")
        lock_scenario(engine, sid, "alice")
        # Only update name — budget_usd and turnaround_date should be unchanged
        update_scenario(engine, sid, "alice", name="Updated Name")
        p = get_scenario_params(engine, sid)
        assert p["budget_usd"] == 3_000_000.0
        assert p["turnaround_date"] == "2026-10-01"

    def test_update_without_lock_raises(self, engine):
        sid = create_scenario(engine, "No Lock Update", "alice")
        with pytest.raises(ScenarioLockedError):
            update_scenario(engine, sid, "alice", name="New Name")

    def test_update_locked_by_other_raises(self, engine):
        sid = create_scenario(engine, "Other Lock", "alice")
        lock_scenario(engine, sid, "alice")
        with pytest.raises(ScenarioLockedError):
            update_scenario(engine, sid, "bob", name="Stolen Name")

    def test_update_increments_version(self, engine):
        sid = create_scenario(engine, "Version Update", "alice")
        lock_scenario(engine, sid, "alice")
        new_v = update_scenario(engine, sid, "alice", name="Renamed")
        assert new_v == 3  # create=1, lock=2, update=3

    def test_update_with_nothing_returns_current_version(self, engine):
        sid = create_scenario(engine, "No-op Update", "alice")
        lock_scenario(engine, sid, "alice")
        v = update_scenario(engine, sid, "alice")  # no kwargs
        assert v == 2  # no change → no version bump

    def test_empty_name_raises(self, engine):
        sid = create_scenario(engine, "Empty Name Target", "alice")
        lock_scenario(engine, sid, "alice")
        with pytest.raises(ValueError, match="cannot be empty"):
            update_scenario(engine, sid, "alice", name="")

    def test_update_unknown_scenario_raises(self, engine):
        """Covers manager.py line 411 — ScenarioNotFoundError in update_scenario."""
        with pytest.raises(ScenarioNotFoundError):
            update_scenario(engine, 99999, "alice", name="Ghost")

    def test_update_overlong_name_raises(self, engine):
        """Covers manager.py line 424 — name > 120 chars in update_scenario."""
        sid = create_scenario(engine, "Long Name Target", "alice")
        lock_scenario(engine, sid, "alice")
        with pytest.raises(ValueError, match="120 characters"):
            update_scenario(engine, sid, "alice", name="z" * 121)


# ─── archive_scenario ─────────────────────────────────────────────────────────


class TestArchiveScenario:
    def test_sets_status_to_archived(self, engine):
        sid = create_scenario(engine, "Archive Me", "alice")
        archive_scenario(engine, sid, "alice")
        s = get_scenario(engine, sid)
        assert s.status == ScenarioStatus.ARCHIVED

    def test_excluded_from_list_by_default(self, engine):
        sid = create_scenario(engine, "Hidden", "alice")
        archive_scenario(engine, sid, "alice")
        df = list_scenarios(engine)
        assert len(df) == 0

    def test_cannot_lock_archived(self, engine):
        sid = create_scenario(engine, "Lock Archived", "alice")
        archive_scenario(engine, sid, "alice")
        with pytest.raises(ScenarioArchivedError):
            lock_scenario(engine, sid, "alice")

    def test_cannot_archive_another_users_locked_scenario(self, engine):
        sid = create_scenario(engine, "Locked By Alice", "alice")
        lock_scenario(engine, sid, "alice")
        with pytest.raises(ScenarioLockedError):
            archive_scenario(engine, sid, "bob")

    def test_owner_can_archive_their_own_locked_scenario(self, engine):
        """The locker can archive their own locked scenario."""
        sid = create_scenario(engine, "Self Archive", "alice")
        lock_scenario(engine, sid, "alice")
        archive_scenario(engine, sid, "alice")  # should not raise
        s = get_scenario(engine, sid)
        assert s.status == ScenarioStatus.ARCHIVED

    def test_unknown_scenario_raises(self, engine):
        with pytest.raises(ScenarioNotFoundError):
            archive_scenario(engine, 99999, "alice")


# ─── clone_scenario ───────────────────────────────────────────────────────────


class TestCloneScenario:
    def test_clone_returns_new_id(self, engine):
        sid = create_scenario(engine, "Original", "alice", budget_usd=5_000_000.0)
        cid = clone_scenario(engine, sid, "Clone", "bob")
        assert cid != sid
        assert cid > 0

    def test_clone_inherits_params(self, engine):
        sid = create_scenario(engine, "Orig Params", "alice",
                              budget_usd=5_000_000.0, max_mech_hours=12_000.0,
                              turnaround_date="2026-10-01")
        cid = clone_scenario(engine, sid, "Clone Params", "bob")
        p = get_scenario_params(engine, cid)
        assert p["budget_usd"] == 5_000_000.0
        assert p["max_mech_hours"] == 12_000.0
        assert p["turnaround_date"] == "2026-10-01"

    def test_clone_with_budget_cut(self, engine):
        sid = create_scenario(engine, "Full Budget", "alice", budget_usd=5_000_000.0)
        cid = clone_scenario(engine, sid, "Budget -15%", "alice", budget_adjustment_pct=-15.0)
        p = get_scenario_params(engine, cid)
        assert abs(p["budget_usd"] - 4_250_000.0) < 1.0

    def test_clone_with_budget_increase(self, engine):
        sid = create_scenario(engine, "Base Budget", "alice", budget_usd=5_000_000.0)
        cid = clone_scenario(engine, sid, "Budget +10%", "alice", budget_adjustment_pct=10.0)
        p = get_scenario_params(engine, cid)
        assert abs(p["budget_usd"] - 5_500_000.0) < 1.0

    def test_clone_records_parent_id(self, engine):
        sid = create_scenario(engine, "Parent", "alice")
        cid = clone_scenario(engine, sid, "Child", "alice")
        child = get_scenario(engine, cid)
        assert child.parent_scenario_id == sid

    def test_clone_records_budget_adjustment_pct(self, engine):
        sid = create_scenario(engine, "Orig Adj", "alice", budget_usd=4_000_000.0)
        cid = clone_scenario(engine, sid, "Clone Adj", "alice", budget_adjustment_pct=-15.0)
        child = get_scenario(engine, cid)
        assert child.budget_adjustment_pct == -15.0

    def test_clone_starts_as_draft(self, engine):
        sid = create_scenario(engine, "Orig Draft", "alice")
        cid = clone_scenario(engine, sid, "Clone Draft", "alice")
        child = get_scenario(engine, cid)
        assert child.status == ScenarioStatus.DRAFT

    def test_clone_archived_raises(self, engine):
        sid = create_scenario(engine, "Orig Archived", "alice")
        archive_scenario(engine, sid, "alice")
        with pytest.raises(ScenarioArchivedError):
            clone_scenario(engine, sid, "Clone Archived", "alice")

    def test_clone_nonexistent_raises(self, engine):
        with pytest.raises(ScenarioNotFoundError):
            clone_scenario(engine, 99999, "Clone Missing", "alice")

    def test_invalid_budget_adjustment_raises(self, engine):
        sid = create_scenario(engine, "Bad Adj", "alice", budget_usd=5_000_000.0)
        with pytest.raises(ValueError, match="budget_adjustment_pct"):
            clone_scenario(engine, sid, "Bad Clone", "alice", budget_adjustment_pct=-100.0)

    def test_clone_budget_none_source_records_pct_but_no_value(self, engine):
        """When source has no explicit budget, the pct is recorded but budget stays None."""
        sid = create_scenario(engine, "No Budget Source", "alice")  # budget_usd=None
        cid = clone_scenario(engine, sid, "No Budget Clone", "alice", budget_adjustment_pct=-15.0)
        p = get_scenario_params(engine, cid)
        # budget_usd stays None because there's nothing to apply -15% to
        assert p["budget_usd"] is None
        child = get_scenario(engine, cid)
        assert child.budget_adjustment_pct == -15.0


# ─── set_current_run ──────────────────────────────────────────────────────────


class TestSetCurrentRun:
    def test_sets_current_run_id(self, engine):
        sid = create_scenario(engine, "Run Link", "alice")
        run_id = _insert_run(engine, scenario_id=sid)
        set_current_run(engine, sid, run_id)
        s = get_scenario(engine, sid)
        assert s.current_run_id == run_id

    def test_overwrite_with_newer_run(self, engine):
        sid = create_scenario(engine, "Run Overwrite", "alice")
        run1 = _insert_run(engine, scenario_id=sid, label="Run 1")
        run2 = _insert_run(engine, scenario_id=sid, label="Run 2")
        set_current_run(engine, sid, run1)
        set_current_run(engine, sid, run2)
        s = get_scenario(engine, sid)
        assert s.current_run_id == run2

    def test_unknown_scenario_raises(self, engine):
        with pytest.raises(ScenarioNotFoundError):
            set_current_run(engine, 99999, 1)


# ─── _compare_and_swap_update (unit, not via public API) ─────────────────────


class TestCASUpdate:
    def test_correct_version_succeeds(self, engine):
        sid = create_scenario(engine, "CAS Test", "alice")
        with Session(engine) as session:
            _compare_and_swap_update(session, sid, 1, {"description": "Updated"})
            session.commit()
        s = get_scenario(engine, sid)
        assert s.description == "Updated"
        assert s.version == 2

    def test_stale_version_raises_conflict_error(self, engine):
        sid = create_scenario(engine, "CAS Stale", "alice")
        # Advance version to 2 first
        lock_scenario(engine, sid, "alice")  # v1 → v2
        with Session(engine) as session:
            with pytest.raises(ScenarioConflictError, match="version"):
                _compare_and_swap_update(session, sid, 1, {"description": "Stale"})

    def test_unknown_id_raises_not_found(self, engine):
        with Session(engine) as session:
            with pytest.raises(ScenarioNotFoundError):
                _compare_and_swap_update(session, 99999, 1, {"description": "Ghost"})


# ─── Concurrent lock race ─────────────────────────────────────────────────────


class TestConcurrentLockRace:
    def test_only_one_thread_wins_lock(self, engine):
        """
        Two threads simultaneously try to lock the same scenario.
        Exactly one must succeed and exactly one must see ScenarioLockedError
        (or ScenarioConflictError if the CAS race fires first).
        The global state afterwards must be consistent: status=LOCKED, locked_by
        is one of the two planner names.
        """
        sid = create_scenario(engine, "Race Target", "admin")

        wins = []
        errors = []
        barrier = threading.Barrier(2)

        def try_lock(name):
            barrier.wait()  # synchronize both threads at the last moment
            try:
                lock_scenario(engine, sid, name)
                wins.append(name)
            except (ScenarioLockedError, ScenarioConflictError):
                errors.append(name)

        t1 = threading.Thread(target=try_lock, args=("planner-a",))
        t2 = threading.Thread(target=try_lock, args=("planner-b",))
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        assert len(wins) == 1, f"Expected exactly 1 winner; got {wins}"
        assert len(errors) == 1, f"Expected exactly 1 loser; got {errors}"

        s = get_scenario(engine, sid)
        assert s.status == ScenarioStatus.LOCKED
        assert s.locked_by in ("planner-a", "planner-b")

    def test_version_is_consistent_after_race(self, engine):
        """After a lock race the version must be exactly 2 (not 3 due to double write)."""
        sid = create_scenario(engine, "Version Race", "admin")
        barrier = threading.Barrier(2)

        def try_lock(name):
            barrier.wait()
            try:
                lock_scenario(engine, sid, name)
            except (ScenarioLockedError, ScenarioConflictError):
                pass

        threads = [threading.Thread(target=try_lock, args=(f"p{i}",)) for i in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        s = get_scenario(engine, sid)
        assert s.version == 2


# ─── build_config_for_scenario ────────────────────────────────────────────────


class TestBuildConfigForScenario:
    def _base_cfg(self):
        return TurnaroundConfig(
            turnaround_date="2026-10-01",
            total_budget=5_000_000.0,
            max_mech_hours=15_000.0,
            max_elec_hours=8_000.0,
            max_inst_hours=6_000.0,
            max_civil_hours=2_500.0,
        )

    def test_none_fields_inherit_from_base(self, engine):
        sid = create_scenario(engine, "Inherit Base", "alice")
        s = get_scenario(engine, sid)
        cfg = build_config_for_scenario(s, self._base_cfg())
        assert cfg.total_budget == 5_000_000.0
        assert cfg.max_mech_hours == 15_000.0

    def test_overrides_applied_correctly(self, engine):
        sid = create_scenario(engine, "Override Cfg", "alice",
                              budget_usd=4_250_000.0, max_mech_hours=12_000.0,
                              turnaround_date="2027-04-01")
        s = get_scenario(engine, sid)
        cfg = build_config_for_scenario(s, self._base_cfg())
        assert cfg.total_budget == 4_250_000.0
        assert cfg.max_mech_hours == 12_000.0
        assert cfg.turnaround_date == "2027-04-01"

    def test_global_cfg_not_mutated(self, engine):
        """build_config_for_scenario must never mutate the global TA_CFG singleton."""
        from src.utils.config import TA_CFG
        original_budget = TA_CFG.total_budget
        sid = create_scenario(engine, "Mutate Guard", "alice", budget_usd=1.0)
        s = get_scenario(engine, sid)
        build_config_for_scenario(s)  # uses global TA_CFG as base
        assert TA_CFG.total_budget == original_budget

    def test_partial_override_leaves_others_unchanged(self, engine):
        sid = create_scenario(engine, "Partial Override", "alice", budget_usd=3_000_000.0)
        s = get_scenario(engine, sid)
        base = self._base_cfg()
        cfg = build_config_for_scenario(s, base)
        assert cfg.total_budget == 3_000_000.0
        assert cfg.max_mech_hours == base.max_mech_hours  # unchanged

    def test_none_budget_does_not_zero_it_out(self, engine):
        """Regression: a None budget_usd must NOT produce total_budget=0."""
        sid = create_scenario(engine, "Null Budget Guard", "alice")  # budget_usd=None
        s = get_scenario(engine, sid)
        base = self._base_cfg()
        cfg = build_config_for_scenario(s, base)
        assert cfg.total_budget == base.total_budget  # inherited, not zeroed

    def test_all_five_fields_overridden(self, engine):
        """
        Setting all five resource fields ensures the three elec/inst/civil
        assignment branches (runner.py lines 72, 74, 76) are exercised.
        """
        sid = create_scenario(
            engine, "All Five Fields", "alice",
            budget_usd=3_500_000.0,
            turnaround_date="2027-01-01",
            max_mech_hours=10_000.0,
            max_elec_hours=5_000.0,   # runner.py line 72
            max_inst_hours=4_000.0,   # runner.py line 74
            max_civil_hours=1_500.0,  # runner.py line 76
        )
        s = get_scenario(engine, sid)
        cfg = build_config_for_scenario(s, self._base_cfg())
        assert cfg.total_budget == 3_500_000.0
        assert cfg.max_elec_hours == 5_000.0
        assert cfg.max_inst_hours == 4_000.0
        assert cfg.max_civil_hours == 1_500.0


# ─── queries — list_scenario_runs / get_scenario_facts / kpi_history ─────────


class TestScenarioQueries:
    def test_list_scenario_runs_empty(self, engine):
        sid = create_scenario(engine, "No Runs", "alice")
        df = list_scenario_runs(engine, sid)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_list_scenario_runs_returns_tagged_runs(self, engine):
        sid = create_scenario(engine, "Tagged Runs", "alice")
        run1 = _insert_run(engine, scenario_id=sid, label="R1")
        run2 = _insert_run(engine, scenario_id=sid, label="R2")
        _insert_run(engine, scenario_id=None, label="Untagged")
        df = list_scenario_runs(engine, sid)
        assert len(df) == 2
        assert set(df["run_id"]) == {run1, run2}

    def test_list_scenario_runs_ordered_desc(self, engine):
        sid = create_scenario(engine, "Order Runs", "alice")
        _insert_run(engine, scenario_id=sid, label="First")
        run2 = _insert_run(engine, scenario_id=sid, label="Second")
        df = list_scenario_runs(engine, sid)
        assert df.iloc[0]["run_id"] == run2  # most recent first

    def test_get_scenario_facts_raises_when_not_solved(self, engine):
        sid = create_scenario(engine, "Unsolved Facts", "alice")
        with pytest.raises(KeyError, match="not been solved"):
            get_scenario_facts(engine, sid)

    def test_get_scenario_facts_returns_run_facts(self, engine):
        sid = create_scenario(engine, "Facts Query", "alice")
        run_id = _insert_run(engine, scenario_id=sid)
        _insert_facts(engine, run_id, {"WO-001": True, "WO-002": False})
        set_current_run(engine, sid, run_id)
        df = get_scenario_facts(engine, sid)
        assert len(df) == 2

    def test_get_scenario_facts_unknown_id_raises(self, engine):
        with pytest.raises(KeyError):
            get_scenario_facts(engine, 99999)

    def test_scenario_kpi_history_empty(self, engine):
        sid = create_scenario(engine, "History Empty", "alice")
        df = scenario_kpi_history(engine, sid)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_scenario_kpi_history_grows_with_runs(self, engine):
        sid = create_scenario(engine, "History Grows", "alice")
        _insert_run(engine, scenario_id=sid, label="Solve 1")
        _insert_run(engine, scenario_id=sid, label="Solve 2")
        df = scenario_kpi_history(engine, sid)
        assert len(df) == 2

    def test_scenario_kpi_history_has_kpi_columns(self, engine):
        sid = create_scenario(engine, "History KPIs", "alice")
        _insert_run(engine, scenario_id=sid)
        df = scenario_kpi_history(engine, sid)
        expected = {"run_id", "tasks_selected", "budget_used_usd", "roi_ratio"}
        assert expected.issubset(set(df.columns))


# ─── compare_scenarios ────────────────────────────────────────────────────────


class TestCompareScenarios:
    @pytest.fixture
    def two_solved_scenarios(self, engine):
        """
        Scenario A ($5M budget): selects WO-001, WO-002, WO-003
        Scenario B ($4.25M budget): selects WO-001, WO-002, WO-004 (drops 003, adds 004)
        """
        _seed_lookups(engine)
        sid_a = create_scenario(engine, "Scenario A", "alice", budget_usd=5_000_000.0)
        sid_b = create_scenario(engine, "Scenario B", "alice", budget_usd=4_250_000.0)

        run_a = _insert_run(engine, scenario_id=sid_a, budget=5_000_000.0, label="Scenario A")
        _insert_facts(engine, run_a, {
            "WO-001": True, "WO-002": True, "WO-003": True, "WO-004": False,
        })
        set_current_run(engine, sid_a, run_a)

        run_b = _insert_run(engine, scenario_id=sid_b, budget=4_250_000.0, label="Scenario B")
        _insert_facts(engine, run_b, {
            "WO-001": True, "WO-002": True, "WO-003": False, "WO-004": True,
        })
        set_current_run(engine, sid_b, run_b)

        return engine, sid_a, sid_b

    def test_returns_scenario_comparison(self, two_solved_scenarios):
        engine, sid_a, sid_b = two_solved_scenarios
        result = compare_scenarios(engine, sid_a, sid_b)
        assert isinstance(result, ScenarioComparison)

    def test_kpis_has_two_rows(self, two_solved_scenarios):
        engine, sid_a, sid_b = two_solved_scenarios
        result = compare_scenarios(engine, sid_a, sid_b)
        assert len(result.kpis) == 2

    def test_kpis_scenario_labels_correct(self, two_solved_scenarios):
        engine, sid_a, sid_b = two_solved_scenarios
        result = compare_scenarios(engine, sid_a, sid_b)
        labels = set(result.kpis["scenario_label"])
        assert "Scenario A" in labels
        assert "Scenario B" in labels

    def test_added_contains_wo004(self, two_solved_scenarios):
        engine, sid_a, sid_b = two_solved_scenarios
        result = compare_scenarios(engine, sid_a, sid_b)
        assert "WO-004" in result.added["wo_id"].values

    def test_removed_contains_wo003(self, two_solved_scenarios):
        engine, sid_a, sid_b = two_solved_scenarios
        result = compare_scenarios(engine, sid_a, sid_b)
        assert "WO-003" in result.removed["wo_id"].values

    def test_common_in_contains_wo001_and_wo002(self, two_solved_scenarios):
        engine, sid_a, sid_b = two_solved_scenarios
        result = compare_scenarios(engine, sid_a, sid_b)
        common_ids = set(result.common_in["wo_id"].values)
        assert "WO-001" in common_ids
        assert "WO-002" in common_ids

    def test_n_added_and_n_removed_properties(self, two_solved_scenarios):
        engine, sid_a, sid_b = two_solved_scenarios
        result = compare_scenarios(engine, sid_a, sid_b)
        assert result.n_added == 1
        assert result.n_removed == 1

    def test_delta_budget_negative_for_budget_cut(self, two_solved_scenarios):
        engine, sid_a, sid_b = two_solved_scenarios
        result = compare_scenarios(engine, sid_a, sid_b)
        assert result.delta["delta_budget_usd"] < 0  # B has lower budget than A

    def test_summary_text_contains_scenario_names(self, two_solved_scenarios):
        engine, sid_a, sid_b = two_solved_scenarios
        result = compare_scenarios(engine, sid_a, sid_b)
        summary = result.summary_text()
        assert "Scenario A" in summary
        assert "Scenario B" in summary

    def test_unsolved_scenario_a_raises(self, engine):
        _seed_lookups(engine)
        sid_a = create_scenario(engine, "Unsolved A", "alice")
        sid_b = create_scenario(engine, "Solved B", "alice")
        run_b = _insert_run(engine, scenario_id=sid_b)
        _insert_facts(engine, run_b, {"WO-X": True})
        set_current_run(engine, sid_b, run_b)
        with pytest.raises(ScenarioNotSolvedError, match="never been solved"):
            compare_scenarios(engine, sid_a, sid_b)

    def test_unsolved_scenario_b_raises(self, engine):
        _seed_lookups(engine)
        sid_a = create_scenario(engine, "Solved A2", "alice")
        sid_b = create_scenario(engine, "Unsolved B2", "alice")
        run_a = _insert_run(engine, scenario_id=sid_a)
        _insert_facts(engine, run_a, {"WO-Y": True})
        set_current_run(engine, sid_a, run_a)
        with pytest.raises(ScenarioNotSolvedError):
            compare_scenarios(engine, sid_a, sid_b)


# ─── compare_many_scenarios ──────────────────────────────────────────────────


class TestCompareManyScenarios:
    def test_returns_one_row_per_scenario(self, engine):
        _seed_lookups(engine)
        ids = []
        for name in ["S1", "S2", "S3"]:
            sid = create_scenario(engine, name, "alice")
            run_id = _insert_run(engine, scenario_id=sid, label=name)
            _insert_facts(engine, run_id, {"WO-" + name: True})
            set_current_run(engine, sid, run_id)
            ids.append(sid)
        df = compare_many_scenarios(engine, ids)
        assert len(df) == 3

    def test_unsolved_scenario_raises(self, engine):
        sid = create_scenario(engine, "Many Unsolved", "alice")
        with pytest.raises(ScenarioNotSolvedError):
            compare_many_scenarios(engine, [sid])

    def test_has_required_columns(self, engine):
        _seed_lookups(engine)
        sid = create_scenario(engine, "Multi KPI", "alice")
        run_id = _insert_run(engine, scenario_id=sid)
        _insert_facts(engine, run_id, {"WO-K1": True})
        set_current_run(engine, sid, run_id)
        df = compare_many_scenarios(engine, [sid])
        assert "scenario_id" in df.columns
        assert "tasks_selected" in df.columns
        assert "roi_ratio" in df.columns
