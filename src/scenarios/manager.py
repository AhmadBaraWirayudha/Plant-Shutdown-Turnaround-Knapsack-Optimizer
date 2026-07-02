"""
manager.py — Scenario lifecycle: create, list, lock, unlock, update, clone, archive.

Overview
--------
A "scenario" is a saved, named, collaborative planning container — e.g.
"Standard Budget" or "15% Budget Cut" — that wraps a specific set of
optimizer parameters and links to every run that was executed under it.

This module owns the complete lifecycle of ``DimScenario`` rows.  All writes
use an optimistic-concurrency pattern (``version`` column) so that two
planners attempting to lock the same scenario simultaneously, or two API
requests racing to save an edit, always get a clean conflict error rather
than silently clobbering each other.

Concurrency model (re-stated from schema.py)
--------------------------------------------
Two mechanisms work together:

1. **Pessimistic / human-facing lock** — ``status=LOCKED``, ``locked_by``,
   ``locked_at``.  This tells a second planner *who* is editing and *since
   when*, so they can decide whether to wait or contact that person.  It is
   NOT a database-level lock; it is purely advisory and enforced at the
   application layer.

2. **Optimistic-concurrency token** — ``version`` (int).  Every write that
   changes a scenario goes through ``_compare_and_swap_update()``, which
   issues a single ``UPDATE ... WHERE scenario_id = :id AND version = :v``.
   If no row is updated (because another writer already incremented ``version``
   in between), the function raises ``ScenarioConflictError`` rather than
   silently overwriting.  This closes the classic check-then-act TOCTOU race
   that the human-facing lock alone cannot prevent.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd
from sqlalchemy import Engine, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from src.db.schema import Base, DimScenario, ScenarioStatus
from src.utils.helpers import get_logger

log = get_logger("scenarios.manager")


# ─── Custom exceptions ────────────────────────────────────────────────────────


class ScenarioNotFoundError(KeyError):
    """Raised when a scenario_id does not exist in the database."""


class ScenarioLockedError(RuntimeError):
    """
    Raised when an operation requires the scenario to be unlocked (DRAFT) but
    it is currently LOCKED by someone else, or vice-versa.
    """


class ScenarioConflictError(RuntimeError):
    """
    Raised by ``_compare_and_swap_update`` when the optimistic-concurrency
    check fails — another writer modified the row between our read and write.
    The caller should re-read the scenario and retry if appropriate.
    """


class ScenarioArchivedError(RuntimeError):
    """Raised when an operation is attempted on an ARCHIVED scenario."""


# ─── Schema initialisation ────────────────────────────────────────────────────


def init_scenario_tables(engine: Engine) -> None:
    """Ensure ``dim_scenario`` (and all other star-schema tables) exist."""
    Base.metadata.create_all(engine)


# ─── Core CAS helper ─────────────────────────────────────────────────────────


def _compare_and_swap_update(
    session: Session,
    scenario_id: int,
    expected_version: int,
    updates: dict[str, Any],
) -> None:
    """
    Update ``DimScenario`` row atomically, incrementing ``version``.

    Issues a single ``UPDATE dim_scenario SET ... WHERE scenario_id = :id
    AND version = :expected_version``.  If the row count is 0 (another writer
    already changed the row), raises ``ScenarioConflictError``.  If the row
    does not exist at all, raises ``ScenarioNotFoundError``.

    This is intentionally NOT exposed as a public API — all mutation helpers
    in this module call it internally.
    """
    updates_with_version = dict(updates)
    updates_with_version["version"] = expected_version + 1
    updates_with_version["updated_at"] = datetime.now(timezone.utc)

    stmt = (
        update(DimScenario)
        .where(DimScenario.scenario_id == scenario_id)
        .where(DimScenario.version == expected_version)
        .values(**updates_with_version)
    )
    result = session.execute(stmt)

    if result.rowcount == 0:
        # Distinguish "scenario does not exist" from "version mismatch".
        exists = session.scalars(
            select(DimScenario.scenario_id).where(DimScenario.scenario_id == scenario_id)
        ).first()
        if exists is None:
            raise ScenarioNotFoundError(f"Scenario {scenario_id} not found")
        raise ScenarioConflictError(
            f"Scenario {scenario_id} was modified by another writer "
            f"(expected version {expected_version}). Re-read and retry."
        )


# ─── Public API ───────────────────────────────────────────────────────────────


def create_scenario(
    engine: Engine,
    name: str,
    created_by: str,
    *,
    description: str | None = None,
    turnaround_date: str | None = None,
    budget_usd: float | None = None,
    max_mech_hours: float | None = None,
    max_elec_hours: float | None = None,
    max_inst_hours: float | None = None,
    max_civil_hours: float | None = None,
    is_shared: bool = True,
) -> int:
    """
    Create a new DRAFT scenario and return its ``scenario_id``.

    Parameters
    ----------
    engine
        SQLAlchemy engine pointing at the star-schema database.
    name
        Human-readable scenario name — must be unique across all non-archived
        scenarios (enforced by the ``uq_scenario_name`` constraint).
    created_by
        Username or identifier of the planner creating the scenario.
    description
        Optional free-text notes describing the scenario's purpose.
    turnaround_date, budget_usd, max_*_hours
        Optimizer parameter overrides.  ``None`` means "use the process-wide
        default from ``TA_CFG``" at solve time.
    is_shared
        ``True`` (default) → visible to all planners.
        ``False`` → private draft; only the creator sees it in ``list_scenarios``.

    Returns
    -------
    int
        The auto-generated ``scenario_id``.

    Raises
    ------
    ValueError
        If ``name`` is empty or exceeds 120 characters.
    IntegrityError
        If a scenario with the same ``name`` already exists (unique constraint).
    """
    name = name.strip()
    if not name:
        raise ValueError("Scenario name cannot be empty.")
    if len(name) > 120:
        raise ValueError(f"Scenario name must be ≤ 120 characters (got {len(name)}).")

    init_scenario_tables(engine)

    scenario = DimScenario(
        name=name,
        description=description,
        created_by=created_by,
        status=ScenarioStatus.DRAFT,
        is_shared=is_shared,
        turnaround_date=turnaround_date,
        budget_usd=budget_usd,
        max_mech_hours=max_mech_hours,
        max_elec_hours=max_elec_hours,
        max_inst_hours=max_inst_hours,
        max_civil_hours=max_civil_hours,
    )
    with Session(engine) as session:
        try:
            session.add(scenario)
            session.commit()
            scenario_id = scenario.scenario_id
        except IntegrityError:
            session.rollback()
            raise
    log.info("Created scenario %d: %r (owner=%s)", scenario_id, name, created_by)
    return scenario_id


def list_scenarios(
    engine: Engine,
    *,
    include_archived: bool = False,
    created_by: str | None = None,
    shared_only: bool = False,
) -> pd.DataFrame:
    """
    Return a summary DataFrame of saved scenarios, most recently updated first.

    Columns: scenario_id, name, description, created_by, created_at,
    updated_at, status, locked_by, is_shared, budget_usd, turnaround_date,
    current_run_id, parent_scenario_id, budget_adjustment_pct.
    """
    init_scenario_tables(engine)
    stmt = select(DimScenario).order_by(DimScenario.updated_at.desc())
    if not include_archived:
        stmt = stmt.where(DimScenario.status != ScenarioStatus.ARCHIVED)
    if created_by is not None:
        stmt = stmt.where(DimScenario.created_by == created_by)
    if shared_only:
        stmt = stmt.where(DimScenario.is_shared.is_(True))

    with Session(engine) as session:
        scenarios = session.scalars(stmt).all()

    rows = []
    for s in scenarios:
        rows.append(
            {
                "scenario_id": s.scenario_id,
                "name": s.name,
                "description": s.description,
                "created_by": s.created_by,
                "created_at": s.created_at,
                "updated_at": s.updated_at,
                "status": s.status,
                "locked_by": s.locked_by,
                "is_shared": s.is_shared,
                "budget_usd": s.budget_usd,
                "turnaround_date": s.turnaround_date,
                "current_run_id": s.current_run_id,
                "parent_scenario_id": s.parent_scenario_id,
                "budget_adjustment_pct": s.budget_adjustment_pct,
                "version": s.version,
            }
        )
    return pd.DataFrame(rows)


def get_scenario(engine: Engine, scenario_id: int) -> DimScenario:
    """
    Return the ``DimScenario`` ORM instance for ``scenario_id``.

    Raises ``ScenarioNotFoundError`` if the ID does not exist.

    Note: the returned instance is detached from its session — read its
    attributes freely, but do not add it to another session or use lazy
    relationships.  Use ``get_scenario_params`` if you only need the
    parameter dict.
    """
    init_scenario_tables(engine)
    with Session(engine) as session:
        s = session.get(DimScenario, scenario_id)
        if s is None:
            raise ScenarioNotFoundError(f"Scenario {scenario_id} not found")
        # Expunge before session closes so attributes are readable outside.
        session.expunge(s)
        return s


def get_scenario_params(engine: Engine, scenario_id: int) -> dict[str, Any]:
    """
    Return a plain dict of the optimizer parameter overrides for a scenario.
    ``None`` values mean "inherit from TA_CFG at solve time".
    """
    s = get_scenario(engine, scenario_id)
    return {
        "turnaround_date": s.turnaround_date,
        "budget_usd": s.budget_usd,
        "max_mech_hours": s.max_mech_hours,
        "max_elec_hours": s.max_elec_hours,
        "max_inst_hours": s.max_inst_hours,
        "max_civil_hours": s.max_civil_hours,
    }


def lock_scenario(engine: Engine, scenario_id: int, locked_by: str) -> int:
    """
    Acquire an advisory lock on the scenario.

    Returns the new ``version`` number on success.

    Raises
    ------
    ScenarioNotFoundError  — scenario does not exist.
    ScenarioLockedError    — already locked by someone else.
    ScenarioArchivedError  — cannot lock an archived scenario.
    ScenarioConflictError  — optimistic-concurrency race; re-read and retry.
    """
    with Session(engine) as session:
        s = session.get(DimScenario, scenario_id)
        if s is None:
            raise ScenarioNotFoundError(f"Scenario {scenario_id} not found")
        if s.status == ScenarioStatus.ARCHIVED:
            raise ScenarioArchivedError(f"Scenario {scenario_id} is archived and cannot be locked.")
        if s.status == ScenarioStatus.LOCKED and s.locked_by != locked_by:
            raise ScenarioLockedError(
                f"Scenario {scenario_id} is already locked by {s.locked_by!r} "
                f"since {s.locked_at}. Cannot acquire lock for {locked_by!r}."
            )
        current_version = s.version
        _compare_and_swap_update(
            session,
            scenario_id,
            current_version,
            {
                "status": ScenarioStatus.LOCKED,
                "locked_by": locked_by,
                "locked_at": datetime.now(timezone.utc),
            },
        )
        session.commit()
        new_version = current_version + 1
    log.info("Scenario %d locked by %r (v%d → v%d)", scenario_id, locked_by, current_version, new_version)
    return new_version


def unlock_scenario(engine: Engine, scenario_id: int, locked_by: str) -> int:
    """
    Release an advisory lock.  Only the planner who acquired the lock can
    release it.

    Returns the new ``version`` number on success.

    Raises
    ------
    ScenarioNotFoundError  — scenario does not exist.
    ScenarioLockedError    — not locked by ``locked_by`` (or not locked at all).
    ScenarioConflictError  — optimistic-concurrency race; re-read and retry.
    """
    with Session(engine) as session:
        s = session.get(DimScenario, scenario_id)
        if s is None:
            raise ScenarioNotFoundError(f"Scenario {scenario_id} not found")
        if s.status != ScenarioStatus.LOCKED:
            raise ScenarioLockedError(
                f"Scenario {scenario_id} is not locked (status={s.status!r})."
            )
        if s.locked_by != locked_by:
            raise ScenarioLockedError(
                f"Scenario {scenario_id} is locked by {s.locked_by!r}, not {locked_by!r}."
            )
        current_version = s.version
        _compare_and_swap_update(
            session,
            scenario_id,
            current_version,
            {"status": ScenarioStatus.DRAFT, "locked_by": None, "locked_at": None},
        )
        session.commit()
        new_version = current_version + 1
    log.info("Scenario %d unlocked by %r (v%d)", scenario_id, locked_by, new_version)
    return new_version


def update_scenario(
    engine: Engine,
    scenario_id: int,
    locked_by: str,
    *,
    name: str | None = None,
    description: str | None = ...,  # type: ignore[assignment]
    turnaround_date: str | None = ...,  # type: ignore[assignment]
    budget_usd: float | None = ...,  # type: ignore[assignment]
    max_mech_hours: float | None = ...,  # type: ignore[assignment]
    max_elec_hours: float | None = ...,  # type: ignore[assignment]
    max_inst_hours: float | None = ...,  # type: ignore[assignment]
    max_civil_hours: float | None = ...,  # type: ignore[assignment]
) -> int:
    """
    Update editable fields on a scenario that is currently locked by ``locked_by``.

    ``...`` (Ellipsis) as a default means "do not change this field".
    ``None`` means "clear this field (use the process-wide default)".

    Returns the new ``version`` number on success.

    Raises
    ------
    ScenarioNotFoundError  — scenario does not exist.
    ScenarioLockedError    — not locked by ``locked_by``.
    ScenarioConflictError  — race condition; re-read and retry.
    ValueError             — invalid ``name``.
    """
    with Session(engine) as session:
        s = session.get(DimScenario, scenario_id)
        if s is None:
            raise ScenarioNotFoundError(f"Scenario {scenario_id} not found")
        if s.status != ScenarioStatus.LOCKED or s.locked_by != locked_by:
            raise ScenarioLockedError(
                f"Scenario {scenario_id} must be locked by {locked_by!r} to update. "
                f"Current status: {s.status}, locked_by: {s.locked_by!r}."
            )

        updates: dict[str, Any] = {}
        if name is not None:
            name = name.strip()
            if not name:
                raise ValueError("Scenario name cannot be empty.")
            if len(name) > 120:
                raise ValueError(f"Scenario name must be ≤ 120 characters (got {len(name)}).")
            updates["name"] = name
        # For optional nullable fields, only include them if the caller passed
        # an explicit value (not Ellipsis).
        for attr, value in [
            ("description", description),
            ("turnaround_date", turnaround_date),
            ("budget_usd", budget_usd),
            ("max_mech_hours", max_mech_hours),
            ("max_elec_hours", max_elec_hours),
            ("max_inst_hours", max_inst_hours),
            ("max_civil_hours", max_civil_hours),
        ]:
            if value is not ...:
                updates[attr] = value  # type: ignore[index]

        if not updates:
            log.info("Scenario %d update: nothing to change", scenario_id)
            return s.version

        current_version = s.version
        _compare_and_swap_update(session, scenario_id, current_version, updates)
        session.commit()
        new_version = current_version + 1
    log.info("Scenario %d updated by %r (v%d → v%d)", scenario_id, locked_by, current_version, new_version)
    return new_version


def archive_scenario(engine: Engine, scenario_id: int, archived_by: str) -> None:
    """
    Mark a scenario as ARCHIVED (soft delete).

    Archived scenarios are excluded from ``list_scenarios`` by default and
    cannot be locked or updated.  Archiving is permanent within this API —
    there is no unarchive, because a re-activated scenario would silently
    re-enter any running planner's scenario lists and comparison views.

    Raises
    ------
    ScenarioNotFoundError  — scenario does not exist.
    ScenarioLockedError    — cannot archive a scenario locked by someone else.
    ScenarioConflictError  — race condition; re-read and retry.
    """
    with Session(engine) as session:
        s = session.get(DimScenario, scenario_id)
        if s is None:
            raise ScenarioNotFoundError(f"Scenario {scenario_id} not found")
        if s.status == ScenarioStatus.LOCKED and s.locked_by != archived_by:
            raise ScenarioLockedError(
                f"Scenario {scenario_id} is locked by {s.locked_by!r}. "
                f"It must be unlocked before {archived_by!r} can archive it."
            )
        current_version = s.version
        _compare_and_swap_update(
            session,
            scenario_id,
            current_version,
            {"status": ScenarioStatus.ARCHIVED, "locked_by": None, "locked_at": None},
        )
        session.commit()
    log.info("Scenario %d archived by %r", scenario_id, archived_by)


def clone_scenario(
    engine: Engine,
    source_scenario_id: int,
    new_name: str,
    created_by: str,
    *,
    budget_adjustment_pct: float | None = None,
    description: str | None = None,
) -> int:
    """
    Clone a scenario, optionally adjusting the budget by a percentage.

    The clone starts as a DRAFT with all parameters copied from the source.
    If ``budget_adjustment_pct`` is supplied (e.g. ``-15.0`` for a 15% cut),
    the clone's ``budget_usd`` is scaled accordingly — but only if the source
    has an explicit ``budget_usd`` (if the source inherits from TA_CFG, the
    percentage is recorded descriptively but cannot be applied without knowing
    the TA_CFG value, which belongs to the solve-time config).

    Parameters
    ----------
    source_scenario_id
        The ID of the scenario to clone.
    new_name
        Name for the new scenario (must be unique).
    created_by
        Username of the planner creating the clone.
    budget_adjustment_pct
        Signed percentage change to apply to ``budget_usd``.
        -15.0 → 15% budget cut.  +10.0 → 10% budget increase.
    description
        Override description for the clone.  Defaults to a descriptive
        string that records the source and adjustment.

    Returns
    -------
    int
        The new ``scenario_id``.

    Raises
    ------
    ScenarioNotFoundError  — source scenario does not exist.
    ScenarioArchivedError  — cannot clone an archived scenario.
    ValueError             — invalid ``new_name`` or ``budget_adjustment_pct``.
    """
    if budget_adjustment_pct is not None and not (-100.0 < budget_adjustment_pct < 500.0):
        raise ValueError(
            f"budget_adjustment_pct must be between -100% and +500% (got {budget_adjustment_pct})."
        )

    source = get_scenario(engine, source_scenario_id)
    if source.status == ScenarioStatus.ARCHIVED:
        raise ScenarioArchivedError(
            f"Cannot clone archived scenario {source_scenario_id} ({source.name!r})."
        )

    new_budget = source.budget_usd
    if budget_adjustment_pct is not None and new_budget is not None:
        new_budget = round(new_budget * (1 + budget_adjustment_pct / 100.0), 2)

    if description is None:
        pct_str = f" ({budget_adjustment_pct:+.1f}% budget)" if budget_adjustment_pct else ""
        description = f"Cloned from '{source.name}' (id={source_scenario_id}){pct_str}."

    new_id = create_scenario(
        engine,
        new_name,
        created_by,
        description=description,
        turnaround_date=source.turnaround_date,
        budget_usd=new_budget,
        max_mech_hours=source.max_mech_hours,
        max_elec_hours=source.max_elec_hours,
        max_inst_hours=source.max_inst_hours,
        max_civil_hours=source.max_civil_hours,
        is_shared=source.is_shared,
    )

    # Back-fill parent linkage and budget adjustment % (informational only).
    with Session(engine) as session:
        new_s = session.get(DimScenario, new_id)
        if new_s:
            new_s.parent_scenario_id = source_scenario_id
            new_s.budget_adjustment_pct = budget_adjustment_pct
            session.commit()

    log.info(
        "Cloned scenario %d → %d: %r (adj=%s%%)",
        source_scenario_id,
        new_id,
        new_name,
        budget_adjustment_pct,
    )
    return new_id


def set_current_run(engine: Engine, scenario_id: int, run_id: int) -> None:
    """
    Record that ``run_id`` is the most recent solve for this scenario.

    Called exclusively by ``src/scenarios/runner.py`` immediately after a
    successful ``write_results_to_db`` — see the schema.py docstring for why
    this is the ONLY place that sets ``current_run_id``.
    """
    with Session(engine) as session:
        s = session.get(DimScenario, scenario_id)
        if s is None:
            raise ScenarioNotFoundError(f"Scenario {scenario_id} not found")
        s.current_run_id = run_id
        s.updated_at = datetime.now(timezone.utc)
        session.commit()
    log.debug("Scenario %d: current_run_id set to %d", scenario_id, run_id)
