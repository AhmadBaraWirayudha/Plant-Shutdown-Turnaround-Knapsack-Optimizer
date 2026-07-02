"""
schema.py — Star schema for the turnaround optimizer database.

Design
------
This is a textbook star schema, not a dump of flat pandas DataFrames into
tables. The distinction matters because Power BI (and any BI tool) performs
dramatically better, and is far easier to build correct DAX measures
against, when it can establish 1-to-many relationships between a single
fact table and a handful of small dimension tables — rather than working
against one wide denormalized sheet.

    dim_scenario ──► dim_run ──┐
                     dim_asset ─┼──► fact_work_order_decision
                     dim_task_type ┤
                     dim_priority ──┤
                     dim_risk_level ┘

`dim_run` is the key design choice that makes this genuinely useful in
Power BI rather than a one-time data dump: every execution of the optimizer
(e.g. each budget scenario in the sensitivity sweep from
`notebooks/03_optimization_walkthrough.ipynb`) is persisted as its own row,
so a Power BI report can slice by run/scenario and build a live "budget
sensitivity" page driven by real stored history instead of a static chart.

`dim_scenario` sits one level above `dim_run` (see "Scenario vs. Run" below)
and is what turns this from "every CLI invocation happens to land in the
same database" into actual multi-planner collaboration: a named, owned,
lockable container that planners save, share, and revisit, which a Power BI
report can also slice by — independently of, or in combination with, the
existing per-run `dim_run` slicer.

Grain of the fact table: one row per (run_id, wo_id) — the same work order
can legitimately appear across multiple runs with different selected /
cost / risk values if the scenario inputs changed between runs.

Scenario vs. Run
-----------------
`dim_run` is an immutable, append-only execution record — every solve adds
one row and existing rows are never edited. `dim_scenario` is the mutable,
collaborative container planners actually work in: it can exist before
ever being solved (a draft with chosen parameters, no results yet), gets
re-solved multiple times as planners iterate (each solve appends a new
`dim_run` row tagged with `DimRun.scenario_id`), and can be locked while
one planner is mid-edit. `DimScenario.current_run_id` always points at the
most recent solve, which is what "compare Scenario A vs. Scenario B"
actually compares. See `src/scenarios/manager.py` for the save/share/lock
lifecycle and `src/scenarios/comparison.py` for the side-by-side diff.

`DimScenario.current_run_id` is intentionally a plain Integer column, NOT a
SQL-level ForeignKey — `dim_run.scenario_id` already points the other way
(scenario → its runs), and a FK back from `dim_scenario.current_run_id` to
`dim_run.run_id` would make the two tables mutually dependent. Postgres/
MySQL/SQL Server can resolve that with `ALTER TABLE ... ADD CONSTRAINT`
after both tables exist, but SQLite's ALTER TABLE cannot add a foreign-key
constraint to an existing table at all — and this project's whole premise
is one schema that behaves identically on every supported backend (see
connection.py). The invariant ("current_run_id always names a dim_run row
that has this scenario's scenario_id") is instead guaranteed by
construction: it is set in exactly one place (`src/scenarios/runner.py`,
immediately after `write_results_to_db` returns the new run's id) and
nowhere else.
"""

from __future__ import annotations
from datetime import datetime, timezone

from sqlalchemy import (
    String,
    Float,
    Integer,
    Boolean,
    DateTime,
    ForeignKey,
    UniqueConstraint,
    Index,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


# ─── Dimension: Scenario (the collaborative planning container) ──────────────


class ScenarioStatus:
    """
    String constants for `DimScenario.status`.

    Kept as plain validated strings rather than a native SQL ENUM type:
    SQLite has no ENUM type at all, and this project targets Postgres/
    MySQL/SQL Server too (see connection.py) — a portable `String` column
    validated at the single application layer that writes it
    (`src/scenarios/manager.py`) avoids three different ENUM dialects.
    """

    DRAFT = "DRAFT"
    LOCKED = "LOCKED"
    ARCHIVED = "ARCHIVED"
    ALL = (DRAFT, LOCKED, ARCHIVED)


class DimScenario(Base):
    """
    A saved, named, collaborative planning scenario — e.g. "Standard
    Budget" or "15% Budget Cut" — see the "Scenario vs. Run" note in this
    module's docstring for how this relates to `DimRun`.

    Two independent concurrency mechanisms protect two independent things:
      - `status` / `locked_by` / `locked_at` is a PESSIMISTIC, human-facing
        lock: "Planner X is actively editing this scenario right now."
        It exists purely for UX — so a second planner gets an immediate,
        friendly "locked by X since 14:02" message instead of silently
        clobbering work in progress.
      - `version` is an OPTIMISTIC concurrency token, incremented on every
        update. It protects the lock-acquisition and parameter-update
        operations THEMSELVES from being raced — two requests trying to
        lock, or two trying to save an edit, at the same instant — which
        the human-facing flag alone cannot prevent, since checking
        `status` and then writing it is itself a check-then-act race. See
        `src/scenarios/manager.py::_compare_and_swap_update` for the
        single-statement UPDATE ... WHERE version = :expected that makes
        this safe under real concurrent access, including across multiple
        processes (not just threads) sharing one database.
    """

    __tablename__ = "dim_scenario"
    __table_args__ = (UniqueConstraint("name", name="uq_scenario_name"),)

    scenario_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(120))
    description: Mapped[str | None] = mapped_column(String(1000), nullable=True)

    created_by: Mapped[str] = mapped_column(String(80))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    status: Mapped[str] = mapped_column(String(12), default=ScenarioStatus.DRAFT)
    locked_by: Mapped[str | None] = mapped_column(String(80), nullable=True)
    locked_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    # Optimistic-concurrency token — see class docstring above.
    version: Mapped[int] = mapped_column(Integer, default=1)

    is_shared: Mapped[bool] = mapped_column(Boolean, default=True)
    # Self-referential FK ("cloned from") — unlike current_run_id below,
    # this is safe as a real FK because it points within the SAME table,
    # so there is no cross-table creation-order/ALTER TABLE problem.
    parent_scenario_id: Mapped[int | None] = mapped_column(
        ForeignKey("dim_scenario.scenario_id"), nullable=True
    )
    # Purely descriptive/derived — "this scenario is a -15% clone of its
    # parent" — set by clone_scenario() for display purposes; nothing
    # downstream depends on it being internally consistent with budget_usd.
    budget_adjustment_pct: Mapped[float | None] = mapped_column(Float, nullable=True)

    # ── Scenario input parameters ──────────────────────────────────────
    # Planner-editable overrides. None means "inherit the process-wide
    # default from src.utils.config.TA_CFG at solve time" (see
    # src/scenarios/runner.py::build_config_for_scenario) rather than
    # "force zero", which a 0.0 default would incorrectly imply for a
    # numeric column — exactly the kind of zero-vs-unset ambiguity
    # solver.py's _safe_ratio docstring warns about elsewhere in this
    # codebase, avoided here by using a nullable column instead.
    turnaround_date: Mapped[str | None] = mapped_column(String(20), nullable=True)
    budget_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_mech_hours: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_elec_hours: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_inst_hours: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_civil_hours: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Deliberately NOT a ForeignKey — see "Scenario vs. Run" in this
    # module's docstring for why a hard FK here is cross-backend-unsafe.
    current_run_id: Mapped[int | None] = mapped_column(Integer, nullable=True)

    runs: Mapped[list["DimRun"]] = relationship(
        back_populates="scenario", foreign_keys="DimRun.scenario_id"
    )


# ─── Dimension: Run (one row per optimizer EXECUTION) ─────────────────────────


class DimRun(Base):
    """
    One row per optimizer execution. This is the scenario dimension that
    makes multi-run comparison possible in Power BI — slice any visual by
    run_id (or run_label) to compare budget scenarios, craft-hour caps, or
    turnaround dates side by side.
    """

    __tablename__ = "dim_run"

    run_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_timestamp: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    run_label: Mapped[str] = mapped_column(String(120))  # human-readable, e.g. "Default $5M"

    # Nullable: most runs throughout this project's history (every test in
    # tests/test_db.py, every plain `run_optimizer.py` invocation) are NOT
    # tied to a saved DimScenario — they are one-off, ad-hoc executions,
    # and that remains fully supported. A run only gets a scenario_id when
    # it was produced via src/scenarios/runner.py::solve_scenario().
    scenario_id: Mapped[int | None] = mapped_column(ForeignKey("dim_scenario.scenario_id"), nullable=True)

    turnaround_date: Mapped[str] = mapped_column(String(20))
    budget_usd: Mapped[float] = mapped_column(Float)
    max_mech_hours: Mapped[float] = mapped_column(Float)
    max_elec_hours: Mapped[float] = mapped_column(Float)
    max_inst_hours: Mapped[float] = mapped_column(Float)
    max_civil_hours: Mapped[float] = mapped_column(Float)
    planning_horizon_days: Mapped[int] = mapped_column(Integer)

    solver_status: Mapped[str] = mapped_column(String(20))
    solve_time_s: Mapped[float] = mapped_column(Float)
    tasks_total: Mapped[int] = mapped_column(Integer)
    tasks_selected: Mapped[int] = mapped_column(Integer)
    budget_used_usd: Mapped[float] = mapped_column(Float)
    budget_utilisation: Mapped[float] = mapped_column(Float)
    total_net_value_usd: Mapped[float] = mapped_column(Float)
    roi_ratio: Mapped[float] = mapped_column(Float)
    total_risk_score_reduced: Mapped[int] = mapped_column(Integer)

    facts: Mapped[list["FactWorkOrderDecision"]] = relationship(back_populates="run")
    scenario: Mapped["DimScenario | None"] = relationship(back_populates="runs", foreign_keys=[scenario_id])


# ─── Dimension: Asset ──────────────────────────────────────────────────────────


class DimAsset(Base):
    """
    Descriptive, slowly-changing attributes of a physical asset — identity,
    classification, and replacement value. Deliberately does NOT include
    age_days, failure_prob, or risk_score, since those depend on which run
    produced them (a different turnaround_date changes age_days; different
    failure history changes failure_prob). Those measures live in the fact
    table where they belong.
    """

    __tablename__ = "dim_asset"

    asset_tag: Mapped[str] = mapped_column(String(20), primary_key=True)
    asset_class: Mapped[str] = mapped_column(String(10), index=True)
    asset_name: Mapped[str] = mapped_column(String(60))
    area: Mapped[str] = mapped_column(String(40), index=True)
    install_date: Mapped[str] = mapped_column(String(20))
    replace_usd: Mapped[float] = mapped_column(Float)
    c_safety: Mapped[int] = mapped_column(Integer)
    c_env: Mapped[int] = mapped_column(Integer)
    c_prod: Mapped[int] = mapped_column(Integer)
    c_cost: Mapped[int] = mapped_column(Integer)

    facts: Mapped[list["FactWorkOrderDecision"]] = relationship(back_populates="asset")


# ─── Small lookup dimensions ──────────────────────────────────────────────────


class DimTaskType(Base):
    __tablename__ = "dim_task_type"

    task_type_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    task_type_name: Mapped[str] = mapped_column(String(30), unique=True)

    facts: Mapped[list["FactWorkOrderDecision"]] = relationship(back_populates="task_type")


class DimPriority(Base):
    __tablename__ = "dim_priority"

    priority_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    priority_name: Mapped[str] = mapped_column(String(20), unique=True)
    priority_weight: Mapped[int] = mapped_column(Integer)

    facts: Mapped[list["FactWorkOrderDecision"]] = relationship(back_populates="priority")


class DimRiskLevel(Base):
    __tablename__ = "dim_risk_level"

    risk_level_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    risk_level_name: Mapped[str] = mapped_column(String(20), unique=True)
    # Sort order for Power BI visuals (LOW < MEDIUM < HIGH < CRITICAL), since
    # the alphabetical default would otherwise put CRITICAL before HIGH.
    sort_order: Mapped[int] = mapped_column(Integer)

    facts: Mapped[list["FactWorkOrderDecision"]] = relationship(back_populates="risk_level")


# ─── Fact table ────────────────────────────────────────────────────────────────


class FactWorkOrderDecision(Base):
    """
    One row per (run_id, wo_id): every measure the optimizer produced for
    that work order in that specific run — cost, craft-hours, reliability
    output, risk score, and the solver's INCLUDE/DEFER decision.
    """

    __tablename__ = "fact_work_order_decision"
    __table_args__ = (
        UniqueConstraint("run_id", "wo_id", name="uq_run_wo"),
        Index("ix_fact_run_selected", "run_id", "selected"),
    )

    fact_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    run_id: Mapped[int] = mapped_column(ForeignKey("dim_run.run_id"), index=True)
    asset_tag: Mapped[str] = mapped_column(ForeignKey("dim_asset.asset_tag"), index=True)
    task_type_id: Mapped[int] = mapped_column(ForeignKey("dim_task_type.task_type_id"))
    priority_id: Mapped[int] = mapped_column(ForeignKey("dim_priority.priority_id"))
    risk_level_id: Mapped[int] = mapped_column(ForeignKey("dim_risk_level.risk_level_id"))

    wo_id: Mapped[str] = mapped_column(String(20))
    description: Mapped[str] = mapped_column(String(500))
    predecessor_wo_id: Mapped[str | None] = mapped_column(String(20), nullable=True)
    mandatory: Mapped[bool] = mapped_column(Boolean)

    age_days: Mapped[float] = mapped_column(Float)
    estimated_cost_usd: Mapped[float] = mapped_column(Float)
    mech_hours: Mapped[float] = mapped_column(Float)
    elec_hours: Mapped[float] = mapped_column(Float)
    inst_hours: Mapped[float] = mapped_column(Float)
    civil_hours: Mapped[float] = mapped_column(Float)
    total_craft_hours: Mapped[float] = mapped_column(Float)
    duration_days: Mapped[int] = mapped_column(Integer)

    fitted_beta: Mapped[float] = mapped_column(Float)
    fitted_eta: Mapped[float] = mapped_column(Float)
    failure_prob: Mapped[float] = mapped_column(Float)
    rul_days: Mapped[float] = mapped_column(Float)

    consequence_score: Mapped[float] = mapped_column(Float)
    likelihood_tier: Mapped[int] = mapped_column(Integer)
    consequence_tier: Mapped[int] = mapped_column(Integer)
    risk_score: Mapped[int] = mapped_column(Integer)
    deferred_cost_usd: Mapped[float] = mapped_column(Float)
    net_value_usd: Mapped[float] = mapped_column(Float)

    selected: Mapped[bool] = mapped_column(Boolean, index=True)
    decision: Mapped[str] = mapped_column(String(10))

    run: Mapped["DimRun"] = relationship(back_populates="facts")
    asset: Mapped["DimAsset"] = relationship(back_populates="facts")
    task_type: Mapped["DimTaskType"] = relationship(back_populates="facts")
    priority: Mapped["DimPriority"] = relationship(back_populates="facts")
    risk_level: Mapped["DimRiskLevel"] = relationship(back_populates="facts")
