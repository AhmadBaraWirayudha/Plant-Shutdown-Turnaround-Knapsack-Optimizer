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

    dim_run ──┐
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

Grain of the fact table: one row per (run_id, wo_id) — the same work order
can legitimately appear across multiple runs with different selected /
cost / risk values if the scenario inputs changed between runs.
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


# ─── Dimension: Run (the "scenario" dimension) ────────────────────────────────


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
