"""
runner.py — Execute the optimizer pipeline for a saved scenario.

``solve_scenario`` is the single entry point.  It:
  1. Looks up the scenario's parameter overrides.
  2. Merges them onto a copy of the process-wide TA_CFG (so nothing is
     mutated globally — each call is isolated).
  3. Tags the resulting DimRun row with the scenario's ``scenario_id``.
  4. Updates ``DimScenario.current_run_id`` so "compare A vs B" always
     reflects each scenario's latest solve.

It intentionally re-uses ``run_pipeline`` from ``src/main.py`` unchanged —
all the ETL, Weibull, risk-scoring, and ILP logic lives there and the scenario
system is purely a thin collaboration wrapper on top of it.
"""

from __future__ import annotations

import copy

from sqlalchemy import Engine

from src.utils.config import TA_CFG, TurnaroundConfig
from src.db.schema import DimScenario
from src.main import run_pipeline, PipelineResult
from src.scenarios.manager import (
    get_scenario,
    set_current_run,
    ScenarioArchivedError,
)
from src.utils.helpers import get_logger

log = get_logger("scenarios.runner")


def build_config_for_scenario(scenario: DimScenario, base_cfg: TurnaroundConfig | None = None) -> TurnaroundConfig:
    """
    Build a ``TurnaroundConfig`` for ``scenario`` by overlaying its explicit
    parameter overrides onto ``base_cfg`` (defaults to the global ``TA_CFG``).

    ``None`` fields in the scenario mean "inherit from base_cfg" — they are
    NOT treated as zero, which would silently zero-out a budget or craft-hour
    cap and produce a mis-leading "zero-budget" solve.

    Parameters
    ----------
    scenario
        A detached ``DimScenario`` ORM instance (e.g. from ``get_scenario``).
    base_cfg
        The baseline config to start from.  Defaults to the process-wide
        ``TA_CFG`` singleton.  Pass a custom config in tests to avoid
        global-state pollution.

    Returns
    -------
    TurnaroundConfig
        A **new** TurnaroundConfig instance with scenario overrides applied.
        The global ``TA_CFG`` singleton is never mutated.
    """
    if base_cfg is None:
        base_cfg = TA_CFG

    cfg = copy.copy(base_cfg)  # shallow copy — all fields are immutable scalars

    if scenario.turnaround_date is not None:
        cfg.turnaround_date = scenario.turnaround_date
    if scenario.budget_usd is not None:
        cfg.total_budget = scenario.budget_usd
    if scenario.max_mech_hours is not None:
        cfg.max_mech_hours = scenario.max_mech_hours
    if scenario.max_elec_hours is not None:
        cfg.max_elec_hours = scenario.max_elec_hours
    if scenario.max_inst_hours is not None:
        cfg.max_inst_hours = scenario.max_inst_hours
    if scenario.max_civil_hours is not None:
        cfg.max_civil_hours = scenario.max_civil_hours

    log.info(
        "Scenario %d config: budget=$%.0f, ta_date=%s, mech=%.0fh",
        scenario.scenario_id,
        cfg.total_budget,
        cfg.turnaround_date,
        cfg.max_mech_hours,
    )
    return cfg


def solve_scenario(  # pragma: no cover
    engine: Engine,
    scenario_id: int,
    *,
    regenerate_data: bool = False,
    database_url: str | None = None,
    base_cfg: TurnaroundConfig | None = None,
) -> PipelineResult:
    """
    Run the full ETL → Weibull → Risk → ILP → Reporting → DB pipeline for
    a saved scenario, then update ``DimScenario.current_run_id``.

    Parameters
    ----------
    engine
        SQLAlchemy engine for the star-schema database.  The same engine is
        used for both reading the scenario parameters and writing the results.
    scenario_id
        Which saved scenario to solve.
    regenerate_data
        Force re-generation of the synthetic CMMS dataset even if cached CSVs
        exist.  Forwarded verbatim to ``run_pipeline``.
    database_url
        Override the database URL used by ``run_pipeline`` for its own
        ``write_results_to_db`` call.  ``None`` → use the engine's URL.
    base_cfg
        Baseline ``TurnaroundConfig`` to overlay scenario params onto.
        ``None`` → use the global ``TA_CFG`` singleton.

    Returns
    -------
    PipelineResult
        The same bundle returned by ``run_pipeline`` — solver result, Excel
        path, dashboard path, audit log path, elapsed time, and db_run_id.

    Raises
    ------
    ScenarioNotFoundError  — ``scenario_id`` does not exist.
    ScenarioArchivedError  — cannot solve an archived scenario.
    RuntimeError           — pipeline failure (propagated from run_pipeline).
    """
    scenario = get_scenario(engine, scenario_id)

    if scenario.status == "ARCHIVED":
        raise ScenarioArchivedError(
            f"Scenario {scenario_id} ({scenario.name!r}) is archived. "
            "Create a new scenario or clone this one to run a new solve."
        )

    # Build a scenario-specific config without touching the global singleton.
    cfg = build_config_for_scenario(scenario, base_cfg)

    # Derive database URL from the engine if not overridden.
    db_url = database_url or str(engine.url)

    log.info(
        "Solving scenario %d (%r) — budget=$%.0f, ta_date=%s",
        scenario_id,
        scenario.name,
        cfg.total_budget,
        cfg.turnaround_date,
    )

    # Run the pipeline with this scenario's config.
    result = run_pipeline(
        budget=cfg.total_budget,
        turnaround_date=cfg.turnaround_date,
        regenerate_data=regenerate_data,
        enable_db=True,
        database_url=db_url,
        run_label=scenario.name,
    )

    # Tag the newly written dim_run row with our scenario_id.
    if result.db_run_id is not None:
        _tag_run_with_scenario(engine, result.db_run_id, scenario_id)
        set_current_run(engine, scenario_id, result.db_run_id)
        log.info(
            "Scenario %d: run_id=%d tagged, current_run_id updated",
            scenario_id,
            result.db_run_id,
        )

    return result


def _tag_run_with_scenario(engine: Engine, run_id: int, scenario_id: int) -> None:  # pragma: no cover
    """
    Set ``dim_run.scenario_id`` on the newly written run row.

    ``write_results_to_db`` doesn't know about scenarios (it's a lower-level
    helper), so we patch the row immediately after it commits.
    """
    from sqlalchemy.orm import Session
    from src.db.schema import DimRun

    with Session(engine) as session:
        run = session.get(DimRun, run_id)
        if run is not None:
            run.scenario_id = scenario_id
            session.commit()
