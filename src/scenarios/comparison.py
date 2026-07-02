"""
comparison.py — Side-by-side scenario comparison.

``compare_scenarios`` reads the ``current_run_id`` of each scenario from the
database, fetches the headline KPIs from ``dim_run`` and the per-work-order
decisions from ``fact_work_order_decision``, and produces a
``ScenarioComparison`` dataclass containing:

  * ``kpis``        — one-row-per-scenario DataFrame of headline metrics.
  * ``delta``       — a signed-difference summary ("B minus A") as a dict.
  * ``added``       — work orders selected in B but deferred in A.
  * ``removed``     — work orders selected in A but deferred in B.
  * ``common_in``   — work orders selected in both.
  * ``common_out``  — work orders deferred in both.

Design notes
------------
The comparison is always "B relative to A" (scenario_b is the "new" scenario
being evaluated against scenario_a as the "baseline").  This makes the delta
signs intuitive: a positive ``delta_tasks_selected`` means B selects more
tasks than A; a negative ``delta_budget_used_usd`` means B is cheaper.

When a scenario has never been solved (``current_run_id`` is None), a clear
``ScenarioNotSolvedError`` is raised so the caller can surface a helpful
message rather than a cryptic KeyError.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sqlalchemy import Engine

from src.db.schema import DimRun, DimScenario
from src.db.queries import get_run_facts
from src.scenarios.manager import get_scenario
from src.utils.helpers import get_logger

log = get_logger("scenarios.comparison")


# ─── Exceptions ───────────────────────────────────────────────────────────────


class ScenarioNotSolvedError(RuntimeError):
    """
    Raised when a scenario has no ``current_run_id`` — i.e. it was created
    but has never been solved.  The caller should call ``solve_scenario``
    first.
    """


# ─── Result dataclass ─────────────────────────────────────────────────────────


@dataclass
class ScenarioComparison:
    """
    Side-by-side comparison of two solved scenarios.

    All DataFrames use the same column layout as ``get_run_facts`` (see
    ``src/db/queries.py``) so callers can display or export them directly.

    Attributes
    ----------
    scenario_a, scenario_b
        Detached ``DimScenario`` ORM instances (read-only metadata).
    run_a_id, run_b_id
        The ``run_id`` values from ``dim_run`` that were compared.
    kpis
        Two-row DataFrame (one per scenario) with headline KPIs from ``dim_run``.
    delta
        Signed "B − A" differences for every numeric KPI.  Positive means B
        is better/larger; negative means A is better/larger.  Sign convention
        for ``delta_tasks_selected`` and ``delta_total_net_value_usd`` is
        intentionally kept "larger = more scope selected" — whether that is
        better depends on the business context.
    added
        Work orders that scenario B SELECTS but scenario A DEFERS.
    removed
        Work orders that scenario A SELECTS but scenario B DEFERS.
    common_in
        Work orders selected by BOTH scenarios.
    common_out
        Work orders deferred by BOTH scenarios.
    """

    scenario_a: DimScenario
    scenario_b: DimScenario
    run_a_id: int
    run_b_id: int
    kpis: pd.DataFrame
    delta: dict
    added: pd.DataFrame    # in B, not in A
    removed: pd.DataFrame  # in A, not in B
    common_in: pd.DataFrame
    common_out: pd.DataFrame

    @property
    def n_added(self) -> int:
        """Number of work orders gained in scenario B."""
        return len(self.added)

    @property
    def n_removed(self) -> int:
        """Number of work orders lost in scenario B."""
        return len(self.removed)

    def summary_text(self) -> str:
        """
        Return a human-readable two-paragraph summary of the comparison,
        suitable for printing to a terminal or inserting into a report.
        """
        a_name = self.scenario_a.name
        b_name = self.scenario_b.name
        d = self.delta

        budget_a = self.kpis.loc[self.kpis["scenario_label"] == a_name, "budget_usd"].iloc[0]
        budget_b = self.kpis.loc[self.kpis["scenario_label"] == b_name, "budget_usd"].iloc[0]

        label_a = self.kpis["scenario_label"] == a_name
        label_b = self.kpis["scenario_label"] == b_name
        tasks_a = self.kpis.loc[label_a, "tasks_selected"].iloc[0]
        tasks_b = self.kpis.loc[label_b, "tasks_selected"].iloc[0]

        lines = [
            f"Scenario comparison: '{a_name}'  vs  '{b_name}'",
            "─" * 60,
            f"  Budget             : ${budget_a:>14,.0f}  →  ${budget_b:>14,.0f}"
            f"  ({d['delta_budget_usd']:+,.0f})",
            f"  Tasks selected     : {tasks_a:>6d}  →  {tasks_b:>6d}"
            f"  ({d['delta_tasks_selected']:+d})",
            f"  Budget used        : ${d.get('budget_used_a', 0):>14,.0f}  →  "
            f"${d.get('budget_used_b', 0):>14,.0f}  ({d['delta_budget_used_usd']:+,.0f})",
            f"  Total net value    : ${d.get('net_value_a', 0):>14,.0f}  →  "
            f"${d.get('net_value_b', 0):>14,.0f}  ({d['delta_total_net_value_usd']:+,.0f})",
            f"  ROI ratio          : {d.get('roi_a', 0):>9.2f}x  →  "
            f"{d.get('roi_b', 0):>9.2f}x  ({d['delta_roi_ratio']:+.2f})",
            "",
            f"  Work-order changes : +{self.n_added} added, -{self.n_removed} removed "
            f"({len(self.common_in)} common selected, "
            f"{len(self.common_out)} common deferred)",
        ]
        return "\n".join(lines)


# ─── Public API ───────────────────────────────────────────────────────────────


def compare_scenarios(
    engine: Engine,
    scenario_id_a: int,
    scenario_id_b: int,
) -> ScenarioComparison:
    """
    Compare two scenarios side by side and return a ``ScenarioComparison``.

    Both scenarios must already have been solved (i.e. have a non-None
    ``current_run_id`` pointing at a ``dim_run`` row).

    Parameters
    ----------
    engine
        SQLAlchemy engine pointing at the star-schema database.
    scenario_id_a
        The BASELINE scenario (scenario "A").
    scenario_id_b
        The NEW scenario being evaluated (scenario "B").

    Returns
    -------
    ScenarioComparison

    Raises
    ------
    ScenarioNotFoundError   — either scenario does not exist.
    ScenarioNotSolvedError  — either scenario has no ``current_run_id``.
    """
    scenario_a = get_scenario(engine, scenario_id_a)
    scenario_b = get_scenario(engine, scenario_id_b)

    if scenario_a.current_run_id is None:
        raise ScenarioNotSolvedError(
            f"Scenario {scenario_id_a} ({scenario_a.name!r}) has never been solved. "
            "Call solve_scenario() first."
        )
    if scenario_b.current_run_id is None:
        raise ScenarioNotSolvedError(
            f"Scenario {scenario_id_b} ({scenario_b.name!r}) has never been solved. "
            "Call solve_scenario() first."
        )

    run_a_id = scenario_a.current_run_id
    run_b_id = scenario_b.current_run_id

    log.info(
        "Comparing scenario %d (run %d) vs scenario %d (run %d)",
        scenario_id_a, run_a_id, scenario_id_b, run_b_id,
    )

    # ── Fetch KPI rows from dim_run ────────────────────────────────────────
    kpis_a = _fetch_run_kpis(engine, run_a_id, label=scenario_a.name)
    kpis_b = _fetch_run_kpis(engine, run_b_id, label=scenario_b.name)
    kpis = pd.concat([kpis_a, kpis_b], ignore_index=True)

    # ── Fetch per-work-order decisions ────────────────────────────────────
    facts_a = get_run_facts(engine, run_a_id)
    facts_b = get_run_facts(engine, run_b_id)

    # Selected work-order ID sets
    sel_a = set(facts_a.loc[facts_a["selected"].astype(bool), "wo_id"])
    sel_b = set(facts_b.loc[facts_b["selected"].astype(bool), "wo_id"])

    # Build per-category DataFrames (using B's fact rows as the "current" view)
    all_wos = set(facts_a["wo_id"]).union(set(facts_b["wo_id"]))

    added_ids = sel_b - sel_a
    removed_ids = sel_a - sel_b
    common_in_ids = sel_a & sel_b
    common_out_ids = (all_wos - sel_a) & (all_wos - sel_b)

    added = _subset_facts(facts_b, added_ids)
    removed = _subset_facts(facts_a, removed_ids)
    common_in = _subset_facts(facts_b, common_in_ids)
    common_out = _subset_facts(facts_b, common_out_ids)

    # ── Build delta dict ──────────────────────────────────────────────────
    def _kpi(label: str, col: str) -> float:
        row = kpis.loc[kpis["scenario_label"] == label]
        return float(row[col].iloc[0]) if not row.empty else 0.0

    budget_used_a = _kpi(scenario_a.name, "budget_used_usd")
    budget_used_b = _kpi(scenario_b.name, "budget_used_usd")
    net_value_a = _kpi(scenario_a.name, "total_net_value_usd")
    net_value_b = _kpi(scenario_b.name, "total_net_value_usd")
    tasks_sel_a = int(_kpi(scenario_a.name, "tasks_selected"))
    tasks_sel_b = int(_kpi(scenario_b.name, "tasks_selected"))
    roi_a = _kpi(scenario_a.name, "roi_ratio")
    roi_b = _kpi(scenario_b.name, "roi_ratio")
    risk_a = _kpi(scenario_a.name, "total_risk_score_reduced")
    risk_b = _kpi(scenario_b.name, "total_risk_score_reduced")

    delta = {
        # Scenario inputs
        "delta_budget_usd": (_kpi(scenario_b.name, "budget_usd") - _kpi(scenario_a.name, "budget_usd")),
        # Outcomes
        "delta_tasks_selected": tasks_sel_b - tasks_sel_a,
        "delta_budget_used_usd": budget_used_b - budget_used_a,
        "delta_total_net_value_usd": net_value_b - net_value_a,
        "delta_roi_ratio": roi_b - roi_a,
        "delta_total_risk_score_reduced": risk_b - risk_a,
        "delta_tasks_added": len(added_ids),
        "delta_tasks_removed": len(removed_ids),
        # Point values (for summary_text)
        "budget_used_a": budget_used_a,
        "budget_used_b": budget_used_b,
        "net_value_a": net_value_a,
        "net_value_b": net_value_b,
        "roi_a": roi_a,
        "roi_b": roi_b,
    }

    return ScenarioComparison(
        scenario_a=scenario_a,
        scenario_b=scenario_b,
        run_a_id=run_a_id,
        run_b_id=run_b_id,
        kpis=kpis,
        delta=delta,
        added=added,
        removed=removed,
        common_in=common_in,
        common_out=common_out,
    )


def compare_many_scenarios(
    engine: Engine,
    scenario_ids: list[int],
) -> pd.DataFrame:
    """
    Return a summary DataFrame with one row per scenario for easy multi-scenario
    comparison in a notebook or Power BI table.

    All scenarios must have been solved (non-None ``current_run_id``).

    Columns include all headline KPIs from ``dim_run`` plus the scenario
    name, description, created_by, and status from ``dim_scenario``.

    Raises
    ------
    ScenarioNotFoundError   — any scenario ID does not exist.
    ScenarioNotSolvedError  — any scenario has no ``current_run_id``.
    """
    rows = []
    for sid in scenario_ids:
        s = get_scenario(engine, sid)
        if s.current_run_id is None:
            raise ScenarioNotSolvedError(
                f"Scenario {sid} ({s.name!r}) has never been solved."
            )
        kpis = _fetch_run_kpis(engine, s.current_run_id, label=s.name)
        row = kpis.iloc[0].to_dict()
        row["scenario_id"] = sid
        row["description"] = s.description
        row["created_by"] = s.created_by
        row["status"] = s.status
        row["turnaround_date"] = s.turnaround_date
        rows.append(row)
    return pd.DataFrame(rows)


# ─── Private helpers ──────────────────────────────────────────────────────────


def _fetch_run_kpis(engine: Engine, run_id: int, label: str) -> pd.DataFrame:
    """
    Fetch the headline KPI columns from one ``dim_run`` row and return a
    one-row DataFrame, adding a ``scenario_label`` column for display.
    """
    from sqlalchemy.orm import Session

    with Session(engine) as session:
        run = session.get(DimRun, run_id)
        if run is None:  # pragma: no cover
            raise KeyError(f"run_id={run_id} not found in dim_run")
        row = {
            "scenario_label": label,
            "run_id": run.run_id,
            "run_timestamp": run.run_timestamp,
            "budget_usd": run.budget_usd,
            "turnaround_date": run.turnaround_date,
            "tasks_total": run.tasks_total,
            "tasks_selected": run.tasks_selected,
            "budget_used_usd": run.budget_used_usd,
            "budget_utilisation": run.budget_utilisation,
            "total_net_value_usd": run.total_net_value_usd,
            "roi_ratio": run.roi_ratio,
            "total_risk_score_reduced": run.total_risk_score_reduced,
            "solver_status": run.solver_status,
            "solve_time_s": run.solve_time_s,
        }
    return pd.DataFrame([row])


def _subset_facts(facts: pd.DataFrame, wo_ids: set[str]) -> pd.DataFrame:
    """
    Return a filtered copy of ``facts`` for work orders in ``wo_ids``,
    sorted by ``net_value_usd`` descending.
    """
    if not wo_ids:
        return pd.DataFrame(columns=facts.columns)
    mask = facts["wo_id"].isin(wo_ids)
    subset = facts[mask].copy()
    if "net_value_usd" in subset.columns:
        subset = subset.sort_values("net_value_usd", ascending=False)
    return subset.reset_index(drop=True)
