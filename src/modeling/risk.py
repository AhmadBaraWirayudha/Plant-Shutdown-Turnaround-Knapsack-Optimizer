"""
risk.py — Risk scoring and criticality-matrix classification.

Risk Score = Failure Probability  ×  Weighted Consequence Score

Consequence dimensions (each 1–5):
  Safety       (personnel injury / fatality)
  Environmental (spill, emission)
  Production   (production loss, throughput impact)
  Cost         (repair / replacement cost)

The 5 × 5 Criticality Matrix follows API RP-580 / ISO 31000 conventions.
Likelihood tiers are mapped from the Weibull failure probability.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from src.utils.config import RISK_CFG
from src.utils.helpers import get_logger, timed

log = get_logger("modeling.risk")

# ─── Likelihood tiering (Probability → 1-5 integer) ──────────────────────────
# Bins align with common industry risk-matrix practice
LIKELIHOOD_BINS = [0.0, 0.05, 0.15, 0.35, 0.60, 1.01]
LIKELIHOOD_LABELS = [1, 2, 3, 4, 5]  # 1=Rare … 5=Almost Certain

LIKELIHOOD_NAMES = {
    1: "Rare (<5%)",
    2: "Unlikely (5-15%)",
    3: "Possible (15-35%)",
    4: "Likely (35-60%)",
    5: "Almost Certain (>60%)",
}

# Consequence tier labels (1-5)
CONSEQUENCE_NAMES = {
    1: "Negligible",
    2: "Minor",
    3: "Moderate",
    4: "Major",
    5: "Catastrophic",
}


# ─── Risk Level thresholds (Likelihood × Consequence, range 1-25) ─────────────
def risk_level(lik: int, con: int) -> str:
    """Map a (likelihood, consequence) pair to a risk level label."""
    score = lik * con
    if score >= 17:
        return "CRITICAL"
    if score >= 10:
        return "HIGH"
    if score >= 5:
        return "MEDIUM"
    return "LOW"


# ─── Per-task risk computation ────────────────────────────────────────────────


def consequence_score(
    c_safety: float,
    c_env: float,
    c_prod: float,
    c_cost: float,
    w_safety: float | None = None,
    w_env: float | None = None,
    w_prod: float | None = None,
    w_cost: float | None = None,
) -> float:
    """
    Weighted average of four consequence dimensions, clipped to [1, 5].

    The safety weight dominates by design (40 %) to ensure personnel-risk
    tasks are always treated as high consequence.

    Each weight defaults to `None`, resolved against the LIVE value of the
    matching RISK_CFG field inside this function body rather than baked in
    as a Python default-argument value at import time — see
    weibull.py's failure_probability() docstring for why a literal
    `= RISK_CFG.w_safety` default here would be a latent bug the moment
    anything mutates RISK_CFG after this module is first imported.
    """
    if w_safety is None:
        w_safety = RISK_CFG.w_safety
    if w_env is None:
        w_env = RISK_CFG.w_environmental
    if w_prod is None:
        w_prod = RISK_CFG.w_production
    if w_cost is None:
        w_cost = RISK_CFG.w_cost
    raw = w_safety * c_safety + w_env * c_env + w_prod * c_prod + w_cost * c_cost
    return float(np.clip(raw, 1.0, 5.0))


def deferred_risk_cost(
    failure_prob: float,
    consequence: float,
    replace_usd: float,
    factor: float | None = None,
) -> float:
    """
    Monetary value of the risk avoided by executing the task now rather than
    deferring it to the next turnaround cycle.

    Deferred risk cost ($) = P(failure) × consequence_score × replace_usd × factor

    `factor` defaults to `None`, resolved against the LIVE value of
    RISK_CFG.deferral_cost_factor (default 0.15) inside this function body —
    represents the fraction of replacement value that is actually at stake
    (accounts for partial consequences, partial repairs, etc.).
    """
    if factor is None:
        factor = RISK_CFG.deferral_cost_factor
    return failure_prob * consequence * replace_usd * factor


# ─── Batch scoring ────────────────────────────────────────────────────────────


@timed
def compute_risk_scores(wos: pd.DataFrame) -> pd.DataFrame:
    """
    Append risk columns to the work-order DataFrame:

      consequence_score   – weighted average of 4 consequence dimensions (1-5)
      likelihood_tier     – integer 1-5 from failure_prob
      consequence_tier    – integer 1-5 (rounded consequence_score)
      risk_score          – likelihood_tier × consequence_tier (1-25)
      risk_level          – "CRITICAL" / "HIGH" / "MEDIUM" / "LOW"
      deferred_cost_usd   – monetary estimate of deferred risk
      net_value_usd       – deferred_cost_usd - estimated_cost_usd  (the ILP value)
    """
    df = wos.copy()

    # Consequence score (weighted, continuous)
    df["consequence_score"] = df.apply(
        lambda r: consequence_score(r.c_safety, r.c_env, r.c_prod, r.c_cost),
        axis=1,
    ).round(3)

    # Likelihood tier from failure probability
    df["likelihood_tier"] = pd.cut(
        df["failure_prob"],
        bins=LIKELIHOOD_BINS,
        labels=LIKELIHOOD_LABELS,
        include_lowest=True,
    ).astype(int)

    # Consequence tier (round to nearest integer)
    df["consequence_tier"] = df["consequence_score"].round().astype(int).clip(1, 5)

    # Risk score on 5×5 matrix
    df["risk_score"] = df["likelihood_tier"] * df["consequence_tier"]

    # Risk level label
    df["risk_level"] = df.apply(lambda r: risk_level(r.likelihood_tier, r.consequence_tier), axis=1)

    # Monetary value of deferred risk
    df["deferred_cost_usd"] = df.apply(
        lambda r: deferred_risk_cost(r.failure_prob, r.consequence_score, r.replace_usd),
        axis=1,
    ).round(2)

    # Net value = true risk-adjusted economic benefit of executing now vs. cost.
    # NOTE: Mandatory tasks are forced into the solution via an explicit
    # x_i == 1 constraint in the ILP (see knapsack_model.py), NOT via an
    # objective-function bonus. This keeps the reported $ value honest —
    # a mandatory statutory inspection that costs more than its computed
    # deferred-risk value will correctly show a negative net_value_usd,
    # rather than having that economic reality masked by an artificial
    # bonus. Executives reading the dashboard see real numbers.
    df["net_value_usd"] = (df["deferred_cost_usd"] - df["estimated_cost_usd"]).round(2)

    # Summary
    level_counts = df["risk_level"].value_counts()
    log.info(
        "Risk scoring done | CRITICAL=%d  HIGH=%d  MEDIUM=%d  LOW=%d",
        level_counts.get("CRITICAL", 0),
        level_counts.get("HIGH", 0),
        level_counts.get("MEDIUM", 0),
        level_counts.get("LOW", 0),
    )
    log.info(
        "  Deferred risk portfolio: $%s | Mean P(fail)=%.3f",
        f"{df.deferred_cost_usd.sum():,.0f}",
        df.failure_prob.mean(),
    )
    return df


# ─── Criticality matrix summary ───────────────────────────────────────────────


def build_criticality_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a 5×5 pivot table (likelihood tiers vs consequence tiers)
    with task counts in each cell.  Used for heatmap visualisation.
    """
    pivot = (
        df.groupby(["likelihood_tier", "consequence_tier"])
        .size()
        .reset_index(name="count")
        .pivot(index="likelihood_tier", columns="consequence_tier", values="count")
        .fillna(0)
        .astype(int)
    )
    # Ensure all 5 rows and 5 columns exist
    for i in range(1, 6):
        if i not in pivot.index:
            pivot.loc[i] = 0
        if i not in pivot.columns:
            pivot[i] = 0
    pivot = pivot.sort_index().sort_index(axis=1)
    return pivot
