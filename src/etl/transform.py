"""
transform.py — Data cleaning, validation, and feature engineering.

Applies business rules and quality checks before the data feeds into
Weibull modeling and the ILP optimizer.
"""

from __future__ import annotations
import pandas as pd

from src.utils.helpers import get_logger, timed

log = get_logger("etl.transform")


# ─── Work Order Cleaning ──────────────────────────────────────────────────────


@timed
def clean_work_orders(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and clean the raw work-order dataset.

    Steps
    -----
    1. Drop exact duplicates on wo_id.
    2. Coerce numeric columns; clamp negatives to 0.
    3. Fill missing craft hours with column medians.
    4. Cap individual task cost at $1 M (data-entry guard).
    5. Add `total_craft_hours` if absent.
    6. Normalise categorical strings to Title Case.
    7. Assert referential integrity for predecessor_wo_id.
    """
    df = raw.copy()

    # 1. Deduplicate
    before = len(df)
    df = df.drop_duplicates(subset=["wo_id"])
    if n_dropped := before - len(df):
        log.warning("Dropped %d duplicate wo_id rows", n_dropped)

    # 2. Numeric coercion & clamp
    numeric_cols = [
        "estimated_cost_usd",
        "mech_hours",
        "elec_hours",
        "inst_hours",
        "civil_hours",
        "total_craft_hours",
        "weibull_beta",
        "weibull_eta",
        "c_safety",
        "c_env",
        "c_prod",
        "c_cost",
        "replace_usd",
        "age_days",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].clip(lower=0)

    # 3. Fill missing hours with median
    hour_cols = ["mech_hours", "elec_hours", "inst_hours", "civil_hours"]
    for col in hour_cols:
        if df[col].isna().any():
            med = df[col].median()
            df[col] = df[col].fillna(med)
            log.warning("Filled %d NaN in %s with median %.1f", df[col].isna().sum(), col, med)

    # 4. Cost ceiling
    MAX_COST = 1_000_000
    over_cap = df["estimated_cost_usd"] > MAX_COST
    if over_cap.any():
        log.warning("Capping %d tasks with cost > $1 M", over_cap.sum())
        df.loc[over_cap, "estimated_cost_usd"] = MAX_COST

    # 5. Recalculate total craft hours
    df["total_craft_hours"] = df[hour_cols].sum(axis=1)

    # 6. Normalise categoricals
    for col in ["priority", "task_type", "area", "asset_class"]:
        if col in df.columns:
            df[col] = df[col].str.strip().str.title()

    # 7. Predecessor integrity check
    if "predecessor_wo_id" in df.columns:
        valid_ids = set(df["wo_id"])
        pred_mask = df["predecessor_wo_id"].notna()
        bad_preds = ~df.loc[pred_mask, "predecessor_wo_id"].isin(valid_ids)
        if bad_preds.any():
            n_bad = bad_preds.sum()
            log.warning("Nullifying %d unknown predecessor references", n_bad)
            df.loc[df[pred_mask].index[bad_preds], "predecessor_wo_id"] = None

    log.info(
        "Clean WOs: %d rows | %d mandatory | %d with predecessor",
        len(df),
        df.get("mandatory", pd.Series([False])).sum(),
        (df["predecessor_wo_id"].notna().sum() if "predecessor_wo_id" in df.columns else 0),
    )
    return df


# ─── Feature Engineering ──────────────────────────────────────────────────────


def add_priority_weight(df: pd.DataFrame) -> pd.DataFrame:
    """Map text priority to a numeric weight used in tie-breaking."""
    weight_map = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}
    df["priority_weight"] = df["priority"].map(weight_map).fillna(1).astype(int)
    return df


def add_wo_index_map(df: pd.DataFrame) -> dict[str, int]:
    """
    Return {wo_id → integer index} for use in the ILP model.
    Also ensures the DataFrame row-index aligns with optimizer index.
    """
    df.reset_index(drop=True, inplace=True)
    return {row.wo_id: i for i, row in df.iterrows()}


# ─── Failure History Cleaning ─────────────────────────────────────────────────


@timed
def clean_failure_history(raw: pd.DataFrame) -> pd.DataFrame:
    """Drop records with zero or negative TTF; remove obvious outliers (> 20 yr)."""
    df = raw.copy()
    df["time_to_failure_d"] = pd.to_numeric(df["time_to_failure_d"], errors="coerce")
    df = df[df["time_to_failure_d"].between(1, 20 * 365)].copy()
    log.info("Clean failure history: %d valid records", len(df))
    return df


# ─── Cross-table Referential Integrity ────────────────────────────────────────


def validate_referential_integrity(wos: pd.DataFrame, asset_master: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-check that every work order's asset_tag exists in the asset master.

    A missing master-data link is a data-quality signal to surface to
    planners (it usually means a CMMS export drifted out of sync with the
    asset register) — NOT a reason to silently drop the work order, since it
    may still represent real, billable, safety-relevant scope. We flag it
    via a boolean column and log a warning instead.
    """
    valid_tags = set(asset_master["asset_tag"])
    orphaned = ~wos["asset_tag"].isin(valid_tags)

    df = wos.copy()
    df["asset_master_linked"] = ~orphaned

    if orphaned.any():
        sample = wos.loc[orphaned, "asset_tag"].unique()[:10].tolist()
        log.warning(
            "%d work orders reference an asset_tag NOT found in asset_master "
            "(sample: %s) — flagged via asset_master_linked=False, not dropped",
            int(orphaned.sum()),
            sample,
        )
    else:
        log.info(
            "Referential integrity OK: all %d work orders link to a known asset",
            len(df),
        )

    return df


def enrich_with_asset_name(wos: pd.DataFrame, asset_master: pd.DataFrame) -> pd.DataFrame:
    """
    Bring the human-readable `asset_name` (e.g. "Centrifugal Pump") from the
    asset master onto each work order via a left merge on asset_tag.

    Without this, downstream consumers (the Excel export, the dashboard
    table, and the database's dim_asset) would each need to either re-derive
    a display name by parsing the free-text `description` field — fragile,
    and a different bug magnet in every consumer — or fall back to showing
    the bare asset_class code. Doing the merge once, here, means every
    downstream layer can just read `asset_name` directly.
    """
    name_lookup = asset_master[["asset_tag", "asset_name"]].drop_duplicates("asset_tag")
    merged = wos.merge(name_lookup, on="asset_tag", how="left")

    missing = merged["asset_name"].isna()
    if missing.any():
        log.warning(
            "%d work orders had no asset_name match (orphaned asset_tag) — "
            "falling back to asset_class as the display name",
            int(missing.sum()),
        )
        merged.loc[missing, "asset_name"] = merged.loc[missing, "asset_class"]

    return merged


# ─── Full transform pipeline ──────────────────────────────────────────────────


@timed
def run_transforms(
    raw_wos: pd.DataFrame,
    raw_failures: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply all transforms; return (clean_wos, clean_failures)."""
    clean_wos = clean_work_orders(raw_wos)
    clean_wos = add_priority_weight(clean_wos)
    clean_fails = clean_failure_history(raw_failures)
    return clean_wos, clean_fails
