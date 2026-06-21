"""
data_generator.py — Synthetic CMMS dataset for a petroleum-refinery turnaround.

Generates:
  • 550 work orders across 10 equipment classes
  • Historical failure-time records per equipment class
  • Asset master table with replacement costs and Weibull parameters
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.stats import weibull_min

from src.utils.config import DGEN_CFG, DATA_RAW
from src.utils.helpers import get_logger, timed

log = get_logger("data_generator")

# NOTE: There is deliberately NO module-level `rng = np.random.default_rng(...)`
# here. An earlier version had one, created once at import time from
# DGEN_CFG.random_seed — which meant any later `--seed` CLI override (or
# direct mutation of DGEN_CFG.random_seed) was silently ignored by every
# function that used it, since Python only evaluates that kind of
# module-level statement once, the first time this file is imported, no
# matter when the import happens relative to argument parsing. Every
# generator function below now takes an explicit `rng` parameter instead,
# created fresh by `generate_all()` at CALL time (not import time) from
# whatever DGEN_CFG.random_seed currently is. See
# tests/test_data_generator.py for the regression tests that catch this
# class of bug — they call generate_all() with two different seeds in the
# same process and assert every output table actually differs.

# ─── Equipment Catalog ────────────────────────────────────────────────────────
# Each entry describes ONE equipment CLASS with its population statistics.
EQUIPMENT_CATALOG: dict[str, dict] = {
    "PMP": {
        "name": "Centrifugal Pump",
        "count": 80,
        "beta": 2.5,
        "eta": 1825,
        "c_safety": 3,
        "c_env": 2,
        "c_prod": 4,
        "c_cost": 3,
        "replace_usd": 48_000,
    },
    "HX": {
        "name": "Heat Exchanger",
        "count": 35,
        "beta": 2.0,
        "eta": 2190,
        "c_safety": 4,
        "c_env": 3,
        "c_prod": 5,
        "c_cost": 4,
        "replace_usd": 195_000,
    },
    "CMP": {
        "name": "Reciprocating Compressor",
        "count": 18,
        "beta": 3.0,
        "eta": 1460,
        "c_safety": 5,
        "c_env": 4,
        "c_prod": 5,
        "c_cost": 5,
        "replace_usd": 490_000,
    },
    "VLV": {
        "name": "Control Valve",
        "count": 120,
        "beta": 1.5,
        "eta": 1095,
        "c_safety": 3,
        "c_env": 2,
        "c_prod": 3,
        "c_cost": 2,
        "replace_usd": 13_500,
    },
    "VSL": {
        "name": "Pressure Vessel",
        "count": 28,
        "beta": 2.8,
        "eta": 3650,
        "c_safety": 5,
        "c_env": 5,
        "c_prod": 5,
        "c_cost": 5,
        "replace_usd": 290_000,
    },
    "TWR": {
        "name": "Distillation Tower",
        "count": 12,
        "beta": 2.2,
        "eta": 2920,
        "c_safety": 5,
        "c_env": 5,
        "c_prod": 5,
        "c_cost": 5,
        "replace_usd": 980_000,
    },
    "INST": {
        "name": "Instrument/Analyser",
        "count": 180,
        "beta": 1.8,
        "eta": 730,
        "c_safety": 2,
        "c_env": 1,
        "c_prod": 3,
        "c_cost": 2,
        "replace_usd": 4_800,
    },
    "ELEC": {
        "name": "Electrical Equipment",
        "count": 85,
        "beta": 2.0,
        "eta": 1460,
        "c_safety": 3,
        "c_env": 1,
        "c_prod": 3,
        "c_cost": 3,
        "replace_usd": 24_000,
    },
    "TNK": {
        "name": "Storage Tank",
        "count": 20,
        "beta": 3.5,
        "eta": 5475,
        "c_safety": 4,
        "c_env": 5,
        "c_prod": 3,
        "c_cost": 4,
        "replace_usd": 390_000,
    },
    "PPL": {
        "name": "Piping/Pipeline Segment",
        "count": 45,
        "beta": 2.0,
        "eta": 3285,
        "c_safety": 4,
        "c_env": 4,
        "c_prod": 3,
        "c_cost": 3,
        "replace_usd": 88_000,
    },
}

PLANT_AREAS = [
    "Unit-100",
    "Unit-200",
    "Unit-300",
    "Utilities",
    "Offsites",
    "Tank-Farm",
    "Flare-System",
]
PRIORITY_MAP = {0: "Critical", 1: "High", 2: "Medium", 3: "Low"}

# Task type → cost range (USD) and craft hour ranges
TASK_TYPES: dict[str, dict] = {
    "Inspection": {
        "cost": (600, 8_000),
        "mech": (2, 10),
        "elec": (0, 2),
        "inst": (1, 5),
        "civil": (0, 1),
    },
    "Replacement": {
        "cost": (5_000, 120_000),
        "mech": (8, 60),
        "elec": (2, 20),
        "inst": (4, 24),
        "civil": (1, 10),
    },
    "Overhaul": {
        "cost": (15_000, 600_000),
        "mech": (40, 240),
        "elec": (8, 50),
        "inst": (8, 50),
        "civil": (4, 24),
    },
    "Cleaning": {
        "cost": (1_000, 25_000),
        "mech": (4, 28),
        "elec": (0, 2),
        "inst": (0, 2),
        "civil": (2, 10),
    },
    "Calibration": {
        "cost": (200, 3_000),
        "mech": (0, 2),
        "elec": (2, 10),
        "inst": (4, 20),
        "civil": (0, 1),
    },
    "Repair": {
        "cost": (2_000, 80_000),
        "mech": (8, 100),
        "elec": (4, 30),
        "inst": (2, 20),
        "civil": (1, 10),
    },
    "Testing": {
        "cost": (500, 15_000),
        "mech": (2, 20),
        "elec": (4, 24),
        "inst": (4, 24),
        "civil": (0, 2),
    },
}

TASK_TYPE_WEIGHTS = [0.20, 0.20, 0.12, 0.15, 0.15, 0.10, 0.08]  # sum=1


# ─── Asset Master ─────────────────────────────────────────────────────────────
def generate_asset_master(rng: np.random.Generator) -> pd.DataFrame:
    """Create individual asset records with installation date and current age.

    `rng` is required (not defaulted) specifically so a stale, frozen-at-import
    seed can never silently slip in — see the module-level note above.
    """
    records = []
    ref_date = datetime(2026, 10, 1)

    for eqp_code, spec in EQUIPMENT_CATALOG.items():
        for idx in range(1, spec["count"] + 1):
            tag = f"{eqp_code}-{idx:04d}"
            # Random installation date 1–15 years ago
            install_offset = int(rng.uniform(365, 5_475))
            install_date = ref_date - timedelta(days=install_offset)
            age_days = (ref_date - install_date).days

            records.append(
                {
                    "asset_tag": tag,
                    "asset_class": eqp_code,
                    "asset_name": spec["name"],
                    "area": rng.choice(PLANT_AREAS),
                    "install_date": install_date.date(),
                    "age_days": age_days,
                    "weibull_beta": spec["beta"],
                    "weibull_eta": spec["eta"],
                    "c_safety": spec["c_safety"],
                    "c_env": spec["c_env"],
                    "c_prod": spec["c_prod"],
                    "c_cost": spec["c_cost"],
                    "replace_usd": spec["replace_usd"],
                }
            )

    df = pd.DataFrame(records)
    log.info("Asset master: %d assets across %d classes", len(df), df.asset_class.nunique())
    return df


# ─── Failure History ─────────────────────────────────────────────────────────
def generate_failure_history(asset_master: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Simulate historical failure events using the class Weibull parameters."""
    records = []
    base_date = datetime(2026, 10, 1)

    for _, row in asset_master.iterrows():
        beta, eta = row.weibull_beta, row.weibull_eta
        # Simulate 3-8 failure events per asset over its service life
        num_failures = int(rng.integers(2, 7))
        failure_times = sorted(
            weibull_min.rvs(
                beta,
                scale=eta,
                size=num_failures,
                random_state=int(rng.integers(0, 9999)),
            ).tolist()
        )

        for i, tti in enumerate(failure_times):
            fail_date = base_date - timedelta(days=max(0, row.age_days - tti))
            records.append(
                {
                    "asset_tag": row.asset_tag,
                    "asset_class": row.asset_class,
                    "failure_no": i + 1,
                    "time_to_failure_d": round(float(tti), 2),
                    "failure_date": fail_date.date(),
                    "failure_mode": rng.choice(
                        [
                            "Wear",
                            "Corrosion",
                            "Fatigue",
                            "Fouling",
                            "Seal Failure",
                            "Bearing Failure",
                            "Leakage",
                            "Blockage",
                        ]
                    ),
                    "severity": int(rng.integers(1, 6)),
                }
            )

    df = pd.DataFrame(records)
    log.info("Failure history: %d records for %d assets", len(df), df.asset_tag.nunique())
    return df


# ─── Work Orders ──────────────────────────────────────────────────────────────
def _sample_hours(task_type: str, craft: str, rng: np.random.Generator) -> float:
    lo, hi = TASK_TYPES[task_type][craft]
    return round(float(rng.uniform(lo, hi)), 1)


def _sample_cost(task_type: str, rng: np.random.Generator) -> float:
    lo, hi = TASK_TYPES[task_type]["cost"]
    return round(float(rng.uniform(lo, hi)), 2)


@timed
def generate_work_orders(
    asset_master: pd.DataFrame,
    rng: np.random.Generator,
    n: int | None = None,
    mandatory_frac: float | None = None,
    pred_frac: float | None = None,
) -> pd.DataFrame:
    """
    Generate `n` synthetic work orders sampled from the asset population.

    `n`, `mandatory_frac`, and `pred_frac` default to `None`, resolved
    against the LIVE value of DGEN_CFG inside this function body rather
    than baked in as a Python default-argument value at import time — the
    latter is what made an earlier version of `--num-work-orders` silently
    do nothing (see the module-level note near the top of this file). Pass
    them explicitly to override DGEN_CFG for a single call without
    mutating global state.

    Returns a DataFrame ready for the ETL pipeline.
    """
    if n is None:
        n = DGEN_CFG.num_work_orders
    if mandatory_frac is None:
        mandatory_frac = DGEN_CFG.mandatory_fraction
    if pred_frac is None:
        pred_frac = DGEN_CFG.predecessor_fraction

    if n <= 0:
        raise ValueError(
            f"n (number of work orders) must be a positive integer, got {n}. "
            "There is no meaningful synthetic dataset with zero work orders — "
            "the optimizer selects a SUBSET of a nonempty backlog."
        )

    task_type_list = list(TASK_TYPES.keys())
    assets_sample = asset_master.sample(n=n, replace=True, random_state=rng).reset_index(drop=True)

    records = []
    for i, asset_row in assets_sample.iterrows():
        task_type = rng.choice(task_type_list, p=TASK_TYPE_WEIGHTS)
        cost = _sample_cost(task_type, rng)
        mech = _sample_hours(task_type, "mech", rng)
        elec = _sample_hours(task_type, "elec", rng)
        inst = _sample_hours(task_type, "inst", rng)
        civil = _sample_hours(task_type, "civil", rng)
        total_hrs = mech + elec + inst + civil

        # Priority: higher-consequence assets get higher priority
        max_c = max(asset_row.c_safety, asset_row.c_env, asset_row.c_prod, asset_row.c_cost)
        priority_w = np.array([0.40, 0.30, 0.20, 0.10]) if max_c >= 4 else np.array([0.10, 0.30, 0.40, 0.20])
        priority = PRIORITY_MAP[rng.choice([0, 1, 2, 3], p=priority_w)]

        records.append(
            {
                "wo_id": f"WO-{i+1:05d}",
                "description": f"{task_type} – {asset_row.asset_name} {asset_row.asset_tag}",
                "asset_tag": asset_row.asset_tag,
                "asset_class": asset_row.asset_class,
                "area": asset_row.area,
                "install_date": asset_row.install_date,
                "age_days": asset_row.age_days,
                "task_type": task_type,
                "priority": priority,
                "mandatory": False,  # override below
                "estimated_cost_usd": cost,
                "mech_hours": mech,
                "elec_hours": elec,
                "inst_hours": inst,
                "civil_hours": civil,
                "total_craft_hours": total_hrs,
                "duration_days": max(1, round(total_hrs / 8)),
                "weibull_beta": asset_row.weibull_beta,
                "weibull_eta": asset_row.weibull_eta,
                "c_safety": asset_row.c_safety,
                "c_env": asset_row.c_env,
                "c_prod": asset_row.c_prod,
                "c_cost": asset_row.c_cost,
                "replace_usd": asset_row.replace_usd,
                "predecessor_wo_id": None,  # set below
            }
        )

    df = pd.DataFrame(records)

    # ── Mandatory tasks (safety-critical or statutory) ─────────────────────
    mandatory_count = max(1, int(n * mandatory_frac))
    # Prefer Critical-priority or compressor/vessel tasks
    crit_mask = df["priority"] == "Critical"
    m_pool = df[crit_mask].index.tolist() if crit_mask.sum() >= mandatory_count else df.index.tolist()
    m_idx = rng.choice(m_pool, size=mandatory_count, replace=False)
    df.loc[m_idx, "mandatory"] = True

    # ── Precedence relationships ───────────────────────────────────────────
    pred_count = int(n * pred_frac)
    # Successor must be a non-mandatory task with index > predecessor
    succ_pool = df[~df["mandatory"]].index.tolist()[10:]
    pred_pool = df.index.tolist()[:-10]
    chosen_suc = rng.choice(succ_pool, size=min(pred_count, len(succ_pool)), replace=False)
    for s in chosen_suc:
        p = rng.choice([x for x in pred_pool if x < s])
        df.loc[s, "predecessor_wo_id"] = df.loc[p, "wo_id"]

    log.info(
        "Work orders: %d total | %d mandatory | %d with predecessors",
        len(df),
        df.mandatory.sum(),
        df.predecessor_wo_id.notna().sum(),
    )
    return df


# ─── Public entry point ───────────────────────────────────────────────────────
@timed
def generate_all(
    seed: int | None = None,
    n_work_orders: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate asset master, failure history, and work orders; save to CSV.

    Creates exactly ONE `np.random.Generator`, seeded from `seed` if given
    or otherwise from the LIVE value of `DGEN_CFG.random_seed` at the
    moment this function is CALLED (not whenever this module happened to
    be first imported), and threads that single generator through asset
    master, failure history, and work-order generation so the whole
    dataset is reproducible from one seed. `n_work_orders` similarly
    overrides DGEN_CFG.num_work_orders for this call only.
    """
    resolved_seed = DGEN_CFG.random_seed if seed is None else seed
    if resolved_seed < 0:
        raise ValueError(
            f"seed must be a non-negative integer, got {resolved_seed}. "
            "(numpy's random number generator requires seeds >= 0.)"
        )
    rng = np.random.default_rng(resolved_seed)

    assets = generate_asset_master(rng)
    failures = generate_failure_history(assets, rng)
    wos = generate_work_orders(assets, rng, n=n_work_orders)

    assets.to_csv(DATA_RAW / "asset_master.csv", index=False)
    failures.to_csv(DATA_RAW / "failure_history.csv", index=False)
    wos.to_csv(DATA_RAW / "work_orders.csv", index=False)

    log.info("✅  Raw data saved to %s (seed=%d)", DATA_RAW, resolved_seed)
    return assets, failures, wos


if __name__ == "__main__":
    generate_all()
