"""
load.py — Persistence layer: save processed DataFrames to Parquet (fast)
and CSV (human-readable) in the data/processed directory.
"""

from __future__ import annotations
import pandas as pd
from pathlib import Path

from src.utils.config import DATA_PROC
from src.utils.helpers import get_logger, timed

log = get_logger("etl.load")


@timed
def save_processed(df: pd.DataFrame, name: str, also_csv: bool = True) -> Path:
    """Persist a DataFrame to Parquet (+ optional CSV) under data/processed/."""
    pq_path = DATA_PROC / f"{name}.parquet"
    df.to_parquet(pq_path, index=False, engine="pyarrow")
    log.info("Saved %s → %s (%d rows)", name, pq_path, len(df))

    if also_csv:
        csv_path = DATA_PROC / f"{name}.csv"
        df.to_csv(csv_path, index=False)

    return pq_path


def load_processed(name: str) -> pd.DataFrame:
    """Read a Parquet file from data/processed/."""
    path = DATA_PROC / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Processed file not found: {path}")
    return pd.read_parquet(path, engine="pyarrow")
