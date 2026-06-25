"""
test_load.py — Tests for the Parquet/CSV persistence layer
(save_processed / load_processed).

DATA_PROC is monkeypatched to a tmp_path for every test so these never
write into the real project's data/processed/ directory as a side effect.
"""

from __future__ import annotations
import pandas as pd
import pytest

import src.etl.load as load_mod
from src.etl.load import save_processed, load_processed


@pytest.fixture(autouse=True)
def _isolate_data_proc(monkeypatch, tmp_path):
    """Redirect DATA_PROC to a tmp_path for every test in this file."""
    monkeypatch.setattr(load_mod, "DATA_PROC", tmp_path)
    return tmp_path


class TestSaveProcessed:
    def test_writes_parquet_file(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        pq_path = save_processed(df, "my_table")
        assert pq_path == tmp_path / "my_table.parquet"
        assert pq_path.exists()

    def test_writes_csv_alongside_by_default(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2]})
        save_processed(df, "my_table")
        assert (tmp_path / "my_table.csv").exists()

    def test_also_csv_false_skips_csv(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2]})
        save_processed(df, "my_table", also_csv=False)
        assert not (tmp_path / "my_table.csv").exists()
        assert (tmp_path / "my_table.parquet").exists()

    def test_returns_parquet_path(self, tmp_path):
        df = pd.DataFrame({"a": [1]})
        returned = save_processed(df, "another_table")
        assert returned == tmp_path / "another_table.parquet"


class TestLoadProcessed:
    def test_round_trip_preserves_data(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [1.5, 2.5, 3.5], "c": ["x", "y", "z"]})
        save_processed(df, "round_trip_test")
        reloaded = load_processed("round_trip_test")
        pd.testing.assert_frame_equal(df, reloaded)

    def test_round_trip_preserves_bool_dtype(self):
        """Parquet (unlike CSV) must preserve boolean dtype exactly —
        this is the property that makes load_processed() safe for
        'selected'/'mandatory' columns elsewhere in the pipeline."""
        df = pd.DataFrame({"mandatory": [True, False, True]})
        save_processed(df, "bool_test")
        reloaded = load_processed("bool_test")
        assert reloaded["mandatory"].dtype == bool
        assert reloaded["mandatory"].tolist() == [True, False, True]

    def test_raises_clean_error_for_missing_file(self):
        with pytest.raises(FileNotFoundError, match="Processed file not found"):
            load_processed("does_not_exist")
