"""
test_helpers.py — Unit tests for logging/formatting/audit-trail utilities.
"""

from __future__ import annotations
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.utils.helpers import fmt_usd, fmt_pct, fmt_hours, write_run_log, timed, print_banner


class TestFormatters:
    def test_fmt_usd_includes_dollar_sign_and_commas(self):
        assert "$" in fmt_usd(1_234_567)
        assert "1,234,567" in fmt_usd(1_234_567)

    def test_fmt_pct_converts_fraction_to_percent(self):
        assert fmt_pct(0.4567) == "45.7 %"

    def test_fmt_hours_has_unit_suffix(self):
        assert fmt_hours(1500).strip().endswith("h")


class TestWriteRunLog:
    def test_writes_valid_json_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "audit_logs"
            metadata = {"budget_usd": 5_000_000, "tasks_selected": 222, "nested": {"a": 1}}
            log_path = write_run_log(out_dir, metadata)

            assert log_path.exists()
            with open(log_path) as fh:
                loaded = json.load(fh)
            assert loaded["budget_usd"] == 5_000_000
            assert loaded["nested"]["a"] == 1

    def test_creates_output_directory_if_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            nested_dir = Path(tmp) / "does" / "not" / "exist" / "yet"
            assert not nested_dir.exists()
            write_run_log(nested_dir, {"x": 1})
            assert nested_dir.exists()

    def test_handles_non_json_native_types_via_default_str(self):
        """Path objects, etc. aren't natively JSON-serializable — must not crash."""
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            metadata = {"output_path": Path("/some/file.xlsx"), "value": 3.14}
            log_path = write_run_log(out_dir, metadata)
            with open(log_path) as fh:
                loaded = json.load(fh)
            assert "some/file.xlsx" in loaded["output_path"]


class TestWriteRunLogNumpyTypeFidelity:
    """
    Regression tests for a real bug: numpy scalar types (np.int64,
    np.float64, np.bool_ — extremely common in any dict built from
    pandas/numpy reductions like .sum()/.mean()) used to be silently
    stringified by the old `default=str` JSON fallback. This didn't crash,
    which made it easy to miss, but it corrupted type fidelity in the
    audit trail: np.int64(222) became the JSON STRING "222" rather than
    the number 222, and most dangerously, np.bool_(False) became the
    string "False" — which is TRUTHY when reloaded and tested with
    `if value:` in Python, silently inverting the original boolean for any
    downstream consumer of the audit log.
    """

    def test_numpy_int_is_a_real_json_number_not_a_string(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = write_run_log(Path(tmp), {"count": np.int64(222)})
            raw = open(log_path).read()
            # The raw JSON text must contain a bare number, not a quoted string
            assert '"count": 222' in raw
            loaded = json.load(open(log_path))
            assert loaded["count"] == 222
            assert isinstance(loaded["count"], int)

    def test_numpy_float_is_a_real_json_number(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = write_run_log(Path(tmp), {"cost": np.float64(4999401.08)})
            loaded = json.load(open(log_path))
            assert isinstance(loaded["cost"], float)
            assert loaded["cost"] == pytest.approx(4999401.08)

    def test_numpy_bool_false_round_trips_as_false_not_truthy_string(self):
        """The specific dangerous case: a False value must reload as
        Python False, not as the truthy string "False"."""
        with tempfile.TemporaryDirectory() as tmp:
            log_path = write_run_log(Path(tmp), {"flag": np.bool_(False)})
            raw = open(log_path).read()
            assert '"flag": false' in raw  # real JSON boolean, not a quoted string
            loaded = json.load(open(log_path))
            assert loaded["flag"] is False
            assert bool(loaded["flag"]) is False  # would be True if it had become "False"

    def test_numpy_bool_true_round_trips_correctly(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = write_run_log(Path(tmp), {"flag": np.bool_(True)})
            loaded = json.load(open(log_path))
            assert loaded["flag"] is True

    def test_multi_element_numpy_array_falls_back_to_string(self):
        """Arrays (not scalars) can't be reduced via .item() — must still
        fall back to a safe string representation rather than crash."""
        with tempfile.TemporaryDirectory() as tmp:
            log_path = write_run_log(Path(tmp), {"values": np.array([1, 2, 3])})
            loaded = json.load(open(log_path))
            assert isinstance(loaded["values"], str)

    def test_mixed_numpy_and_native_and_path_types_all_correct(self):
        """The realistic case: a summary dict with a mix of numpy scalars,
        native Python types, and Path objects must all serialize with the
        correct respective JSON types in one pass."""
        with tempfile.TemporaryDirectory() as tmp:
            metadata = {
                "tasks_selected": np.int64(199),
                "budget_used_usd": np.float64(4999232.39),
                "mandatory": np.bool_(True),
                "excel_path": Path("/reports/export.xlsx"),
                "native_int": 42,
                "native_str": "OPTIMAL",
            }
            log_path = write_run_log(Path(tmp), metadata)
            loaded = json.load(open(log_path))
            assert loaded == {
                "tasks_selected": 199,
                "budget_used_usd": pytest.approx(4999232.39),
                "mandatory": True,
                "excel_path": "/reports/export.xlsx",
                "native_int": 42,
                "native_str": "OPTIMAL",
            }


class TestTimedDecorator:
    def test_preserves_return_value(self):
        @timed
        def add(a, b):
            return a + b

        assert add(2, 3) == 5

    def test_preserves_function_name(self):
        @timed
        def my_function():
            return None

        assert my_function.__name__ == "my_function"


class TestPrintBanner:
    def test_prints_without_raising(self, capsys):
        print_banner()
        captured = capsys.readouterr()
        assert "TURNAROUND" in captured.out
