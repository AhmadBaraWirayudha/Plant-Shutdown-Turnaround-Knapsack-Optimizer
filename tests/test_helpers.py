"""
test_helpers.py — Unit tests for logging/formatting/audit-trail utilities.
"""

from __future__ import annotations
import json
import tempfile
from pathlib import Path

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
