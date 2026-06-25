"""
test_etl.py — Unit tests for the ETL transform layer.
"""

import pandas as pd
import numpy as np
import pytest

from src.etl.transform import (
    clean_work_orders,
    clean_failure_history,
    add_priority_weight,
    add_wo_index_map,
    validate_referential_integrity,
    enrich_with_asset_name,
    run_transforms,
)


@pytest.fixture
def raw_wo_df():
    return pd.DataFrame(
        {
            "wo_id": ["WO-001", "WO-002", "WO-002", "WO-003"],  # WO-002 duplicated
            "description": ["A", "B", "B", "C"],
            "asset_tag": ["PMP-0001", "HX-0002", "HX-0002", "VLV-0003"],
            "asset_class": ["PMP", "HX", "HX", "VLV"],
            "area": [" unit-100 ", "Unit-200", "Unit-200", "unit-300"],
            "task_type": ["inspection", "REPAIR", "REPAIR", "overhaul"],
            "priority": ["critical", "high", "high", "medium"],
            "mandatory": [True, False, False, False],
            "estimated_cost_usd": [
                1000.0,
                -50.0,
                -50.0,
                2_000_000.0,
            ],  # negative + over-cap
            "mech_hours": [5.0, np.nan, np.nan, 10.0],  # NaN to fill
            "elec_hours": [0.0, 2.0, 2.0, 1.0],
            "inst_hours": [1.0, 0.0, 0.0, 0.0],
            "civil_hours": [0.0, 0.0, 0.0, 0.0],
            "total_craft_hours": [6.0, 2.0, 2.0, 11.0],
            "weibull_beta": [2.0, 2.0, 2.0, 2.0],
            "weibull_eta": [1000.0, 1000.0, 1000.0, 1000.0],
            "c_safety": [3, 4, 4, 2],
            "c_env": [2, 3, 3, 1],
            "c_prod": [4, 5, 5, 2],
            "c_cost": [3, 4, 4, 2],
            "replace_usd": [50000.0, 195000.0, 195000.0, 13500.0],
            "age_days": [-5.0, 800.0, 800.0, 200.0],  # negative age
            "predecessor_wo_id": [
                None,
                "WO-999",
                "WO-999",
                None,
            ],  # WO-999 doesn't exist
        }
    )


class TestCleanWorkOrders:
    def test_deduplicates_on_wo_id(self, raw_wo_df):
        out = clean_work_orders(raw_wo_df)
        assert out["wo_id"].is_unique
        assert len(out) == 3  # one duplicate removed

    def test_negative_cost_clamped_to_zero(self, raw_wo_df):
        out = clean_work_orders(raw_wo_df)
        assert (out["estimated_cost_usd"] >= 0).all()

    def test_cost_capped_at_one_million(self, raw_wo_df):
        out = clean_work_orders(raw_wo_df)
        assert out["estimated_cost_usd"].max() <= 1_000_000

    def test_nan_hours_filled_with_median(self, raw_wo_df):
        out = clean_work_orders(raw_wo_df)
        assert not out["mech_hours"].isna().any()

    def test_total_craft_hours_recalculated(self, raw_wo_df):
        out = clean_work_orders(raw_wo_df)
        recalced = out[["mech_hours", "elec_hours", "inst_hours", "civil_hours"]].sum(axis=1)
        pd.testing.assert_series_equal(
            out["total_craft_hours"].reset_index(drop=True),
            recalced.reset_index(drop=True),
            check_names=False,
        )

    def test_categoricals_title_cased(self, raw_wo_df):
        out = clean_work_orders(raw_wo_df)
        assert set(out["task_type"]) <= {"Inspection", "Repair", "Overhaul"}
        assert "Unit-100" in out["area"].values

    def test_invalid_predecessor_nullified(self, raw_wo_df):
        out = clean_work_orders(raw_wo_df)
        # WO-999 was never a valid wo_id, so it must be nulled out
        assert (
            out["predecessor_wo_id"].isna().all()
            or (out["predecessor_wo_id"].dropna() == "").all()
            or out["predecessor_wo_id"].notna().sum() == 0
        )

    def test_negative_age_clamped(self, raw_wo_df):
        out = clean_work_orders(raw_wo_df)
        assert (out["age_days"] >= 0).all()


class TestPriorityWeight:
    def test_priority_weight_mapping(self):
        df = pd.DataFrame({"priority": ["Critical", "High", "Medium", "Low", "Unknown"]})
        out = add_priority_weight(df)
        assert out["priority_weight"].tolist() == [4, 3, 2, 1, 1]


class TestWoIndexMap:
    def test_index_map_roundtrip(self):
        df = pd.DataFrame({"wo_id": ["WO-A", "WO-B", "WO-C"]})
        mapping = add_wo_index_map(df)
        assert mapping == {"WO-A": 0, "WO-B": 1, "WO-C": 2}


class TestCleanFailureHistory:
    def test_drops_nonpositive_ttf(self):
        df = pd.DataFrame({"time_to_failure_d": [-5, 0, 10, 500]})
        out = clean_failure_history(df)
        assert (out["time_to_failure_d"] > 0).all()
        assert len(out) == 2

    def test_drops_extreme_outliers(self):
        df = pd.DataFrame({"time_to_failure_d": [10, 500, 99999]})  # 99999 days ≈ 274 yr
        out = clean_failure_history(df)
        assert 99999 not in out["time_to_failure_d"].values


class TestValidateReferentialIntegrity:
    def test_flags_orphaned_asset_tags_without_dropping(self):
        wos = pd.DataFrame({"wo_id": ["WO-1", "WO-2"], "asset_tag": ["PMP-0001", "GHOST-9999"]})
        assets = pd.DataFrame({"asset_tag": ["PMP-0001", "HX-0002"]})
        out = validate_referential_integrity(wos, assets)
        # Orphaned row is FLAGGED, not removed — row count must be unchanged
        assert len(out) == 2
        assert bool(out.loc[out.wo_id == "WO-1", "asset_master_linked"].iloc[0])
        assert not bool(out.loc[out.wo_id == "WO-2", "asset_master_linked"].iloc[0])

    def test_all_linked_when_every_tag_is_known(self):
        wos = pd.DataFrame({"wo_id": ["WO-1", "WO-2"], "asset_tag": ["PMP-0001", "HX-0002"]})
        assets = pd.DataFrame({"asset_tag": ["PMP-0001", "HX-0002"]})
        out = validate_referential_integrity(wos, assets)
        assert out["asset_master_linked"].all()


class TestEnrichWithAssetName:
    def test_brings_in_asset_name_via_merge(self):
        wos = pd.DataFrame({"wo_id": ["WO-1"], "asset_tag": ["PMP-0001"], "asset_class": ["PMP"]})
        assets = pd.DataFrame({"asset_tag": ["PMP-0001"], "asset_name": ["Centrifugal Pump"]})
        out = enrich_with_asset_name(wos, assets)
        assert out.loc[0, "asset_name"] == "Centrifugal Pump"

    def test_falls_back_to_asset_class_for_orphaned_tags(self):
        wos = pd.DataFrame({"wo_id": ["WO-1"], "asset_tag": ["GHOST-9999"], "asset_class": ["PMP"]})
        assets = pd.DataFrame({"asset_tag": ["PMP-0001"], "asset_name": ["Centrifugal Pump"]})
        out = enrich_with_asset_name(wos, assets)
        # No match found -> falls back to the asset_class code rather than NaN
        assert out.loc[0, "asset_name"] == "PMP"

    def test_does_not_change_row_count(self):
        wos = pd.DataFrame(
            {
                "wo_id": ["WO-1", "WO-2"],
                "asset_tag": ["PMP-0001", "GHOST-9999"],
                "asset_class": ["PMP", "HX"],
            }
        )
        assets = pd.DataFrame({"asset_tag": ["PMP-0001"], "asset_name": ["Centrifugal Pump"]})
        out = enrich_with_asset_name(wos, assets)
        assert len(out) == 2


class TestRunTransformsWrapper:
    def test_returns_both_cleaned_tables(self):
        wos = pd.DataFrame(
            {
                "wo_id": ["WO-1"],
                "estimated_cost_usd": [1000.0],
                "mech_hours": [5.0],
                "elec_hours": [0.0],
                "inst_hours": [0.0],
                "civil_hours": [0.0],
                "total_craft_hours": [5.0],
                "priority": ["High"],
                "task_type": ["Repair"],
                "area": ["Unit-100"],
                "asset_class": ["PMP"],
                "predecessor_wo_id": [None],
            }
        )
        failures = pd.DataFrame({"time_to_failure_d": [100.0, 200.0, -5.0]})
        clean_wos, clean_fails = run_transforms(wos, failures)
        assert "priority_weight" in clean_wos.columns
        assert (clean_fails["time_to_failure_d"] > 0).all()


class TestExtractLoadFunctionsResolveLiveDataRaw:
    """
    Regression tests for the same stale-default-argument pattern fixed
    elsewhere (see docs/METHODOLOGY.md §5): load_work_orders/
    load_asset_master/load_failure_history previously declared
    `path: Path = DATA_RAW / "..."` directly in the signature, evaluated
    once at import time. Nothing in this codebase currently mutates
    DATA_RAW after import, so this was a latent consistency issue rather
    than an actively-triggered bug — fixed here for consistency with the
    project-wide remediation of this pattern class.
    """

    def test_load_work_orders_resolves_live_data_raw(self, monkeypatch, tmp_path):
        import src.etl.extract as extract_mod

        csv_path = tmp_path / "work_orders.csv"
        pd.DataFrame(
            {
                "wo_id": ["WO-1"],
                "asset_tag": ["PMP-0001"],
                "asset_class": ["PMP"],
                "area": ["Unit-100"],
                "task_type": ["Repair"],
                "priority": ["High"],
                "mandatory": [True],
                "predecessor_wo_id": [None],
                "install_date": ["2020-01-01"],
            }
        ).to_csv(csv_path, index=False)

        monkeypatch.setattr(extract_mod, "DATA_RAW", tmp_path)
        df = extract_mod.load_work_orders()  # path omitted -> resolves live DATA_RAW
        assert len(df) == 1
        assert df.iloc[0]["wo_id"] == "WO-1"

    def test_load_work_orders_explicit_path_overrides_data_raw(self, tmp_path):
        from src.etl.extract import load_work_orders

        csv_path = tmp_path / "custom_location.csv"
        pd.DataFrame(
            {
                "wo_id": ["WO-9"],
                "asset_tag": ["PMP-0009"],
                "asset_class": ["PMP"],
                "area": ["Unit-200"],
                "task_type": ["Inspection"],
                "priority": ["Low"],
                "mandatory": [False],
                "predecessor_wo_id": [None],
                "install_date": ["2021-01-01"],
            }
        ).to_csv(csv_path, index=False)

        df = load_work_orders(path=csv_path)
        assert df.iloc[0]["wo_id"] == "WO-9"

    def test_load_asset_master_accepts_plain_string_path(self, tmp_path):
        from src.etl.extract import load_asset_master

        csv_path = tmp_path / "asset_master.csv"
        pd.DataFrame({"asset_tag": ["PMP-0001"], "asset_class": ["PMP"]}).to_csv(csv_path, index=False)
        df = load_asset_master(path=str(csv_path))  # plain string, not Path
        assert len(df) == 1

    def test_load_asset_master_resolves_live_data_raw(self, monkeypatch, tmp_path):
        import src.etl.extract as extract_mod

        csv_path = tmp_path / "asset_master.csv"
        pd.DataFrame({"asset_tag": ["VLV-0003"], "asset_class": ["VLV"]}).to_csv(csv_path, index=False)

        monkeypatch.setattr(extract_mod, "DATA_RAW", tmp_path)
        df = extract_mod.load_asset_master()  # path omitted -> resolves live DATA_RAW
        assert df.iloc[0]["asset_tag"] == "VLV-0003"

    def test_load_failure_history_resolves_live_data_raw(self, monkeypatch, tmp_path):
        import src.etl.extract as extract_mod

        csv_path = tmp_path / "failure_history.csv"
        pd.DataFrame(
            {
                "asset_tag": ["PMP-0001"],
                "asset_class": ["PMP"],
                "time_to_failure_d": [500.0],
                "failure_date": ["2024-01-01"],
                "failure_mode": ["Wear"],
                "severity": [3],
            }
        ).to_csv(csv_path, index=False)

        monkeypatch.setattr(extract_mod, "DATA_RAW", tmp_path)
        df = extract_mod.load_failure_history()  # path omitted -> resolves live DATA_RAW
        assert len(df) == 1
        assert df.iloc[0]["time_to_failure_d"] == 500.0

    def test_load_failure_history_explicit_path(self, tmp_path):
        from src.etl.extract import load_failure_history

        csv_path = tmp_path / "custom.csv"
        pd.DataFrame(
            {
                "asset_tag": ["HX-0002"],
                "asset_class": ["HX"],
                "time_to_failure_d": [800.0],
                "failure_date": ["2023-06-01"],
                "failure_mode": ["Fouling"],
                "severity": [2],
            }
        ).to_csv(csv_path, index=False)
        df = load_failure_history(path=csv_path)
        assert df.iloc[0]["asset_tag"] == "HX-0002"


class TestLoadFromDb:
    """
    Tests load_from_db() against a REAL local SQLite engine — no network
    access required, since SQLAlchemy's sqlite:// dialect is purely
    file/memory-based. Exercises the actual query-execution path, not just
    a mock, while staying fully offline.
    """

    def test_executes_real_query_against_sqlite(self, tmp_path):
        from src.etl.extract import load_from_db
        from sqlalchemy import create_engine, text

        db_path = tmp_path / "test_cmms.db"
        engine = create_engine(f"sqlite:///{db_path}")
        with engine.begin() as conn:
            conn.execute(text("CREATE TABLE work_orders (wo_id TEXT, status TEXT)"))
            conn.execute(text("INSERT INTO work_orders VALUES ('WO-1', 'Planned'), ('WO-2', 'Complete')"))

        df = load_from_db(f"sqlite:///{db_path}", "SELECT * FROM work_orders WHERE status = 'Planned'")
        assert len(df) == 1
        assert df.iloc[0]["wo_id"] == "WO-1"

    def test_returns_empty_dataframe_for_no_matching_rows(self, tmp_path):
        from src.etl.extract import load_from_db
        from sqlalchemy import create_engine, text

        db_path = tmp_path / "test_cmms2.db"
        engine = create_engine(f"sqlite:///{db_path}")
        with engine.begin() as conn:
            conn.execute(text("CREATE TABLE work_orders (wo_id TEXT, status TEXT)"))

        df = load_from_db(f"sqlite:///{db_path}", "SELECT * FROM work_orders")
        assert len(df) == 0


class TestLoadFromApi:
    """
    Tests load_from_api() with requests.get mocked — no real network call.
    """

    def test_parses_list_response(self, monkeypatch):
        from src.etl import extract as extract_mod

        class _FakeResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return [{"wo_id": "WO-1", "cost": 1000}, {"wo_id": "WO-2", "cost": 2000}]

        def _fake_get(url, headers=None, timeout=None):
            assert url == "https://fake-cmms.example.com/api/work-orders"
            return _FakeResponse()

        monkeypatch.setattr(extract_mod.requests, "get", _fake_get)
        df = extract_mod.load_from_api("https://fake-cmms.example.com/api/work-orders")
        assert len(df) == 2
        assert df.iloc[0]["wo_id"] == "WO-1"

    def test_parses_wrapped_data_key_response(self, monkeypatch):
        """Some CMMS APIs wrap the array in a top-level 'data' key rather
        than returning a bare list — load_from_api must handle both."""
        from src.etl import extract as extract_mod

        class _FakeResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return {"data": [{"wo_id": "WO-9"}], "page": 1}

        monkeypatch.setattr(extract_mod.requests, "get", lambda *a, **k: _FakeResponse())
        df = extract_mod.load_from_api("https://fake-cmms.example.com/api/work-orders")
        assert df.iloc[0]["wo_id"] == "WO-9"

    def test_passes_bearer_token_in_headers(self, monkeypatch):
        from src.etl import extract as extract_mod

        captured = {}

        class _FakeResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return []

        def _fake_get(url, headers=None, timeout=None):
            captured["headers"] = headers
            return _FakeResponse()

        monkeypatch.setattr(extract_mod.requests, "get", _fake_get)
        extract_mod.load_from_api("https://fake-cmms.example.com/api", token="secret-token-123")
        assert captured["headers"] == {"Authorization": "Bearer secret-token-123"}

    def test_no_token_means_no_auth_header(self, monkeypatch):
        from src.etl import extract as extract_mod

        captured = {}

        class _FakeResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return []

        def _fake_get(url, headers=None, timeout=None):
            captured["headers"] = headers
            return _FakeResponse()

        monkeypatch.setattr(extract_mod.requests, "get", _fake_get)
        extract_mod.load_from_api("https://fake-cmms.example.com/api")
        assert captured["headers"] == {}
