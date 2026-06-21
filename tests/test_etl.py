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
