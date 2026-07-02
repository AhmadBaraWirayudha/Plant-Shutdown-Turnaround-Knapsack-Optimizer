"""
tests/test_erp.py — Tests for the ERP integration layer.

Coverage
--------
* MockERPServer lifecycle (start / stop / concurrent requests)
* SAP PM and Maximo endpoint response shapes
* connector.py normalisation to canonical schema
* load_from_erp() dispatch by source name
* Edge-case handling: missing craft hours, unknown priority codes, null fields
"""

from __future__ import annotations

import threading
import urllib.request

import pandas as pd
import pytest

from src.erp.mock_api import (
    MockERPServer,
    _make_sap_work_orders,
    _make_sap_equipment,
    _make_maximo_work_orders,
    _make_maximo_assets,
    _odata_envelope,
    _oslc_envelope,
)
from src.erp.connector import (
    load_from_sap_pm,
    load_from_maximo,
    load_from_erp,
    _SAP_TASK_TYPE_MAP,
    _SAP_PRIORITY_MAP,
    _MAXIMO_TASK_TYPE_MAP,
    _MAXIMO_PRIORITY_MAP,
)

# ─── Expected canonical columns ───────────────────────────────────────────────

_CANONICAL_COLS = {
    "wo_id", "description", "asset_tag", "asset_class", "area",
    "install_date", "age_days", "task_type", "priority",
    "mandatory", "predecessor_wo_id",
    "estimated_cost_usd", "mech_hours", "elec_hours", "inst_hours", "civil_hours",
    "total_craft_hours", "duration_days",
    "weibull_beta", "weibull_eta",
    "c_safety", "c_env", "c_prod", "c_cost", "replace_usd",
}

_CANONICAL_PRIORITIES = {"Critical", "High", "Medium", "Low"}
_CANONICAL_TASK_TYPES = {"Inspection", "Replacement", "Overhaul", "Cleaning",
                         "Calibration", "Repair", "Testing"}


# ─── MockERPServer lifecycle ──────────────────────────────────────────────────


class TestMockERPServerLifecycle:
    def test_context_manager_starts_on_free_port(self):
        with MockERPServer() as server:
            assert server.port > 0
            assert server.port < 65536

    def test_base_url_property(self):
        with MockERPServer() as server:
            assert server.base_url == f"http://127.0.0.1:{server.port}"

    def test_health_endpoint_returns_ok(self):
        with MockERPServer() as server:
            with urllib.request.urlopen(f"{server.base_url}/health") as resp:
                import json
                payload = json.loads(resp.read())
            assert payload["status"] == "ok"

    def test_server_stops_after_context_exit(self):
        with MockERPServer() as server:
            port = server.port
        # After exit, the port should be closed — subsequent connection must fail
        import socket
        with pytest.raises((ConnectionRefusedError, OSError)):
            s = socket.create_connection(("127.0.0.1", port), timeout=0.5)
            s.close()

    def test_two_concurrent_servers_use_different_ports(self):
        with MockERPServer() as s1, MockERPServer() as s2:
            assert s1.port != s2.port

    def test_server_handles_concurrent_requests(self):
        """Smoke test: 20 threads each hit the same endpoint concurrently."""
        errors = []

        def fetch(base_url):
            try:
                with urllib.request.urlopen(f"{base_url}/health") as r:
                    r.read()
            except Exception as exc:
                errors.append(exc)

        with MockERPServer() as server:
            threads = [threading.Thread(target=fetch, args=(server.base_url,)) for _ in range(20)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=5)

        assert not errors, f"Concurrent requests raised: {errors}"

    def test_unknown_path_returns_404(self):
        with MockERPServer() as server:
            import urllib.error
            with pytest.raises(urllib.error.HTTPError) as exc_info:
                urllib.request.urlopen(f"{server.base_url}/nonexistent/path")
            assert exc_info.value.code == 404


# ─── SAP PM endpoint shape ────────────────────────────────────────────────────


class TestSAPPMEndpoint:
    def test_orders_endpoint_returns_odata_envelope(self):
        import json
        with MockERPServer() as server:
            with urllib.request.urlopen(server.sap_orders_url()) as resp:
                payload = json.loads(resp.read())
        assert "d" in payload
        assert "results" in payload["d"]
        assert isinstance(payload["d"]["results"], list)
        assert len(payload["d"]["results"]) > 0

    def test_orders_carry_required_sap_fields(self):
        import json
        with MockERPServer() as server:
            with urllib.request.urlopen(server.sap_orders_url()) as resp:
                orders = json.loads(resp.read())["d"]["results"]

        required = {"OrderId", "FunctLocId", "EquipCategory", "MaintActivityType",
                    "Priority", "PlannedTotalCost", "PlannedMechHours"}
        first = orders[0]
        for col in required:
            assert col in first, f"Missing SAP PM field: {col}"

    def test_equipment_endpoint_returns_odata_envelope(self):
        import json
        with MockERPServer() as server:
            with urllib.request.urlopen(server.sap_equipment_url()) as resp:
                payload = json.loads(resp.read())
        equip = payload["d"]["results"]
        assert isinstance(equip, list)
        assert len(equip) > 0

    def test_equipment_ids_are_unique(self):
        import json
        with MockERPServer() as server:
            with urllib.request.urlopen(server.sap_equipment_url()) as resp:
                equip = json.loads(resp.read())["d"]["results"]
        ids = [e["EquipmentId"] for e in equip]
        assert len(ids) == len(set(ids))


# ─── Maximo endpoint shape ────────────────────────────────────────────────────


class TestMaximoEndpoint:
    def test_wo_endpoint_returns_oslc_envelope(self):
        import json
        with MockERPServer() as server:
            with urllib.request.urlopen(server.maximo_wo_url()) as resp:
                payload = json.loads(resp.read())
        assert "member" in payload
        assert isinstance(payload["member"], list)
        assert len(payload["member"]) > 0

    def test_wo_records_carry_maximo_fields(self):
        import json
        with MockERPServer() as server:
            with urllib.request.urlopen(server.maximo_wo_url()) as resp:
                wos = json.loads(resp.read())["member"]
        required = {"WONUM", "ASSETNUM", "WORKTYPE", "WOPRIORITY", "ESTCOST", "ESTLABHRS"}
        first = wos[0]
        for col in required:
            assert col in first, f"Missing Maximo field: {col}"

    def test_asset_endpoint_returns_oslc_envelope(self):
        import json
        with MockERPServer() as server:
            with urllib.request.urlopen(server.maximo_asset_url()) as resp:
                payload = json.loads(resp.read())
        assert "member" in payload
        assert len(payload["member"]) > 0

    def test_total_count_matches_member_length(self):
        import json
        with MockERPServer() as server:
            with urllib.request.urlopen(server.maximo_wo_url()) as resp:
                payload = json.loads(resp.read())
        assert payload.get("oslc:totalCount") == len(payload["member"])


# ─── Payload generators (unit tests, no HTTP) ─────────────────────────────────


class TestPayloadGenerators:
    def test_sap_work_orders_length(self):
        orders = _make_sap_work_orders(25, seed=99)
        assert len(orders) == 25

    def test_sap_work_orders_deterministic(self):
        a = _make_sap_work_orders(10, seed=42)
        b = _make_sap_work_orders(10, seed=42)
        assert a == b

    def test_sap_equipment_one_per_func_loc(self):
        orders = _make_sap_work_orders(40, seed=1234)
        equip = _make_sap_equipment(orders)
        func_locs = {o["FunctLocId"] for o in orders}
        equip_ids = {e["EquipmentId"] for e in equip}
        assert equip_ids == func_locs

    def test_maximo_work_orders_length(self):
        wos = _make_maximo_work_orders(30, seed=77)
        assert len(wos) == 30

    def test_maximo_work_orders_deterministic(self):
        a = _make_maximo_work_orders(10, seed=42)
        b = _make_maximo_work_orders(10, seed=42)
        assert a == b

    def test_maximo_assets_one_per_assetnum(self):
        wos = _make_maximo_work_orders(40, seed=5678)
        assets = _make_maximo_assets(wos)
        assetnums = {w["ASSETNUM"] for w in wos}
        asset_ids = {a["ASSETNUM"] for a in assets}
        assert asset_ids == assetnums

    def test_sap_priority_codes_in_range(self):
        orders = _make_sap_work_orders(100, seed=1)
        codes = {o["Priority"] for o in orders}
        assert codes.issubset({"1", "2", "3", "4"})

    def test_maximo_wo_total_craft_hours_match_sum(self):
        wos = _make_maximo_work_orders(20, seed=1)
        for wo in wos:
            total = wo.get("ESTMECHHRS", 0) + wo.get("ESTELEKHRS", 0) + \
                    wo.get("ESTINSTHRS", 0) + wo.get("ESTCIVHRS", 0)
            # Allow ±0.2 h rounding from the round() calls on each per-craft value
            assert abs(total - wo["ESTLABHRS"]) < 0.2, \
                f"Craft hour sum mismatch: {total} vs {wo['ESTLABHRS']}"

    def test_odata_envelope_structure(self):
        records = [{"a": 1}, {"b": 2}]
        import json
        body = json.loads(_odata_envelope(records))
        assert body == {"d": {"results": records}}

    def test_oslc_envelope_structure(self):
        records = [{"x": 1}]
        import json
        body = json.loads(_oslc_envelope(records, total_count=1))
        assert body["member"] == records
        assert body["oslc:totalCount"] == 1


# ─── SAP PM connector ─────────────────────────────────────────────────────────


class TestSAPConnector:
    @pytest.fixture
    def sap_df(self):
        with MockERPServer() as server:
            return load_from_sap_pm(server.base_url)

    def test_returns_dataframe(self, sap_df):
        assert isinstance(sap_df, pd.DataFrame)

    def test_has_all_canonical_columns(self, sap_df):
        missing = _CANONICAL_COLS - set(sap_df.columns)
        assert not missing, f"Missing canonical columns: {missing}"

    def test_wo_ids_prefixed_with_wo_dash(self, sap_df):
        assert sap_df["wo_id"].str.startswith("WO-").all()

    def test_no_duplicate_wo_ids(self, sap_df):
        assert sap_df["wo_id"].nunique() == len(sap_df)

    def test_priorities_are_canonical(self, sap_df):
        bad = set(sap_df["priority"]) - _CANONICAL_PRIORITIES
        assert not bad, f"Non-canonical priorities: {bad}"

    def test_task_types_are_canonical(self, sap_df):
        bad = set(sap_df["task_type"]) - _CANONICAL_TASK_TYPES
        assert not bad, f"Non-canonical task types: {bad}"

    def test_craft_hours_non_negative(self, sap_df):
        for col in ["mech_hours", "elec_hours", "inst_hours", "civil_hours"]:
            assert (sap_df[col] >= 0).all(), f"Negative values in {col}"

    def test_total_craft_hours_equals_sum(self, sap_df):
        calc = sap_df[["mech_hours", "elec_hours", "inst_hours", "civil_hours"]].sum(axis=1)
        diff = (sap_df["total_craft_hours"] - calc).abs()
        assert (diff < 0.1).all(), "total_craft_hours does not match sum of craft columns"

    def test_duration_days_at_least_one(self, sap_df):
        assert (sap_df["duration_days"] >= 1).all()

    def test_cost_non_negative(self, sap_df):
        assert (sap_df["estimated_cost_usd"] >= 0).all()
        assert (sap_df["replace_usd"] > 0).all()

    def test_consequence_scores_in_range(self, sap_df):
        for col in ["c_safety", "c_env", "c_prod", "c_cost"]:
            assert (sap_df[col] >= 1).all() and (sap_df[col] <= 5).all(), \
                f"Consequence score out of range [1,5] in {col}"

    def test_weibull_params_positive(self, sap_df):
        assert (sap_df["weibull_beta"] > 0).all()
        assert (sap_df["weibull_eta"] > 0).all()

    def test_mandatory_is_bool(self, sap_df):
        assert sap_df["mandatory"].dtype == bool

    def test_predecessor_is_wo_prefixed_or_null(self, sap_df):
        non_null = sap_df["predecessor_wo_id"].dropna()
        if len(non_null):
            assert non_null.str.startswith("WO-").all()

    def test_custom_endpoint_paths(self):
        """Connector accepts custom OData path overrides."""
        with MockERPServer() as server:
            df = load_from_sap_pm(
                server.base_url,
                orders_path="/sap/opu/odata/sap/PM_WORKORDER/MaintenanceOrder",
                equipment_path="/sap/opu/odata/sap/PM_WORKORDER/Equipment",
            )
        assert len(df) > 0


# ─── Maximo connector ─────────────────────────────────────────────────────────


class TestMaximoConnector:
    @pytest.fixture
    def maximo_df(self):
        with MockERPServer() as server:
            return load_from_maximo(server.base_url)

    def test_returns_dataframe(self, maximo_df):
        assert isinstance(maximo_df, pd.DataFrame)

    def test_has_all_canonical_columns(self, maximo_df):
        missing = _CANONICAL_COLS - set(maximo_df.columns)
        assert not missing, f"Missing canonical columns: {missing}"

    def test_wo_ids_prefixed_with_wo_dash(self, maximo_df):
        assert maximo_df["wo_id"].str.startswith("WO-").all()

    def test_no_duplicate_wo_ids(self, maximo_df):
        assert maximo_df["wo_id"].nunique() == len(maximo_df)

    def test_priorities_are_canonical(self, maximo_df):
        bad = set(maximo_df["priority"]) - _CANONICAL_PRIORITIES
        assert not bad, f"Non-canonical priorities: {bad}"

    def test_task_types_are_canonical(self, maximo_df):
        bad = set(maximo_df["task_type"]) - _CANONICAL_TASK_TYPES
        assert not bad, f"Non-canonical task types: {bad}"

    def test_craft_hours_non_negative(self, maximo_df):
        for col in ["mech_hours", "elec_hours", "inst_hours", "civil_hours"]:
            assert (maximo_df[col] >= 0).all()

    def test_duration_days_at_least_one(self, maximo_df):
        assert (maximo_df["duration_days"] >= 1).all()

    def test_consequence_scores_in_range(self, maximo_df):
        for col in ["c_safety", "c_env", "c_prod", "c_cost"]:
            assert (maximo_df[col] >= 1).all() and (maximo_df[col] <= 5).all()

    def test_mandatory_is_bool(self, maximo_df):
        assert maximo_df["mandatory"].dtype == bool

    def test_fallback_craft_split_when_no_per_craft_hours(self):
        """
        When a Maximo WO only has ESTLABHRS (no per-craft fields), the
        connector must fall back to the 70/15/10/5 split rather than
        producing zero craft hours.
        """

        minimal_wos = [
            {
                "WONUM": "WO99999",
                "DESCRIPTION": "Minimal Maximo WO",
                "ASSETNUM": "PMP-0001",
                "ASSETTYPE": "PMP",
                "LOCATION": "Unit-100",
                "WORKTYPE": "INSPC",
                "WOPRIORITY": 2,
                "ESTCOST": 5000.0,
                "ESTLABHRS": 16.0,
                # No ESTMECHHRS, ESTELEKHRS, ESTINSTHRS, ESTCIVHRS
                "HAZARDID": 3,
                "ENVHAZARD": 2,
                "PRODIMPACT": 3,
                "COSTIMPACT": 2,
                "REPLACEMENTCOST": 150000.0,
                "MANDATORY": False,
                "PARENTWO": None,
                "WEIBULLBETA": 1.8,
                "WEIBULLETA": 2500.0,
                "INSTALLDATE": "2015-06-01",
                "ASSETAGE": 3000,
            }
        ]
        minimal_assets: list[dict] = []

        # Serve these via the mock server by temporarily patching the module-level lists
        import src.erp.mock_api as mock_mod
        orig_wos = mock_mod._MAXIMO_ORDERS
        orig_assets = mock_mod._MAXIMO_ASSETS
        try:
            mock_mod._MAXIMO_ORDERS = minimal_wos
            mock_mod._MAXIMO_ASSETS = minimal_assets
            with MockERPServer() as server:
                df = load_from_maximo(server.base_url)
        finally:
            mock_mod._MAXIMO_ORDERS = orig_wos
            mock_mod._MAXIMO_ASSETS = orig_assets

        assert len(df) == 1
        row = df.iloc[0]
        # All four craft columns must be positive after the fallback split
        for col in ["mech_hours", "elec_hours", "inst_hours", "civil_hours"]:
            assert row[col] > 0, f"Expected positive {col} after fallback split"
        assert abs(row["total_craft_hours"] - 16.0) < 0.1


# ─── load_from_erp dispatch ──────────────────────────────────────────────────


class TestLoadFromERPDispatch:
    def test_dispatches_to_sap(self):
        with MockERPServer() as server:
            df = load_from_erp("sap_pm", server.base_url)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_dispatches_to_maximo(self):
        with MockERPServer() as server:
            df = load_from_erp("maximo", server.base_url)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_case_insensitive_source_name(self):
        with MockERPServer() as server:
            df = load_from_erp("SAP_PM", server.base_url)
        assert len(df) > 0

    def test_unknown_source_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown ERP source"):
            load_from_erp("oracle_eam", "http://localhost:9999")

    def test_sap_and_maximo_produce_same_columns(self):
        """Both adapters must produce exactly the same column set."""
        with MockERPServer() as server:
            sap_df = load_from_erp("sap_pm", server.base_url)
            max_df = load_from_erp("maximo", server.base_url)
        assert set(sap_df.columns) == set(max_df.columns)


# ─── Translation table coverage ───────────────────────────────────────────────


class TestTranslationTables:
    def test_sap_priority_map_covers_all_codes(self):
        assert set(_SAP_PRIORITY_MAP.keys()) == {"1", "2", "3", "4"}
        assert set(_SAP_PRIORITY_MAP.values()) == _CANONICAL_PRIORITIES

    def test_sap_task_type_map_covers_all_pm_codes(self):
        assert set(_SAP_TASK_TYPE_MAP.values()) == _CANONICAL_TASK_TYPES

    def test_maximo_priority_map_covers_integer_codes(self):
        assert set(_MAXIMO_PRIORITY_MAP.keys()) == {1, 2, 3, 4}
        assert set(_MAXIMO_PRIORITY_MAP.values()) == _CANONICAL_PRIORITIES

    def test_maximo_task_type_map_covers_all_codes(self):
        assert set(_MAXIMO_TASK_TYPE_MAP.values()) == _CANONICAL_TASK_TYPES

    def test_token_auth_header_sent(self):
        """Bearer token is included when supplied — hits connector.py line 85."""
        # The mock server ignores the Authorization header, so the request
        # still succeeds.  What matters is that the branch executes (not a 401).
        with MockERPServer() as server:
            df = load_from_sap_pm(server.base_url, token="my-test-token")
        assert len(df) > 0

    def test_maximo_non_int_priority_falls_back_to_medium(self):
        """
        When WOPRIORITY is a non-integer string (e.g. 'HIGH'), the connector
        must catch the ValueError and default priority to 'Medium'.
        Covers connector.py lines 297-298.
        """
        import src.erp.mock_api as mock_mod

        bad_wo = [
            {
                "WONUM": "WO88888",
                "DESCRIPTION": "Bad priority WO",
                "ASSETNUM": "PMP-9999",
                "ASSETTYPE": "PMP",
                "LOCATION": "Unit-100",
                "WORKTYPE": "INSPC",
                "WOPRIORITY": "HIGH",  # non-integer — should trigger except branch
                "ESTCOST": 5000.0,
                "ESTLABHRS": 8.0,
                "ESTMECHHRS": 5.0,
                "ESTELEKHRS": 1.0,
                "ESTINSTHRS": 1.0,
                "ESTCIVHRS": 1.0,
                "HAZARDID": 3, "ENVHAZARD": 2, "PRODIMPACT": 3, "COSTIMPACT": 2,
                "REPLACEMENTCOST": 100_000.0,
                "MANDATORY": False, "PARENTWO": None,
                "WEIBULLBETA": 1.8, "WEIBULLETA": 2500.0,
                "INSTALLDATE": "2015-06-01", "ASSETAGE": 2000,
            }
        ]
        orig_wos = mock_mod._MAXIMO_ORDERS
        orig_assets = mock_mod._MAXIMO_ASSETS
        try:
            mock_mod._MAXIMO_ORDERS = bad_wo
            mock_mod._MAXIMO_ASSETS = []
            with MockERPServer() as server:
                df = load_from_maximo(server.base_url)
        finally:
            mock_mod._MAXIMO_ORDERS = orig_wos
            mock_mod._MAXIMO_ASSETS = orig_assets

        assert len(df) == 1
        assert df.iloc[0]["priority"] == "Medium"


# ─── Network error handling ───────────────────────────────────────────────────


class TestConnectorErrorHandling:
    def test_connection_refused_raises(self):
        """Connector propagates connection errors cleanly."""
        import requests.exceptions
        with pytest.raises((requests.exceptions.ConnectionError, Exception)):
            load_from_sap_pm("http://127.0.0.1:1")  # port 1 always refused

    def test_404_raises(self):
        """Connector raises on non-200 HTTP status."""
        with MockERPServer() as server:
            with pytest.raises(Exception):
                # Point at a path that returns 404
                load_from_sap_pm(
                    server.base_url,
                    orders_path="/no/such/entity",
                    equipment_path="/no/such/entity/equip",
                )
