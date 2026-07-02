"""
mock_api.py — Lightweight HTTP mock server simulating SAP PM and IBM Maximo REST APIs.

Purpose
-------
This module provides a realistic, zero-dependency simulation of the two most
common industrial CMMS REST APIs so that:

  1. The connector code (src/erp/connector.py) can be developed and tested
     without access to a real SAP PM or Maximo instance.
  2. CI pipelines and demos can show authentic field names, OData envelope
     structure, and pagination from each vendor's actual API shape.
  3. A new developer sees exactly what an enterprise integration looks like,
     and can swap in the real endpoint URL + credentials with zero connector
     changes.

The server is intentionally NOT a general-purpose mock framework — it only
serves the endpoints and response shapes that connector.py actually calls.

Architecture
------------
The server runs in a background daemon thread and is managed via the
``MockERPServer`` context manager, which binds to an OS-assigned free port
and shuts down cleanly on ``__exit__``.  The assigned port is available via
``server.port`` immediately after ``__enter__``, so tests can build the base
URL dynamically without any hard-coded port numbers.

SAP PM shape
------------
GET /sap/opu/odata/sap/PM_WORKORDER/MaintenanceOrder
    → OData JSON with ``d.results`` list, each object carrying real SAP PM
      field names (OrderId, FunctLocId, MaintActivityType, Priority, …)
GET /sap/opu/odata/sap/PM_WORKORDER/Equipment
    → Equipment master records with EquipmentId, EquipCategoryDesc, …

IBM Maximo shape
----------------
GET /maximo/oslc/os/mxwo
    → OSLC paged JSON with ``member`` list, each object carrying Maximo
      field names (WONUM, ASSETNUM, WORKTYPE, WOPRIORITY, …)
GET /maximo/oslc/os/mxasset
    → Asset master with ASSETNUM, ASSETTYPE, LOCATION, INSTALLDATE, …
"""

from __future__ import annotations

import json
import random
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse


# ─── Synthetic payload generators ────────────────────────────────────────────
# These functions generate realistic, deterministically-seeded lists of records
# using the same asset classes, task types, and priority conventions as the
# project's own synthetic data generator (src/utils/data_generator.py), so
# that the round-trip "ERP → connector.py → canonical schema → pipeline" can
# be exercised end-to-end in tests without needing to reconcile two different
# vocabularies.

_RNG = random.Random(1234)  # module-level seed for deterministic payloads

_SAP_ASSET_CLASSES = ["CMP", "HX", "PMP", "VLV", "VSL", "TWR", "INST", "ELEC", "TNK", "PPL"]
_SAP_PRIORITY_CODES = ["1", "2", "3", "4"]  # 1=Critical, 2=High, 3=Medium, 4=Low
_SAP_TASK_TYPES = ["PM01", "PM02", "PM03", "PM04", "PM05", "PM06", "PM07"]
_SAP_AREAS = ["Unit-100", "Unit-200", "Unit-300", "Tank-Farm", "Utilities"]

_MAXIMO_WORKTYPES = ["INSPC", "REPLC", "OVHUL", "CLEAN", "CALIB", "REPAIR", "TESTS"]
_MAXIMO_PRIORITIES = [1, 2, 3, 4]
_MAXIMO_ASSET_CLASSES = ["PMP", "HX", "CMP", "VLV", "VSL", "TWR", "INST", "ELEC", "TNK", "PPL"]


def _make_sap_work_orders(n: int = 40, seed: int = 1234) -> list[dict]:
    """
    Generate `n` synthetic SAP PM MaintenanceOrder records.

    Field names match real SAP PM OData API responses.  Values are seeded so
    the same call always returns the same payloads, making connector tests
    fully deterministic.
    """
    rng = random.Random(seed)
    orders = []
    for i in range(1, n + 1):
        asset_cls = rng.choice(_SAP_ASSET_CLASSES)
        priority = rng.choice(_SAP_PRIORITY_CODES)
        mech = round(rng.uniform(2, 40), 1)
        elec = round(rng.uniform(0, 20), 1)
        inst = round(rng.uniform(0, 15), 1)
        civil = round(rng.uniform(0, 8), 1)
        orders.append(
            {
                "OrderId": f"4{i:07d}",
                "OrderDescription": f"SAP order {i}: {asset_cls} scheduled maintenance",
                "FunctLocId": f"{asset_cls}-{i:04d}",
                "EquipCategory": asset_cls,
                "EquipCategoryDesc": asset_cls,
                "MaintActivityType": rng.choice(_SAP_TASK_TYPES),
                "Priority": priority,
                "PriorityDesc": {"1": "Very High", "2": "High", "3": "Medium", "4": "Low"}[priority],
                "Area": rng.choice(_SAP_AREAS),
                "BasicStartDate": f"2026-{rng.randint(1, 12):02d}-01",
                # Cost / effort planning
                "PlannedTotalCost": round(rng.uniform(5_000, 250_000), 2),
                "PlannedMechHours": mech,
                "PlannedElecHours": elec,
                "PlannedInstHours": inst,
                "PlannedCivilHours": civil,
                # Risk / consequence ratings (SAP PM FMEA fields)
                "SafetyConsequence": rng.randint(1, 5),
                "EnvConsequence": rng.randint(1, 5),
                "ProdConsequence": rng.randint(1, 5),
                "CostConsequence": rng.randint(1, 5),
                "ReplacementCost": round(rng.uniform(50_000, 2_000_000), 2),
                # Scheduling
                "IsMandatory": rng.random() < 0.09,
                "PredecessorOrderId": f"4{(i - 1):07d}" if i > 1 and rng.random() < 0.10 else None,
                # Weibull shape/scale carried from the equipment record
                "WeibullBeta": round(rng.uniform(1.2, 3.5), 3),
                "WeibullEta": round(rng.uniform(800, 4000), 1),
                "InstallDate": f"20{rng.randint(5, 20):02d}-{rng.randint(1, 12):02d}-01",
                "AssetAge": rng.randint(365, 5000),
            }
        )
    return orders


def _make_sap_equipment(orders: list[dict]) -> list[dict]:
    """
    Return one Equipment master record per unique FunctLocId in `orders`.
    In real SAP PM, equipment master data lives in a separate OData entity;
    the connector joins them by FunctLocId.
    """
    seen: dict[str, dict] = {}
    for o in orders:
        tag = o["FunctLocId"]
        if tag not in seen:
            seen[tag] = {
                "EquipmentId": tag,
                "EquipmentName": f"Equipment {tag}",
                "EquipCategory": o["EquipCategory"],
                "Area": o["Area"],
                "InstallDate": o["InstallDate"],
                "ReplacementCost": o["ReplacementCost"],
                "SafetyConsequence": o["SafetyConsequence"],
                "EnvConsequence": o["EnvConsequence"],
                "ProdConsequence": o["ProdConsequence"],
                "CostConsequence": o["CostConsequence"],
                "WeibullBeta": o["WeibullBeta"],
                "WeibullEta": o["WeibullEta"],
            }
    return list(seen.values())


def _make_maximo_work_orders(n: int = 40, seed: int = 5678) -> list[dict]:
    """
    Generate `n` synthetic IBM Maximo work-order records.

    Field names match real Maximo OSLC REST API responses.  Maximo uses
    ALL-CAPS field names and a flat JSON structure (no OData wrapping).
    """
    rng = random.Random(seed)
    orders = []
    for i in range(1, n + 1):
        asset_cls = rng.choice(_MAXIMO_ASSET_CLASSES)
        priority = rng.choice(_MAXIMO_PRIORITIES)
        total_hrs = round(rng.uniform(4, 80), 1)
        # Maximo stores labour by craft in separate LABOUR child records; we
        # simulate that by splitting total hours proportionally, which is
        # what real Maximo API responses expose after OSLC expand.
        # Ranges chosen so mech+elec+inst always ≤ 0.85 → civil ≥ 0.15 h.
        mech_frac = rng.uniform(0.45, 0.60)   # max 0.60
        elec_frac = rng.uniform(0.10, 0.15)   # max 0.15
        inst_frac = rng.uniform(0.05, 0.10)   # max 0.10  → sum max = 0.85
        mech = round(total_hrs * mech_frac, 1)
        elec = round(total_hrs * elec_frac, 1)
        inst = round(total_hrs * inst_frac, 1)
        civil = round(max(0.0, total_hrs - mech - elec - inst), 1)
        orders.append(
            {
                "WONUM": f"WO{i:05d}",
                "DESCRIPTION": f"Maximo WO {i}: {asset_cls} maintenance",
                "ASSETNUM": f"{asset_cls}-{i:04d}",
                "ASSETTYPE": asset_cls,
                "LOCATION": rng.choice(_SAP_AREAS),  # share area vocabulary
                "WORKTYPE": rng.choice(_MAXIMO_WORKTYPES),
                "WOPRIORITY": priority,
                # Maximo stores cost in ESTCOST and hours in ESTLABHRS
                "ESTCOST": round(rng.uniform(5_000, 250_000), 2),
                "ESTLABHRS": total_hrs,
                # Per-craft hours (from LABOUR child table, here flattened)
                "ESTMECHHRS": mech,
                "ESTELEKHRS": elec,
                "ESTINSTHRS": inst,
                "ESTCIVHRS": civil,
                # Risk fields
                "HAZARDID": rng.randint(1, 5),
                "ENVHAZARD": rng.randint(1, 5),
                "PRODIMPACT": rng.randint(1, 5),
                "COSTIMPACT": rng.randint(1, 5),
                "REPLACEMENTCOST": round(rng.uniform(50_000, 2_000_000), 2),
                # Flags
                "MANDATORY": rng.random() < 0.09,
                "PARENTWO": f"WO{(i - 1):05d}" if i > 1 and rng.random() < 0.10 else None,
                # Reliability fields
                "WEIBULLBETA": round(rng.uniform(1.2, 3.5), 3),
                "WEIBULLETA": round(rng.uniform(800, 4000), 1),
                "INSTALLDATE": f"20{rng.randint(5, 20):02d}-{rng.randint(1, 12):02d}-01",
                "ASSETAGE": rng.randint(365, 5000),
            }
        )
    return orders


def _make_maximo_assets(orders: list[dict]) -> list[dict]:
    """Return one asset master record per unique ASSETNUM in `orders`."""
    seen: dict[str, dict] = {}
    for o in orders:
        tag = o["ASSETNUM"]
        if tag not in seen:
            seen[tag] = {
                "ASSETNUM": tag,
                "DESCRIPTION": f"Asset {tag}",
                "ASSETTYPE": o["ASSETTYPE"],
                "LOCATION": o["LOCATION"],
                "INSTALLDATE": o["INSTALLDATE"],
                "REPLACEMENTCOST": o["REPLACEMENTCOST"],
                "HAZARDID": o["HAZARDID"],
                "ENVHAZARD": o["ENVHAZARD"],
                "PRODIMPACT": o["PRODIMPACT"],
                "COSTIMPACT": o["COSTIMPACT"],
                "WEIBULLBETA": o["WEIBULLBETA"],
                "WEIBULLETA": o["WEIBULLETA"],
            }
    return list(seen.values())


# ─── Pre-build payloads at import time ────────────────────────────────────────
# Built once at module level so every handler call returns the same content
# without re-generating, keeping per-request latency negligible in tests.

_SAP_ORDERS: list[dict] = _make_sap_work_orders(40, seed=1234)
_SAP_EQUIPMENT: list[dict] = _make_sap_equipment(_SAP_ORDERS)
_MAXIMO_ORDERS: list[dict] = _make_maximo_work_orders(40, seed=5678)
_MAXIMO_ASSETS: list[dict] = _make_maximo_assets(_MAXIMO_ORDERS)


def _odata_envelope(results: list[dict]) -> bytes:
    """Wrap `results` in a minimal OData v2 JSON envelope (SAP PM uses OData v2)."""
    return json.dumps({"d": {"results": results}}).encode()


def _oslc_envelope(members: list[dict], total_count: int | None = None) -> bytes:
    """Wrap `members` in a minimal OSLC JSON envelope (IBM Maximo uses OSLC)."""
    payload: dict = {"member": members}
    if total_count is not None:
        payload["oslc:totalCount"] = total_count
    return json.dumps(payload).encode()


# ─── HTTP request handler ─────────────────────────────────────────────────────


class _ERPRequestHandler(BaseHTTPRequestHandler):
    """Route GET requests to the appropriate SAP PM or Maximo payload."""

    # Map of path prefix → (content_bytes_fn)
    _ROUTES: dict[str, bytes | None] = {}

    def log_message(self, fmt, *args):  # silence access log during tests
        pass

    def do_GET(self):  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path

        if path.startswith("/sap/opu/odata/sap/PM_WORKORDER/MaintenanceOrder"):
            body = _odata_envelope(_SAP_ORDERS)
        elif path.startswith("/sap/opu/odata/sap/PM_WORKORDER/Equipment"):
            body = _odata_envelope(_SAP_EQUIPMENT)
        elif path.startswith("/maximo/oslc/os/mxwo"):
            body = _oslc_envelope(_MAXIMO_ORDERS, total_count=len(_MAXIMO_ORDERS))
        elif path.startswith("/maximo/oslc/os/mxasset"):
            body = _oslc_envelope(_MAXIMO_ASSETS, total_count=len(_MAXIMO_ASSETS))
        elif path == "/health":
            body = json.dumps({"status": "ok"}).encode()
        else:
            self.send_response(404)
            self.end_headers()
            return

        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


# ─── Context-manager server wrapper ──────────────────────────────────────────


class MockERPServer:
    """
    Context manager that runs a deterministic mock SAP PM / IBM Maximo HTTP
    server in a background daemon thread.

    Usage::

        with MockERPServer() as server:
            sap_url = f"http://localhost:{server.port}"
            df = load_from_sap_pm(sap_url)

    The server binds to ``localhost`` on an OS-assigned free port (port 0),
    so there are no hard-coded port numbers and multiple test processes can
    run concurrently without colliding.

    Attributes
    ----------
    port : int
        The port the server is listening on (available after ``__enter__``).
    base_url : str
        ``http://localhost:{port}`` shorthand.
    """

    def __init__(self) -> None:
        self._httpd: HTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.port: int = 0

    def __enter__(self) -> "MockERPServer":
        # Bind to localhost:0 → OS picks a free port.
        self._httpd = HTTPServer(("127.0.0.1", 0), _ERPRequestHandler)
        self.port = self._httpd.server_address[1]
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    # ── Convenience URL builders (match connector.py's endpoint patterns) ──

    def sap_orders_url(self) -> str:
        return f"{self.base_url}/sap/opu/odata/sap/PM_WORKORDER/MaintenanceOrder"

    def sap_equipment_url(self) -> str:
        return f"{self.base_url}/sap/opu/odata/sap/PM_WORKORDER/Equipment"

    def maximo_wo_url(self) -> str:
        return f"{self.base_url}/maximo/oslc/os/mxwo"

    def maximo_asset_url(self) -> str:
        return f"{self.base_url}/maximo/oslc/os/mxasset"
