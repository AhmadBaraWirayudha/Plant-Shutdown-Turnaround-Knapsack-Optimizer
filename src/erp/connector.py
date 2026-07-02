"""
connector.py — ERP integration adapters for SAP PM and IBM Maximo.

Each adapter:
  1. Calls the vendor's REST endpoint via ``requests``.
  2. Normalises the vendor-specific JSON fields into the canonical work-order
     schema used throughout this project (the same column names as
     ``data/raw/work_orders.csv`` — see docs/DATA_DICTIONARY.md).
  3. Returns a ``pd.DataFrame`` ready to pass directly into the ETL pipeline
     at the point where ``load_work_orders()`` would otherwise have been called.

Swap the base_url from the mock server to the real SAP PM / Maximo host (and
supply a token) and the rest of the pipeline works without any other change.

Field-mapping tables
--------------------
SAP PM uses integer priority codes and short type codes.  Maximo uses
ALL-CAPS numeric fields.  Both sets are translated into the project's own
vocabulary (Critical/High/Medium/Low priorities; Inspection/Replacement/…
task types) so that downstream code — the knapsack model, the dashboard,
the Excel export — never needs to know which system the data came from.
"""

from __future__ import annotations

import pandas as pd
import requests

from src.utils.helpers import get_logger, timed

log = get_logger("erp.connector")

# ─── Translation tables ───────────────────────────────────────────────────────

# SAP PM MaintActivityType code → canonical task_type
_SAP_TASK_TYPE_MAP: dict[str, str] = {
    "PM01": "Inspection",
    "PM02": "Replacement",
    "PM03": "Overhaul",
    "PM04": "Cleaning",
    "PM05": "Calibration",
    "PM06": "Repair",
    "PM07": "Testing",
}

# SAP PM Priority code → canonical priority
_SAP_PRIORITY_MAP: dict[str, str] = {
    "1": "Critical",
    "2": "High",
    "3": "Medium",
    "4": "Low",
}

# IBM Maximo WORKTYPE code → canonical task_type
_MAXIMO_TASK_TYPE_MAP: dict[str, str] = {
    "INSPC": "Inspection",
    "REPLC": "Replacement",
    "OVHUL": "Overhaul",
    "CLEAN": "Cleaning",
    "CALIB": "Calibration",
    "REPAIR": "Repair",
    "TESTS": "Testing",
}

# IBM Maximo WOPRIORITY (integer 1–4) → canonical priority
_MAXIMO_PRIORITY_MAP: dict[int, str] = {
    1: "Critical",
    2: "High",
    3: "Medium",
    4: "Low",
}


# ─── HTTP helpers ─────────────────────────────────────────────────────────────


def _get_json(url: str, token: str | None = None, timeout: int = 30) -> dict | list:
    """
    GET `url`, raise on any HTTP error, parse JSON, and surface a clear error
    on content-type mismatches (SAP's SM5x series returns HTML on auth errors,
    which is notoriously difficult to diagnose from a raw JSONDecodeError).
    """
    headers: dict[str, str] = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    headers["Accept"] = "application/json"

    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()

    try:
        return resp.json()
    except Exception as exc:  # pragma: no cover
        ct = resp.headers.get("Content-Type", "unknown")
        raise ValueError(
            f"Non-JSON response from {url!r} (Content-Type: {ct}). "
            "Check authentication — SAP SM5x and Maximo both return HTML on 401."
        ) from exc


# ─── SAP PM adapter ───────────────────────────────────────────────────────────


@timed
def load_from_sap_pm(
    base_url: str,
    token: str | None = None,
    *,
    orders_path: str = "/sap/opu/odata/sap/PM_WORKORDER/MaintenanceOrder",
    equipment_path: str = "/sap/opu/odata/sap/PM_WORKORDER/Equipment",
) -> pd.DataFrame:
    """
    Fetch and normalise SAP PM work orders into the canonical schema.

    Parameters
    ----------
    base_url
        Root URL of the SAP PM OData service, e.g.
        ``https://my-sap-host.example.com`` or ``http://localhost:PORT``
        when using the ``MockERPServer`` for testing.
    token
        Bearer token for API authentication (SAP OAuth2 / SAML token).
        Omit when connecting to the mock server (no auth required).
    orders_path
        OData entity path for maintenance orders.  Default matches the
        standard SAP PM OData API.
    equipment_path
        OData entity path for equipment master.  Joined on ``FunctLocId``
        to carry install_date and consequence ratings that live on the
        equipment record, not the order.

    Returns
    -------
    pd.DataFrame
        One row per work order with the same columns as ``work_orders.csv``.
    """
    # ── 1. Fetch orders + equipment ────────────────────────────────────────
    orders_payload = _get_json(f"{base_url}{orders_path}", token)
    equipment_payload = _get_json(f"{base_url}{equipment_path}", token)

    # SAP PM OData v2 envelope: {"d": {"results": [...]}}
    orders_raw: list[dict] = orders_payload["d"]["results"]
    equip_raw: list[dict] = equipment_payload["d"]["results"]

    log.info("SAP PM: fetched %d orders, %d equipment records", len(orders_raw), len(equip_raw))

    # ── 2. Build equipment lookup (FunctLocId → equipment row) ────────────
    equip_lookup: dict[str, dict] = {e["EquipmentId"]: e for e in equip_raw}

    # ── 3. Normalise to canonical schema ──────────────────────────────────
    rows = []
    for o in orders_raw:
        func_loc = o["FunctLocId"]
        equip = equip_lookup.get(func_loc, {})

        # Craft hours: SAP stores them directly on the order in real PM OData;
        # the mock follows the same pattern.
        mech = float(o.get("PlannedMechHours", 0) or 0)
        elec = float(o.get("PlannedElecHours", 0) or 0)
        inst = float(o.get("PlannedInstHours", 0) or 0)
        civil = float(o.get("PlannedCivilHours", 0) or 0)
        total_craft = round(mech + elec + inst + civil, 1)
        duration = max(1, round(total_craft / 8))

        pred_raw = o.get("PredecessorOrderId")
        predecessor = f"WO-{pred_raw}" if pred_raw else None

        rows.append(
            {
                # Identity
                "wo_id": f"WO-{o['OrderId']}",
                "description": str(o.get("OrderDescription", "")),
                "asset_tag": func_loc,
                "asset_class": o.get("EquipCategory") or equip.get("EquipCategory", "UNK"),
                "area": o.get("Area") or equip.get("Area", "Unknown"),
                # Reliability / age
                "install_date": str(o.get("InstallDate") or equip.get("InstallDate", "")),
                "age_days": float(o.get("AssetAge") or equip.get("AssetAge", 0) or 0),
                # Task classification
                "task_type": _SAP_TASK_TYPE_MAP.get(
                    str(o.get("MaintActivityType", "")), "Inspection"
                ),
                "priority": _SAP_PRIORITY_MAP.get(str(o.get("Priority", "3")), "Medium"),
                # Scheduling flags
                "mandatory": bool(o.get("IsMandatory", False)),
                "predecessor_wo_id": predecessor,
                # Cost and effort
                "estimated_cost_usd": float(o.get("PlannedTotalCost", 0) or 0),
                "mech_hours": mech,
                "elec_hours": elec,
                "inst_hours": inst,
                "civil_hours": civil,
                "total_craft_hours": total_craft,
                "duration_days": duration,
                # Weibull parameters (carried from equipment master in SAP)
                "weibull_beta": float(
                    o.get("WeibullBeta") or equip.get("WeibullBeta", 1.5) or 1.5
                ),
                "weibull_eta": float(
                    o.get("WeibullEta") or equip.get("WeibullEta", 2000) or 2000
                ),
                # Consequence ratings (ISO 31000 dimensions)
                "c_safety": int(
                    o.get("SafetyConsequence") or equip.get("SafetyConsequence", 3) or 3
                ),
                "c_env": int(
                    o.get("EnvConsequence") or equip.get("EnvConsequence", 2) or 2
                ),
                "c_prod": int(
                    o.get("ProdConsequence") or equip.get("ProdConsequence", 3) or 3
                ),
                "c_cost": int(
                    o.get("CostConsequence") or equip.get("CostConsequence", 2) or 2
                ),
                "replace_usd": float(
                    o.get("ReplacementCost") or equip.get("ReplacementCost", 100_000) or 100_000
                ),
            }
        )

    df = pd.DataFrame(rows)
    log.info("SAP PM: normalised %d work orders to canonical schema", len(df))
    return df


# ─── IBM Maximo adapter ───────────────────────────────────────────────────────


@timed
def load_from_maximo(
    base_url: str,
    token: str | None = None,
    *,
    wo_path: str = "/maximo/oslc/os/mxwo",
    asset_path: str = "/maximo/oslc/os/mxasset",
) -> pd.DataFrame:
    """
    Fetch and normalise IBM Maximo work orders into the canonical schema.

    Parameters
    ----------
    base_url
        Root URL of the Maximo Application Server, e.g.
        ``https://maximo.example.com`` or ``http://localhost:PORT`` for the mock.
    token
        API key or session token.  Maximo 7.6.1+ supports ``maxauth`` tokens;
        pass them here and they will be sent as a Bearer header.
    wo_path
        OSLC resource path for work orders (default matches Maximo's standard
        application API).
    asset_path
        OSLC resource path for assets (used to enrich work orders with asset-
        master data that Maximo stores separately).

    Returns
    -------
    pd.DataFrame
        One row per work order with the same columns as ``work_orders.csv``.
    """
    # ── 1. Fetch work orders + asset master ───────────────────────────────
    wo_payload = _get_json(f"{base_url}{wo_path}", token)
    asset_payload = _get_json(f"{base_url}{asset_path}", token)

    # Maximo OSLC envelope: {"member": [...], "oslc:totalCount": N}
    wo_raw: list[dict] = wo_payload.get("member", wo_payload)
    asset_raw: list[dict] = asset_payload.get("member", asset_payload)

    log.info("Maximo: fetched %d work orders, %d asset records", len(wo_raw), len(asset_raw))

    # ── 2. Build asset lookup (ASSETNUM → asset row) ──────────────────────
    asset_lookup: dict[str, dict] = {a["ASSETNUM"]: a for a in asset_raw}

    # ── 3. Normalise to canonical schema ──────────────────────────────────
    rows = []
    for wo in wo_raw:
        assetnum = wo.get("ASSETNUM", "")
        asset = asset_lookup.get(assetnum, {})

        mech = float(wo.get("ESTMECHHRS", 0) or 0)
        elec = float(wo.get("ESTELEKHRS", 0) or 0)
        inst = float(wo.get("ESTINSTHRS", 0) or 0)
        civil = float(wo.get("ESTCIVHRS", 0) or 0)
        total_craft = round(mech + elec + inst + civil, 1)
        if total_craft == 0:
            # Maximo sometimes only stores ESTLABHRS (total) without the split;
            # distribute 70/15/10/5 as a sensible default.
            total_craft = float(wo.get("ESTLABHRS", 8) or 8)
            mech = round(total_craft * 0.70, 1)
            elec = round(total_craft * 0.15, 1)
            inst = round(total_craft * 0.10, 1)
            civil = round(total_craft - mech - elec - inst, 1)
        duration = max(1, round(total_craft / 8))

        priority_raw = wo.get("WOPRIORITY", 3)
        try:
            priority_key = int(priority_raw)
        except (TypeError, ValueError):
            priority_key = 3

        parent_raw = wo.get("PARENTWO")
        predecessor = f"WO-{parent_raw}" if parent_raw else None

        rows.append(
            {
                # Identity
                "wo_id": f"WO-{wo['WONUM']}",
                "description": str(wo.get("DESCRIPTION", "")),
                "asset_tag": assetnum,
                "asset_class": wo.get("ASSETTYPE") or asset.get("ASSETTYPE", "UNK"),
                "area": wo.get("LOCATION") or asset.get("LOCATION", "Unknown"),
                # Reliability / age
                "install_date": str(wo.get("INSTALLDATE") or asset.get("INSTALLDATE", "")),
                "age_days": float(wo.get("ASSETAGE") or asset.get("ASSETAGE", 0) or 0),
                # Task classification
                "task_type": _MAXIMO_TASK_TYPE_MAP.get(
                    str(wo.get("WORKTYPE", "INSPC")), "Inspection"
                ),
                "priority": _MAXIMO_PRIORITY_MAP.get(priority_key, "Medium"),
                # Scheduling flags
                "mandatory": bool(wo.get("MANDATORY", False)),
                "predecessor_wo_id": predecessor,
                # Cost and effort
                "estimated_cost_usd": float(wo.get("ESTCOST", 0) or 0),
                "mech_hours": mech,
                "elec_hours": elec,
                "inst_hours": inst,
                "civil_hours": civil,
                "total_craft_hours": total_craft,
                "duration_days": duration,
                # Weibull parameters
                "weibull_beta": float(wo.get("WEIBULLBETA") or asset.get("WEIBULLBETA", 1.5) or 1.5),
                "weibull_eta": float(wo.get("WEIBULLETA") or asset.get("WEIBULLETA", 2000) or 2000),
                # Consequence ratings
                "c_safety": int(wo.get("HAZARDID") or asset.get("HAZARDID", 3) or 3),
                "c_env": int(wo.get("ENVHAZARD") or asset.get("ENVHAZARD", 2) or 2),
                "c_prod": int(wo.get("PRODIMPACT") or asset.get("PRODIMPACT", 3) or 3),
                "c_cost": int(wo.get("COSTIMPACT") or asset.get("COSTIMPACT", 2) or 2),
                "replace_usd": float(
                    wo.get("REPLACEMENTCOST") or asset.get("REPLACEMENTCOST", 100_000) or 100_000
                ),
            }
        )

    df = pd.DataFrame(rows)
    log.info("Maximo: normalised %d work orders to canonical schema", len(df))
    return df


# ─── Unified entry point ──────────────────────────────────────────────────────


def load_from_erp(
    source: str,
    base_url: str,
    token: str | None = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Dispatch to the appropriate ERP adapter by source name.

    Parameters
    ----------
    source
        One of ``"sap_pm"`` or ``"maximo"`` (case-insensitive).
    base_url
        Root URL of the ERP system's API.
    token
        API authentication token (optional for the mock server).
    **kwargs
        Forwarded verbatim to the adapter (e.g. ``orders_path`` for SAP PM).

    Returns
    -------
    pd.DataFrame
        Canonical work-order schema, ready for the transform stage.

    Example
    -------
    ::

        # Real SAP PM
        df = load_from_erp("sap_pm", "https://my-sap.example.com", token="...")

        # Real Maximo
        df = load_from_erp("maximo", "https://maximo.example.com", token="...")

        # Mock server (testing)
        with MockERPServer() as server:
            df = load_from_erp("sap_pm", server.base_url)
    """
    key = source.lower().strip()
    if key == "sap_pm":
        return load_from_sap_pm(base_url, token, **kwargs)
    elif key == "maximo":
        return load_from_maximo(base_url, token, **kwargs)
    else:
        raise ValueError(
            f"Unknown ERP source {source!r}. "
            "Supported values: 'sap_pm', 'maximo'."
        )
