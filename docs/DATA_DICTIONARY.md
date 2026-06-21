# Data Dictionary

Reference for every column that exists at some stage of the pipeline. Columns
are grouped by the pipeline stage that first introduces them.

## Raw Work Order Table (`data/raw/work_orders.csv`)

| Column | Type | Description |
|---|---|---|
| `wo_id` | string | Unique work-order identifier, e.g. `WO-00042` |
| `description` | string | Free-text task description |
| `asset_tag` | string | Foreign key to `asset_master.asset_tag` |
| `asset_class` | string | Equipment class code (`PMP`, `HX`, `CMP`, `VLV`, `VSL`, `TWR`, `INST`, `ELEC`, `TNK`, `PPL`) |
| `area` | string | Plant area / unit (e.g. `Unit-100`, `Tank-Farm`) |
| `install_date` | date | Asset installation date |
| `age_days` | float | Asset age in days as of the turnaround date |
| `task_type` | string | One of: Inspection, Replacement, Overhaul, Cleaning, Calibration, Repair, Testing |
| `priority` | string | Critical / High / Medium / Low |
| `mandatory` | bool | If `True`, the ILP forces this task into the plan regardless of value |
| `estimated_cost_usd` | float | Planning-level cost estimate |
| `mech_hours` / `elec_hours` / `inst_hours` / `civil_hours` | float | Craft-hour requirement by trade |
| `total_craft_hours` | float | Sum of the four trade-hour columns |
| `duration_days` | int | Rough task duration (`total_craft_hours / 8`, min 1) |
| `weibull_beta` / `weibull_eta` | float | Inline Weibull shape/scale carried over from the asset master |
| `c_safety` / `c_env` / `c_prod` / `c_cost` | int (1–5) | Raw consequence ratings by dimension |
| `replace_usd` | float | Full replacement value of the asset |
| `predecessor_wo_id` | string or null | If set, this task cannot run unless the referenced task is also selected |

## Asset Master Table (`data/raw/asset_master.csv`)

| Column | Type | Description |
|---|---|---|
| `asset_tag` | string | Primary key, e.g. `CMP-0007` |
| `asset_class` | string | Equipment class code |
| `asset_name` | string | Human-readable equipment name |
| `area` | string | Plant area |
| `install_date` | date | Installation date |
| `age_days` | int | Age as of the turnaround reference date |
| `weibull_beta` / `weibull_eta` | float | Class-level Weibull parameters (population baseline) |
| `c_safety` / `c_env` / `c_prod` / `c_cost` | int | Class-level consequence baseline |
| `replace_usd` | float | Replacement cost |

## Failure History Table (`data/raw/failure_history.csv`)

| Column | Type | Description |
|---|---|---|
| `asset_tag` | string | Foreign key to asset master |
| `failure_no` | int | Sequence number of this failure event for the asset |
| `time_to_failure_d` | float | Time-to-failure in days, used as raw input to Weibull MLE fitting |
| `failure_date` | date | Calendar date of the failure event |
| `failure_mode` | string | Wear, Corrosion, Fatigue, Fouling, Seal Failure, Bearing Failure, Leakage, Blockage |
| `severity` | int (1–5) | Severity of that specific historical event |

## Columns Added by the Weibull Modeling Stage

| Column | Type | Description |
|---|---|---|
| `fitted_beta` / `fitted_eta` | float | Final Weibull parameters used (class fit, asset-inline, or fallback) |
| `failure_prob` | float (0–1) | P(failure within the planning horizon \| survived to current age) |
| `rul_days` | float | Remaining useful life to a 10% reliability threshold |
| `weibull_source` | string | `class_fit`, `asset_inline`, or `asset_inline_fallback` — audit trail for which method produced the parameters |

## Columns Added by the Risk Scoring Stage

| Column | Type | Description |
|---|---|---|
| `consequence_score` | float (1–5) | Weighted average of the four consequence dimensions |
| `likelihood_tier` | int (1–5) | Binned `failure_prob` per the criticality-matrix convention |
| `consequence_tier` | int (1–5) | Rounded `consequence_score` |
| `risk_score` | int (1–25) | `likelihood_tier × consequence_tier` |
| `risk_level` | string | LOW / MEDIUM / HIGH / CRITICAL, derived from `risk_score` |
| `deferred_cost_usd` | float | Monetary value of the risk avoided by executing now vs. deferring |
| `net_value_usd` | float | `deferred_cost_usd − estimated_cost_usd` — the ILP objective coefficient |
| `priority_weight` | int (1–4) | Numeric encoding of the `priority` field |
| `asset_master_linked` | bool | `False` flags a work order whose `asset_tag` has no match in the asset master (data-quality signal, not auto-dropped) |

## Columns Added by the Optimization Stage

| Column | Type | Description |
|---|---|---|
| `selected` | bool | `True` if the ILP solver included this task in the turnaround plan |
| `decision` | string | `"INCLUDE"` or `"DEFER"` — human-readable mirror of `selected` |
