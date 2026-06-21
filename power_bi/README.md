# Power BI Integration

This optimizer writes every run into a proper star-schema database
(`database/turnaround.db` by default — see
[`docs/DATABASE_SCHEMA.md`](../docs/DATABASE_SCHEMA.md) for the full ERD).
This guide covers every realistic way to get that data into Power BI,
ordered from "zero setup" to "production deployment."

## Which path should I use?

- **Just want to see it working right now, no installs** → Option C (Excel)
- **Want live SQLite refresh, already have Python installed** → Option A (recommended)
- **Want live SQLite refresh without Python on the report machine** → Option B (ODBC)
- **Deploying for a team, need Power BI Service scheduled refresh** → Option D (Postgres/SQL Server)

---

## Option A — Python script connector (recommended for SQLite)

Power BI Desktop has a built-in **Python script** data source. It runs a
Python script locally and imports every pandas DataFrame variable the
script creates as a table — no ODBC driver, no DSN configuration, nothing
to install beyond Python + pandas, which this project already requires.

**Setup (one time):**
1. `File > Options and settings > Options > Python scripting`, and confirm
   Power BI has detected your Python installation (the same environment
   you used to run `pip install -r requirements.txt`).
2. `Home > Get Data > Other > Python script > Connect`.
3. Paste this script, with the path adjusted to wherever your
   `database/turnaround.db` actually lives:

```python
import sqlite3
import pandas as pd

conn = sqlite3.connect(r"C:\path\to\turnaround-optimizer\database\turnaround.db")

dim_run = pd.read_sql("SELECT * FROM dim_run", conn)
dim_asset = pd.read_sql("SELECT * FROM dim_asset", conn)
dim_task_type = pd.read_sql("SELECT * FROM dim_task_type", conn)
dim_priority = pd.read_sql("SELECT * FROM dim_priority", conn)
dim_risk_level = pd.read_sql("SELECT * FROM dim_risk_level", conn)
fact_work_order_decision = pd.read_sql("SELECT * FROM fact_work_order_decision", conn)

conn.close()
```

4. Click OK. The Navigator window shows all six DataFrames as separate
   tables — select all of them, then **Load**.
5. Click **Refresh** any time after re-running the optimizer to pull in
   new runs.

This is genuinely the path of least resistance for SQLite specifically:
Power BI has no native SQLite connector at all (confirmed against current
Microsoft documentation — SQLite is only reachable via a third-party ODBC
driver), but Python support is first-class and built in.

**Limits to know about:** scripts time out at 30 minutes (irrelevant at
this data volume), must end with the data already in a pandas DataFrame
(no interactive prompts), and use absolute paths — Power BI's working
directory when it shells out to Python isn't reliably your project root.

---

## Option B — ODBC driver (no Python required on the report machine)

If the machine rendering the report shouldn't need a Python installation,
use a generic ODBC bridge instead:

1. Install a 64-bit SQLite ODBC driver — the most common is Christian
   Werner's [sqliteodbc](http://www.ch-werner.de/sqliteodbc/) (free); Devart
   also sells a commercial one with more configuration options.
2. Create a System DSN pointing at `database/turnaround.db` (Windows: search
   "ODBC Data Sources (64-bit)" → System DSN tab → Add).
3. In Power BI: `Get Data > Other > ODBC > Connect`, select the DSN.
4. The Navigator shows the same six tables — select all, **Load**.

For scheduled refresh through the Power BI **Service** (not just Desktop),
this path additionally needs an on-premises data gateway installed on a
machine that has the ODBC driver configured and stays powered on — Option D
avoids this entirely for a team deployment.

---

## Option C — Excel star-schema workbook (zero setup)

Every run also produces `reports/power_bi_export.xlsx`, which includes
`Dim_*` sheets alongside the flat human-browsable sheets (see the
[main README](../README.md)'s reporting section). `Get Data > Excel
workbook`, select every `Dim_*` sheet plus `FactWorkOrderDecision`, and
build the same relationships described below.

The tradeoff: Excel reflects only the **most recent** run (whatever was on
disk when the workbook was generated), not the full multi-scenario history
the database accumulates. Use the database path if scenario comparison
matters to you.

---

## Option D — Production deployment (Postgres / SQL Server)

Both have **native** Power BI connectors — no ODBC, no Python script, and
full Power BI Service scheduled-refresh support without a gateway (if
cloud-hosted). Point the optimizer at one of them instead of the SQLite
default:

```bash
pip install psycopg2-binary   # or: pip install pymysql / pyodbc
python run_optimizer.py --database-url postgresql+psycopg2://user:pass@host:5432/turnaround
```

Everything else — schema, writer logic, the `latest_run_facts` view — is
identical; only the connection string changes (see
`src/db/connection.py`). In Power BI: `Get Data > PostgreSQL database`,
enter the host and database name, select the six tables, **Load**.

---

## Setting up relationships (Model view)

Once the six tables are loaded (any option above), open **Model view** and
draw five relationships, all **single-direction, one-to-many, dimension →
fact**:

| From (one side) | To (many side) |
|---|---|
| `dim_run[run_id]` | `fact_work_order_decision[run_id]` |
| `dim_asset[asset_tag]` | `fact_work_order_decision[asset_tag]` |
| `dim_task_type[task_type_id]` | `fact_work_order_decision[task_type_id]` |
| `dim_priority[priority_id]` | `fact_work_order_decision[priority_id]` |
| `dim_risk_level[risk_level_id]` | `fact_work_order_decision[risk_level_id]` |

Power BI usually auto-detects these from the column names on first load —
check **Manage Relationships** to confirm all five exist and are active
before building visuals.

**One data-type gotcha:** SQLite has no native boolean type, so `selected`
and `mandatory` arrive as the **integers 0/1**, not `True`/`False` — this
was verified directly against the actual database file, not assumed. The
DAX measures in [`measures.dax`](measures.dax) are written to compare
against `1` for exactly this reason. If you'd rather work with `TRUE()` /
`FALSE()` in your own measures, convert the column type to Boolean in
Power Query's **Transform Data** step first.

---

## Suggested report pages

1. **Executive Summary** — KPI cards (Total Budget Used, Budget
   Utilization %, ROI Ratio, Tasks Selected) sliced by `dim_run[run_label]`,
   mirroring the existing HTML dashboard's KPI row.
2. **Budget Allocation** — waterfall or stacked bar of cost by
   `dim_task_type[task_type_name]` and `dim_asset[area]`, filtered to
   `selected = 1`.
3. **Risk & Criticality** — a matrix visual with `likelihood_tier` (rows) ×
   `consequence_tier` (columns), values = count, replicating the 5×5
   criticality heatmap from the Python dashboard but now natively
   cross-filterable by area, asset class, or run.
4. **Scenario Comparison** *(the capability that genuinely didn't exist in
   the flat Excel-only version)* — a line or bar chart of `roi_ratio` /
   `tasks_selected` / `budget_used_usd` across `dim_run[run_label]`, fed by
   real persisted runs instead of a one-off notebook sweep. Every time you
   run `run_optimizer.py --budget X --run-label "..."` with a new scenario,
   this page updates on the next refresh.
5. **Deferred Risk Register** — table of everything with `selected = 0`,
   sorted by `net_value_usd` descending, for the "what are we accepting
   risk on, and why" conversation with leadership.

Copy-pasteable DAX for all of the above is in
[`measures.dax`](measures.dax).
