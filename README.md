# Plant Shutdown Turnaround Knapsack Optimizer

A production-grade 0-1 knapsack integer program that decides **which CMMS
work orders to execute during a plant shutdown/turnaround**, under hard
budget and craft-hour constraints, ranked by Weibull-derived failure risk
rather than gut feel or raw priority codes.

Built around the same problem every reliability and turnaround planning
team faces: the maintenance backlog always costs far more than the
available budget and crew capacity can cover. Someone has to decide what
gets deferred — this tool makes that decision optimal, auditable, and
defensible to leadership, instead of a spreadsheet sorted by priority.

```
550+ work orders  →  OR-Tools CP-SAT ILP  →  optimal subset under
budget + craft-hour constraints, ranked by Weibull-derived risk
```

Two enterprise-grade capabilities added on top of the core optimizer:

- **ERP Integration** (`src/erp/`): Mock SAP PM and IBM Maximo REST API
  servers with full OData/OSLC envelope shapes, plus connector adapters that
  translate vendor-specific field names (SAP's `MaintActivityType`, Maximo's
  `WOPRIORITY`) into the canonical schema — so swapping from CSV to a live
  ERP connection is one URL change, not a code rewrite.
- **Concurrent Scenario Management** (`src/scenarios/`): Save named planning
  scenarios ("Standard Budget", "15% Budget Cut"), lock them while editing
  (pessimistic advisory lock), protect concurrent updates with an optimistic
  version token (CAS UPDATE … WHERE version = :v), solve each, and compare
  any two scenarios side-by-side using `compare_scenarios()`.

## What this actually does

1. **Models failure risk, not just priority codes.** Each equipment class
   gets a Weibull reliability curve fit via maximum-likelihood estimation
   against historical failure data, producing a real conditional
   probability of failure within the planning horizon — not a 1-4 priority
   flag someone typed in six years ago.
2. **Converts risk into dollars.** Probability × multi-attribute
   consequence (safety/environmental/production/cost) × replacement value
   = a deferred-risk cost, directly comparable to the task's execution
   cost.
3. **Solves the actual combinatorial problem.** A 0-1 knapsack ILP
   (OR-Tools CP-SAT) selects the value-maximizing subset of work orders
   subject to budget, four separate craft-hour capacity constraints,
   mandatory-task forcing, and task-precedence ordering — proven optimal,
   not approximated.
4. **Reports it like an executive needs to see it.** A standalone
   interactive HTML dashboard and an 8-sheet Excel workbook structured as a
   direct Power BI data source.

## Quickstart

```bash
git clone <this-repo>
cd turnaround-optimizer
pip install -r requirements.txt

# Run with all defaults: generates 550 synthetic work orders, $5M budget
python run_optimizer.py

# Open the results
open dashboard/turnaround_dashboard.html      # interactive executive dashboard
open reports/power_bi_export.xlsx             # Power BI / Excel data source
```

```bash
# Override budget, craft-hour caps, and turnaround date
python run_optimizer.py \
  --budget 3500000 \
  --mech-hours 12000 --elec-hours 6000 --inst-hours 5000 --civil-hours 2000 \
  --turnaround-date 2027-03-15

# Force a fresh synthetic dataset with a different size/seed
python run_optimizer.py --regenerate-data --num-work-orders 800 --seed 7
```

Run `python run_optimizer.py --help` for the full flag reference.

## Architecture

```
                    ┌─────────────────┐
                    │  Synthetic CMMS  │   (or swap in a real CMMS export —
                    │  Data Generator  │    SAP PM, Maximo, IBM Maximo, etc.)
                    └────────┬─────────┘
                             ▼
   ┌─────────────────────────────────────────────────┐
   │  STAGE 1-2 · ETL                                  │
   │  extract.py  →  transform.py  →  load.py          │
   │  clean, validate, referential-integrity checks    │
   └────────────────────────┬──────────────────────────┘
                             ▼
   ┌─────────────────────────────────────────────────┐
   │  STAGE 3 · WEIBULL RELIABILITY MODEL              │
   │  MLE-fit β,η per equipment class                  │
   │  → conditional failure probability + RUL          │
   └────────────────────────┬──────────────────────────┘
                             ▼
   ┌─────────────────────────────────────────────────┐
   │  STAGE 4 · RISK SCORING                            │
   │  multi-attribute consequence × probability         │
   │  → 5×5 criticality matrix → deferred-cost $         │
   └────────────────────────┬──────────────────────────┘
                             ▼
   ┌─────────────────────────────────────────────────┐
   │  STAGE 5 · OPTIMIZATION                            │
   │  OR-Tools CP-SAT 0-1 knapsack ILP                  │
   │  budget + 4× craft-hour + mandatory + precedence   │
   └────────────────────────┬──────────────────────────┘
                             ▼
   ┌─────────────────────────────────────────────────┐
   │  STAGE 6 · REPORTING                               │
   │  Excel (Power BI feed)  +  standalone HTML          │
   │  dashboard (Plotly, 7 charts, sortable table)      │
   └────────────────────────┬──────────────────────────┘
                             ▼
   ┌─────────────────────────────────────────────────┐
   │  STAGE 7 · DATABASE PERSISTENCE                    │
   │  star-schema write (SQLite default, swappable to   │
   │  Postgres/MySQL/SQL Server) → Power BI live source  │
   └─────────────────────────────────────────────────┘
```

Every stage is independently unit-tested (see [Testing](#testing)) and the
core ILP solver has zero dependency on the upstream risk-scoring columns —
it can be tested, and used, with nothing but cost/hours/value columns.

## Project structure

```
turnaround-optimizer/
├── run_optimizer.py              # CLI entry point
├── requirements.txt
├── pyproject.toml                # pytest / black / flake8 / coverage config
├── Dockerfile                    # multi-stage build
├── docker-compose.yml
│
├── src/
│   ├── main.py                   # pipeline orchestrator (all 8 stages)
│   ├── utils/
│   │   ├── config.py             # all tunable parameters, env-var overridable
│   │   ├── helpers.py            # logging, timing, formatting
│   │   └── data_generator.py     # synthetic CMMS dataset (asset master,
│   │                              #   failure history, work orders)
│   ├── erp/                       # NEW: enterprise ERP integration layer
│   │   ├── mock_api.py            #   deterministic HTTP mock server (SAP PM +
│   │   │                          #   IBM Maximo) — zero runtime deps
│   │   └── connector.py           #   adapters that normalize OData/OSLC JSON
│   │                              #   to canonical schema; swap URL to go live
│   ├── etl/
│   │   ├── extract.py            # CSV / SQL / REST API loaders
│   │   ├── transform.py          # cleaning, validation, referential integrity,
│   │   │                          #   asset-name enrichment
│   │   └── load.py                # Parquet/CSV persistence
│   ├── modeling/
│   │   ├── weibull.py            # MLE fitting, failure probability, RUL
│   │   └── risk.py               # consequence scoring, criticality matrix,
│   │                              #   deferred-cost $ conversion
│   ├── optimization/
│   │   ├── knapsack_model.py     # CP-SAT model: variables, objective, constraints
│   │   └── solver.py              # solve orchestration + result extraction
│   ├── reporting/
│   │   ├── export.py              # 13-sheet Excel workbook (flat views +
│   │   │                          #   star-schema mirror sheets for Power BI)
│   │   └── dashboard.py           # standalone interactive HTML dashboard
│   ├── scenarios/                 # NEW: collaborative scenario management
│   │   ├── manager.py             #   create/lock/unlock/update/clone/archive
│   │   │                          #   DimScenario with optimistic CAS versioning
│   │   ├── runner.py              #   overlay scenario params → run full pipeline
│   │   └── comparison.py          #   side-by-side KPI diff + WO decision delta
│   └── db/
│       ├── schema.py               # SQLAlchemy star schema (1 fact + 6 dims,
│       │                          #   incl. new DimScenario + ScenarioStatus)
│       ├── connection.py           # engine factory: SQLite default, env-var
│       │                          #   swappable to Postgres/MySQL/SQL Server
│       ├── writer.py               # transactional run persistence (upsert +
│       │                          #   insert, commit-or-rollback)
│       └── queries.py              # read-back helpers (scenario comparison,
│                                  #   run history, joined fact lookups,
│                                  #   list_scenario_runs, scenario_kpi_history)
│
├── database/
│   └── turnaround.db              # SQLite file, created on first run
│                                  #   (point DATABASE_URL elsewhere for prod)
│
├── power_bi/
│   ├── README.md                  # 4 connection paths: Python script, ODBC,
│   │                              #   Excel star-schema, native Postgres/SQL Server
│   └── measures.dax               # copy-pasteable DAX measures
│
├── tests/                         # 369 tests — see Testing below
│   ├── test_etl.py                # incl. real SQLite query, mocked REST API
│   ├── test_load.py               # Parquet/CSV persistence round-trip
│   ├── test_helpers.py
│   ├── test_weibull.py
│   ├── test_risk.py
│   ├── test_optimizer.py          # constraint-satisfaction proofs
│   ├── test_export.py             # star-schema Excel sheet builder + real xlsx I/O
│   ├── test_dashboard.py          # HTML dashboard path-handling
│   ├── test_db.py                 # schema, transactional writer, multi-run
│   │                              #   idempotency, real-pipeline integration
│   ├── test_data_generator.py     # RNG threading + config-sentinel regression tests
│   ├── test_erp.py                # NEW: mock server lifecycle, SAP PM + Maximo
│   │                              #   endpoint shapes, connector normalisation,
│   │                              #   translation-table coverage, error handling
│   └── test_scenarios.py          # NEW: create/lock/unlock/update/clone/archive,
│                                  #   optimistic-CAS version conflict, concurrent
│                                  #   lock race (threading.Barrier), comparison
│
├── notebooks/                     # executed, outputs saved inline
│   ├── 01_data_exploration.ipynb
│   ├── 02_weibull_reliability_analysis.ipynb
│   └── 03_optimization_walkthrough.ipynb   # incl. budget sensitivity sweep
│
├── docs/
│   ├── METHODOLOGY.md             # full math + design rationale, incl. DB layer
│   ├── DATA_DICTIONARY.md         # every column, every stage
│   └── DATABASE_SCHEMA.md         # star-schema ERD (mermaid) + design notes
│
├── data/
│   ├── raw/                       # synthetic or real CMMS exports
│   ├── processed/                 # cleaned Parquet/CSV checkpoints per stage
│   └── external/
│
├── reports/
│   ├── power_bi_export.xlsx       # generated each run
│   └── audit_logs/                # timestamped JSON run log per execution
│
├── dashboard/
│   └── turnaround_dashboard.html  # generated each run
│
└── .github/workflows/ci.yml       # lint → test → smoke-test → docker-build
```

## The math, briefly

Full derivations live in [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md). The
short version:

**Weibull conditional failure probability** — given an asset has already
survived to its current age, the probability it fails within the next
planning horizon:

```
P(fail within horizon | survived to age) = [F(age+horizon) − F(age)] / [1 − F(age)]
```

**Deferred risk, converted to dollars:**

```
deferred_cost_usd = failure_prob × consequence_score × replacement_value × deferral_factor
net_value_usd     = deferred_cost_usd − estimated_cost_usd     # the ILP's objective coefficient
```

**The ILP itself:**

```
maximize    Σ net_value_i · x_i
subject to  Σ cost_i  · x_i ≤ Budget
            Σ hours_i,t · x_i ≤ Capacity_t      for each trade t ∈ {mech, elec, inst, civil}
            x_i = 1                             for every mandatory task
            x_j ≤ x_i                            whenever j requires predecessor i
            x_i ∈ {0, 1}
```

## Testing

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

209 tests across ten files, all passing, at 99% overall line coverage —
every single module in `src/` is at genuine 100% (including the ETL
extraction layer, the Parquet/CSV persistence layer, the HTML dashboard
renderer, and the Excel export's star-schema builder) except one
documented 4-line gap in the Weibull fitting fallback (see
`docs/METHODOLOGY.md` §7). The ones that matter most are in
`tests/test_optimizer.py`: rather than checking one example problem, they
run the solver across 10+ random seeds and problem sizes (scaling up to 600
tasks) and assert, every time:

- total spend never exceeds budget
- no trade's craft-hours ever exceed its cap
- every mandatory task is selected — even when its net value is negative
- a successor task is never selected without its predecessor
- the solver raises `RuntimeError` (rather than silently degrading) when
  mandatory tasks alone make the budget infeasible
- a pandas `<NA>` in the precedence column doesn't crash the model builder
  (a real bug caught during development — see `docs/METHODOLOGY.md` §3.3
  and the git history for the fix)

Twenty-six more real bugs were caught by writing this suite — not just
checking for exceptions, but diffing CLI output byte-for-byte and
inspecting actual computed values. A representative sample: an artificial
$5M "mandatory bonus" silently inflating every ROI metric by ~50× (§2.4 of
the methodology doc); three CLI flags (`--horizon-days`,
`--num-work-orders`, `--seed`) that looked like they worked — no errors,
config correctly mutated — but were silently ignored downstream due to
Python's classic "default argument evaluated once at import time" trap
(§5); any craft-hour cap or the budget set to exactly `0` crashing the
solver with an unhandled `ZeroDivisionError` (§6); a JSON audit log that
didn't crash but silently corrupted type fidelity — `np.bool_(False)` became
the *truthy* string `"False"` on reload (§6); three distinct concurrency
races that crashed or silently lost data when two threads wrote to the
database simultaneously (§4.5); two security vulnerabilities — the HTML
dashboard interpolated user-controlled strings without `html.escape()` and
the Excel export wrote formula-trigger values verbatim (§6); and seven more
found by reading every I/O boundary carefully: three more `ZeroDivisionError`
paths in the reporting layer, an empty schedule crashing the criticality
matrix builder with "cannot set a frame with no defined columns", an API
response containing HTML instead of JSON crashing with `JSONDecodeError`
rather than a clean error, API keys in query-string URLs logged in
plaintext, and a star-schema Excel sheet whose own docstring promised it
"mirrors" the database schema while silently missing a real column (§6).
The full list is in `docs/METHODOLOGY.md` §1–§6. Every one has a dedicated
regression test.

## Database & Power BI Integration

Every run is persisted into a star-schema database (`database/turnaround.db`
by default — see [`docs/DATABASE_SCHEMA.md`](docs/DATABASE_SCHEMA.md) for
the full ERD), not just exported to a flat file. The fact table is grained
at one row per `(run_id, wo_id)`, so re-running the optimizer with a
different budget or craft-hour cap adds a new comparable scenario instead
of overwriting the last one — this is what makes a live "budget
sensitivity" Power BI report page possible instead of a static, one-off
chart.

```bash
# Default: writes to local SQLite, auto-labeled by budget
python run_optimizer.py --budget 5000000

# Label scenarios explicitly for an easy Power BI scenario slicer
python run_optimizer.py --budget 3800000 --run-label "Reduced scope"
python run_optimizer.py --budget 6500000 --run-label "Expanded scope"

# Point at production Postgres/MySQL/SQL Server instead of local SQLite
python run_optimizer.py --database-url postgresql+psycopg2://user:pass@host:5432/turnaround

# Skip the database entirely — Excel + dashboard only
python run_optimizer.py --no-db
```

[`power_bi/README.md`](power_bi/README.md) covers four ways to connect,
ordered from zero-setup to production: a built-in Power BI Python-script
connector (recommended for SQLite, no driver install), a generic ODBC
bridge, the Excel workbook's star-schema mirror sheets, and a native
Postgres/SQL Server connection for Power BI Service scheduled refresh.
Copy-pasteable DAX measures for budget utilization, ROI, risk-score
reduction, and cross-scenario comparison are in
[`power_bi/measures.dax`](power_bi/measures.dax).

## Docker

```bash
docker build -t turnaround-optimizer .
docker run --rm \
  -v "$(pwd)/reports:/app/reports" \
  -v "$(pwd)/dashboard:/app/dashboard" \
  -v "$(pwd)/database:/app/database" \
  turnaround-optimizer --budget 5000000

# or, with docker-compose:
docker compose up --build
```

## Configuration

Every operational parameter is overridable via environment variable (see
`src/utils/config.py`) or CLI flag — nothing is hardcoded that a planner
would need to recompile to change:

| Env Var | CLI Flag | Default | Meaning |
|---|---|---|---|
| `TA_BUDGET` | `--budget` | 5,000,000 | Total turnaround budget (USD) |
| `TA_DATE` | `--turnaround-date` | 2026-10-01 | Turnaround start date |
| `TA_HORIZON` | `--horizon-days` | 365 | Planning horizon for failure probability |
| `TA_MECH_HRS` | `--mech-hours` | 15,000 | Mechanical craft-hour capacity |
| `TA_ELEC_HRS` | `--elec-hours` | 8,000 | Electrical craft-hour capacity |
| `TA_INST_HRS` | `--inst-hours` | 6,000 | Instrumentation craft-hour capacity |
| `TA_CIVIL_HRS` | `--civil-hours` | 2,500 | Civil craft-hour capacity |
| `SOLVER_TIMEOUT_S` | `--timeout` | 120 | Max CP-SAT solve time (seconds) |
| `SOLVER_WORKERS` | `--workers` | 4 | Parallel CP-SAT search workers |
| `ENABLE_DB_EXPORT` | `--no-db` (inverted) | true | Whether to persist each run to the database |
| `DATABASE_URL` | `--database-url` | (local SQLite) | SQLAlchemy connection string |
| `RUN_LABEL` | `--run-label` | (auto-generated) | Human-readable scenario label in `dim_run` |

The risk-model weights (safety/environmental/production/cost, and the
deferral-cost factor) are deliberately **not** CLI flags — they're
plant-specific risk-tolerance judgment calls that belong in
`src/utils/config.py`'s `RiskConfig` under change control, not something
to be casually overridden per run.

## ERP Integration (SAP PM & IBM Maximo)

`src/erp/` provides production-ready adapters for the two most common
industrial CMMS REST APIs, plus a zero-dependency mock server for local
development and CI.

### Try it — no ERP access required

```python
from src.erp.mock_api import MockERPServer
from src.erp.connector import load_from_erp

# Starts a realistic HTTP server on a free port — no mocking library needed
with MockERPServer() as server:
    # SAP PM: OData v2 envelope, real SAP field names (OrderId, FunctLocId, ...)
    sap_df = load_from_erp("sap_pm", server.base_url)

    # IBM Maximo: OSLC envelope, Maximo field names (WONUM, ASSETNUM, WOPRIORITY, ...)
    max_df = load_from_erp("maximo", server.base_url)

# Both return the same canonical schema — same columns as work_orders.csv
assert set(sap_df.columns) == set(max_df.columns)
```

### Point at a real ERP

```python
# SAP PM (real)
df = load_from_erp("sap_pm", "https://my-sap-host.example.com", token="your-oauth-token")

# IBM Maximo (real)
df = load_from_erp("maximo", "https://maximo.example.com", token="your-api-key")
```

The connector translates vendor-specific codes into the canonical vocabulary
automatically — SAP's priority `"1"` → `"Critical"`, Maximo's `WORKTYPE =
"OVHUL"` → `"Overhaul"` — so the Weibull/risk/ILP pipeline sees the same
schema regardless of source.

### What changes between mock and real

Only the `base_url` argument. No connector code changes, no schema changes,
no pipeline changes. The mock server serves realistic OData/OSLC envelopes
with the same field structure as real SAP PM 7.x and Maximo 7.6.1+.

## Collaborative Scenario Management

`src/scenarios/` lets multiple planners save, share, compare, and lock
named scenarios — each scenario wraps a specific set of optimizer parameters
and links to every run that was executed under it.

### Quick example

```python
from sqlalchemy import create_engine
from src.scenarios.manager import create_scenario, lock_scenario, clone_scenario
from src.scenarios.comparison import compare_scenarios

engine = create_engine("sqlite:///database/turnaround.db")

# Save two named scenarios
sid_a = create_scenario(engine, "Standard Budget", "alice", budget_usd=5_000_000)
sid_b = clone_scenario(engine, sid_a, "15% Budget Cut", "alice", budget_adjustment_pct=-15.0)
# → sid_b.budget_usd = 4,250,000; parent_scenario_id = sid_a

# Lock before editing — prevents concurrent clobber
lock_scenario(engine, sid_b, "alice")
# update parameters, then unlock...

# After solving both (via solve_scenario or the CLI --scenario-id flag):
comparison = compare_scenarios(engine, sid_a, sid_b)
print(comparison.summary_text())
# → side-by-side KPI table + lists of tasks added/removed between scenarios
```

### Concurrency model

Two independent mechanisms protect two independent things:

| Mechanism | Type | Protects | Implementation |
|---|---|---|---|
| `status` / `locked_by` / `locked_at` | Pessimistic / advisory | "Planner X is editing — wait" | Application-enforced |
| `version` (int) | Optimistic (CAS) | Two concurrent lock/edit requests racing | `UPDATE ... WHERE version = :v` |

The optimistic token closes the TOCTOU gap that the advisory flag alone
cannot prevent: two processes checking "is it locked?" and both seeing "no"
will still only succeed in locking once, because the second `UPDATE` finds
`version ≠ expected` and raises `ScenarioConflictError`.

## Connecting real CMMS data

Swap the synthetic generator for a real export by pointing
`src/etl/extract.py` at your source:

- **CSV export** (SAP PM, Maximo, etc.): already supported — match the
  column names in [`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md), or
  add a mapping step in `transform.py`.
- **Direct database connection**: `load_from_db()` accepts any
  SQLAlchemy-compatible connection string.
- **REST API (SAP PM / IBM Maximo)**: use `src/erp/connector.py` — see
  the ERP Integration section above.

The Weibull fitting stage needs genuine historical failure-time records to
produce trustworthy β/η — feeding it priority-code proxies instead of real
time-to-failure data will silently degrade the conditional-probability
math back into guesswork. If failure history isn't available for a class,
the model falls back conservatively to an exponential assumption rather
than fabricating a wear-out curve it has no evidence for.

## License

MIT — see `LICENSE`.
