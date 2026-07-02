# Methodology

Technical reference for the systems chained together in this pipeline:
Weibull reliability analysis, multi-attribute risk scoring, the 0-1
knapsack integer program, and the star-schema database persistence layer.
Each section states the formula or design, the engineering rationale, and
the specific implementation choice made in this codebase.

---

## 1. Weibull Reliability Analysis

### 1.1 Why Weibull instead of a constant failure rate

A constant hazard rate (the exponential distribution) implies a component is
exactly as likely to fail today as it was on day one — which is true for some
truly random failure modes, but false for almost every piece of rotating or
static mechanical equipment, where wear, fatigue, and corrosion accumulate
with age. The two-parameter Weibull distribution generalizes the exponential
by letting the hazard rate itself rise, fall, or stay flat over time,
controlled by the shape parameter β:

- **β < 1** — decreasing hazard. Early-life "infant mortality" failures
  (manufacturing defects, installation errors) that become less likely the
  longer the unit survives initial commissioning.
- **β = 1** — constant hazard. Reduces exactly to the exponential
  distribution; failures are memoryless.
- **β > 1** — increasing hazard. The classic wear-out regime. Most of the
  equipment classes in this model (pumps, compressors, pressure vessels)
  are fit with β > 1, which is consistent with published reliability
  literature for rotating and pressure-retaining equipment.

### 1.2 Core formulas

Probability density function:

```
f(t) = (β/η) · (t/η)^(β-1) · exp(-(t/η)^β)
```

Cumulative distribution function (probability of failure by time *t*):

```
F(t) = 1 - exp(-(t/η)^β)
```

η ("characteristic life" or "scale parameter") has a clean physical
interpretation regardless of β: it's the age at which **63.2%** of the
population is expected to have failed (this falls out of the math —
`F(η) = 1 - e^-1 ≈ 0.632`). The unit-test suite checks the solver's fitted
parameters against this identity directly.

### 1.3 Parameter estimation: Maximum Likelihood

For each equipment class, β and η are estimated via maximum-likelihood
estimation (`scipy.stats.weibull_min.fit`, location fixed at 0 to keep a
true two-parameter model) against that class's pooled time-to-failure
history. MLE is preferred over the classical median-rank / probability-plot
method because it makes full use of every data point rather than relying on
a linearized rank regression, and it has well-understood asymptotic
properties (consistency, efficiency) that are documented in the reliability
engineering literature.

**Fallback behavior:** with fewer than 3 historical failure observations for
a class, MLE fitting becomes unreliable (the likelihood surface is too
flat). The code falls back to an exponential assumption (β=1) anchored to
the empirical mean time-to-failure, which is a deliberately conservative,
auditable choice rather than letting an ill-conditioned optimizer return a
nonsensical shape parameter silently.

### 1.4 The quantity that actually matters: conditional failure probability

A raw `F(t)` answers "what fraction of a *fresh* population fails by age
*t*?" — not the question a turnaround planner is actually asking, which is:
*"given this specific asset has already survived to its current age, what's
the chance it fails within my upcoming planning window?"*

That's a conditional probability, derived directly from the survival
function:

```
P(fail within horizon | survived to age) = [F(age + horizon) - F(age)] / [1 - F(age)]
```

This correctly accounts for the fact that an asset which has already
survived past its characteristic life η carries a *higher*, not lower,
near-term failure probability under a wear-out (β > 1) model — the opposite
of what naively reading the unconditional CDF would suggest. This identity
is validated in `tests/test_weibull.py` against the closed-form definition.

### 1.5 Remaining Useful Life (RUL)

RUL is reported as the additional time (from the asset's current age) until
the survival function R(t) = 1 - F(t) drops to a 10% reliability threshold
— i.e., "how much runway is left before we'd consider this a 90%-confidence
failure candidate?" This is a deliberately conservative planning threshold,
solved numerically via Brent's method (`scipy.optimize.brentq`) since the
inverse of the Weibull survival function has no need for a closed form once
wrapped in this conditional framing.

---

## 2. Multi-Attribute Risk Scoring

### 2.1 Consequence score

Each work order's consequence is rated on four independent 1–5 dimensions
(safety, environmental, production, cost) and combined via a fixed weighted
average:

```
consequence = 0.40·Safety + 0.25·Environmental + 0.25·Production + 0.10·Cost
```

The safety weight dominates by design. This reflects a common convention in
process-safety risk matrices (API RP-580 / ISO 31000-aligned programs): a
task with any meaningful safety consequence should never be out-ranked by a
purely financial one, regardless of how the other three dimensions land.

### 2.2 Likelihood and consequence tiers, and the 5×5 criticality matrix

The continuous `failure_prob` (from §1.4) is binned into a 1–5 likelihood
tier using industry-typical probability bands (Rare / Unlikely / Possible /
Likely / Almost Certain), and the continuous consequence score is rounded to
its nearest integer tier. Their product forms a `risk_score` from 1–25 on
the standard 5×5 criticality matrix, which is then bucketed into
LOW / MEDIUM / HIGH / CRITICAL risk levels. This matrix is the same
visualization convention used in process-safety and asset-integrity
programs, chosen specifically so the dashboard's heatmap is immediately
legible to a reliability or process-safety engineer without translation.

### 2.3 Deferred-risk cost: converting risk into dollars

The ILP needs a single monetary objective, so probabilistic risk has to be
converted into an expected dollar value:

```
deferred_cost_usd = failure_prob × consequence_score × replace_usd × deferral_factor
```

`deferral_factor` (default 0.15) scales the full replacement value down to
represent the fraction of that value genuinely at stake from *deferring this
specific task* — a deferred inspection rarely costs the full replacement
value of the asset; it costs the expected fraction of that value attributable
to the incremental risk window being evaluated. This factor is the single
most sensitive judgment call in the model and is exposed as a config
parameter (`RiskConfig.deferral_cost_factor`) specifically so a reliability
engineer can recalibrate it against their own plant's loss history rather
than trusting a hardcoded assumption.

### 2.4 Net value: the ILP's objective coefficient

```
net_value_usd = deferred_cost_usd - estimated_cost_usd
```

This is deliberately **not** inflated with any artificial bonus for
mandatory tasks. Mandatory selection is enforced structurally in the ILP via
an explicit `x_i = 1` constraint (§3.3), completely independent of the
objective coefficient. A statutory inspection that costs more than its
computed deferred-risk value will correctly show a negative `net_value_usd`
— that's real information (it tells a planner *why* a task is expensive
relative to its measurable risk reduction, even though it still has to be
done), not a number to be hidden by inflating it. An earlier version of this
codebase added a $5M bonus to mandatory tasks specifically to "guarantee"
their selection; removing it (the bonus was redundant — the hard constraint
already guarantees selection) was the single most important correctness fix
in this project, since it had been silently inflating every downstream ROI
metric by roughly 50×. A regression test (`test_net_value_is_true_economics_no_artificial_inflation`)
guards against this reappearing.

---

## 3. The 0-1 Knapsack Integer Program

### 3.1 Why CP-SAT over a classical MILP solver

OR-Tools' CP-SAT is a constraint-programming solver with a SAT-based core,
which handles pure 0-1 (Boolean) decision variables natively and tends to
out-perform classical simplex-based MILP solvers on problems that are
*combinatorially* structured (knapsack, scheduling, assignment) rather than
continuous-relaxation-friendly. For an instance with 500+ binary variables
and a handful of linear constraints, CP-SAT routinely proves optimality in
well under a second, as demonstrated in `tests/test_optimizer.py`'s 600-task
scale test.

### 3.2 Integer scaling

CP-SAT requires integer coefficients. Costs and dollar values are already
whole-dollar integers after rounding. Craft-hours are floats, so they're
scaled by `HOUR_SCALE = 10` (storing tenths-of-an-hour as integers) before
being handed to the solver, and craft-hour capacity constraints are scaled
identically so the ratio — and therefore the constraint's meaning — is
preserved exactly.

### 3.3 Constraint formulation

```
maximize     Σ value_i · x_i

subject to   Σ cost_i  · x_i  ≤ Budget                      (budget)
             Σ mech_i  · x_i  ≤ MaxMechHours                 (mechanical craft-hours)
             Σ elec_i  · x_i  ≤ MaxElecHours                 (electrical craft-hours)
             Σ inst_i  · x_i  ≤ MaxInstHours                 (instrumentation craft-hours)
             Σ civil_i · x_i  ≤ MaxCivilHours                (civil craft-hours)
             x_i = 1                for every mandatory i    (forced inclusion)
             x_j ≤ x_i              whenever j requires i first (precedence)
             x_i ∈ {0, 1}           for all i
```

The mandatory and precedence constraints are both modeled as *hard*
constraints rather than objective penalties — there is no value of "penalty"
large enough that a hard safety requirement should ever be tradeable against
budget in this model. If the mandatory floor genuinely cannot fit within the
available budget or craft-hour caps, the solver correctly returns
`INFEASIBLE` rather than silently dropping a mandatory task to make the
numbers work, and `TurnaroundSolver.solve()` raises a `RuntimeError` with
actionable guidance instead of returning a "successful" but unsafe plan.
This is validated directly in
`test_infeasible_when_mandatory_exceeds_budget`.

### 3.4 Warm-start hint

Before invoking the solver, a greedy bang-per-buck heuristic (sort by
value/cost ratio descending, fill the budget greedily, forcing mandatory
tasks first) constructs a feasible incumbent solution that's passed to
CP-SAT via `add_hint()`. This doesn't change the proven-optimal answer, but
it gives the branch-and-bound search a strong starting bound immediately,
which matters more as problem size scales past a few thousand tasks.

---

## 4. Database Persistence & Power BI Integration

### 4.1 Why a star schema, not a flat dump

Every optimizer run is persisted into a proper star schema (`dim_run`,
`dim_asset`, `dim_task_type`, `dim_priority`, `dim_risk_level`, and a
single `fact_work_order_decision` table) rather than one wide table. The
full rationale and entity-relationship diagram are in
[`docs/DATABASE_SCHEMA.md`](DATABASE_SCHEMA.md); the short version is that
`dim_run` turns every execution into its own row, which is what makes
*scenario comparison* — "show me net value across these three budget
levels" — a live, queryable feature in Power BI rather than a one-off
notebook chart that goes stale the moment someone reruns the optimizer.

### 4.2 A bug only a second write could catch

`src/db/writer.py`'s `_upsert_dim_asset` originally read existing asset
tags via `session.scalars(select(DimAsset.asset_tag))` and then accessed
`row.asset_tag` on each result — but selecting a single column already
causes SQLAlchemy's `scalars()` to unwrap each row to the bare string
value, not an ORM entity with attributes. The bug was syntactically valid
and type-checked fine, and it was **completely invisible on the first
write** to a fresh database, because an empty `dim_asset` table means the
loop body that triggers the bad attribute access never executes at all.
It only surfaced on a *second* write, once the first write had left real
rows behind to iterate over — caught here specifically because the test
suite and manual verification both deliberately wrote multiple sequential
runs rather than testing a single execution in isolation. The fix was to
let `scalars()` return the bare strings it already produces:
`known_tags = set(session.scalars(select(DimAsset.asset_tag)))`. The
regression test (`test_dim_asset_does_not_duplicate_across_runs`) writes
the same schedule twice specifically to keep this class of bug caught
going forward — see §7 below for why "tested once" was never the bar.

### 4.3 Transactional integrity

`write_results_to_db()` wraps every upsert and insert in a single
SQLAlchemy session and commits once, at the end, or rolls back entirely on
any failure. This was verified directly, not just asserted in a docstring:
during development, the bug in §4.2 caused two real write attempts to fail
mid-transaction, and a direct SQLite inspection afterward confirmed
`dim_run` and `dim_asset` were left at exactly their pre-failure row
counts — zero partial rows, zero corruption — before the fix was applied
and the same scenarios were re-run successfully.

### 4.4 Silent truncation of long descriptions

`fact_work_order_decision.description` is a `String(500)` column —
necessary because Postgres/MySQL/SQL Server all strictly enforce declared
VARCHAR lengths (SQLite, notably, does not, which is exactly why this
needs to be tested explicitly against the actual column constraint rather
than just trusting SQLite to catch it). The writer used to silently slice
any longer description to fit, with zero indication anything was cut. The
synthetic data generator's descriptions max out around 40 characters, so
this never triggered for the shipped demo data — but the codebase
explicitly documents "Connecting real CMMS data" as a supported path
(README), and real-world CMMS free-text description fields routinely
carry technician notes well past 500 characters. `_insert_fact_rows()` now
logs a warning naming exactly how many descriptions were truncated
whenever it actually happens, so a real user gets visibility into the data
loss instead of a silently-clipped database column — the full,
untruncated text remains available regardless, in the Excel export and
HTML dashboard, neither of which has a length limit.

### 4.5 Concurrent write safety

Three distinct races in the database writer were found by running five
threads simultaneously against the same database file — a realistic
scenario for a small team running budget scenarios in parallel:

**`init_db()` TABLE ALREADY EXISTS race**: `Base.metadata.create_all()`
uses `checkfirst=True` internally but that is still a SELECT-then-CREATE
sequence; two threads racing through it can both see a table as absent and
both issue `CREATE TABLE dim_run`, producing an `OperationalError: table
already exists` before either could write anything. Fixed with a
`threading.Lock` at the module level, making `init_db()` genuinely atomic.

**Lookup-table upsert TOCTOU race**: Every "SELECT to check if row exists,
then INSERT if not found" upsert in `_upsert_lookup_task_types()` and its
siblings is a classic time-of-check-to-time-of-use race. Two concurrent
writers can both see a lookup value as absent and both attempt to insert it;
the UNIQUE constraint correctly prevents the duplicate, but the application
code previously let the resulting `IntegrityError` propagate and abort the
**entire outer transaction**, losing that writer's whole run. Fixed with
`_insert_or_recover()`, which wraps each insert in a `SAVEPOINT` (standard
SQL, supported across all four target backends). On `IntegrityError` it
rolls back just the savepoint, then re-queries to get the row the winning
writer already inserted — the outer transaction stays alive.

**`_create_latest_run_view()` DROP+CREATE race**: The view was recreated
outside the main transaction via its own `engine.begin()`. Two threads can
race through the DROP-then-CREATE sequence — thread A drops the view,
thread B also successfully drops it (already gone), thread B creates it,
thread A tries to create it and finds it already exists. Fixed by wrapping
the SQLite-dialect DROP+CREATE in a SAVEPOINT so only the inner savepoint
is lost on collision, and using `CREATE OR REPLACE VIEW` on
Postgres/MySQL/SQL Server (which support it natively).

**SQLite busy_timeout**: SQLite has a file-level write lock, and its
default 5-second timeout produced `OperationalError: database is locked`
under heavy concurrent load before the per-connection SAVEPOINT fix
reduced contention. The `busy_timeout` PRAGMA is now set to 30 seconds in
the SQLite event listener alongside `foreign_keys=ON`, giving reasonable
headroom for concurrent scenario sweeps. This is a SQLite-specific
mitigation — Postgres and SQL Server implement row-level locking and do not
have the same single-writer file-lock limitation.

## 5. CLI Overrides & the Stale-Default-Argument Bug Class

### 5.1 The bug, found by actually running the CLI twice

Three CLI flags — `--horizon-days`, `--num-work-orders`, and `--seed` —
looked like they worked: argparse accepted them without error, the
relevant config singleton (`TA_CFG` / `DGEN_CFG`) was correctly mutated,
and nothing crashed. But running the pipeline with two different values of
`--horizon-days` produced **bit-for-bit identical** `failure_prob` columns,
confirmed by directly diffing the output CSVs. `--num-work-orders 100`
silently produced the default 550 work orders. `--seed` only partially
worked — `asset_master.csv` and `failure_history.csv` were byte-identical
across two different seeds, while only `work_orders.csv` differed.

The root cause was the same in all three cases: a function signature like

```python
def run_weibull_analysis(work_orders, failure_history,
                          horizon_days: int = TA_CFG.planning_horizon_days):
```

looks like it reads the *current* value of `TA_CFG.planning_horizon_days`
every time the function is called. It does not. Python evaluates default
argument values exactly once — when the `def` statement itself executes,
i.e. the moment the module is first imported. Every CLI entry point in
this project imports the full pipeline (`from src.main import
run_pipeline`) **before** argparse runs and mutates the config singleton,
so by the time a CLI override happens, the stale default is already baked
into the function's `__defaults__` tuple, permanently, for the rest of the
process. The function call later in the pipeline that omits the argument
silently falls back to that frozen value, not the live one.

The `--seed` case had an extra wrinkle: `src/utils/data_generator.py` also
had a module-level `rng = np.random.default_rng(DGEN_CFG.random_seed)`,
created once at import time and shared by every generator function except
one (`generate_work_orders`'s asset sampling step happened to read
`DGEN_CFG.random_seed` live via `asset_master.sample(random_state=...)`,
which is why work orders alone responded to `--seed` while asset master
and failure history silently didn't).

### 5.2 The fix: explicit dependency injection, resolved at call time

Every affected default was changed to a `None` sentinel, resolved against
the live config value **inside the function body** (which executes at call
time, not def time):

```python
def run_weibull_analysis(work_orders, failure_history, horizon_days: int | None = None):
    if horizon_days is None:
        horizon_days = TA_CFG.planning_horizon_days
    ...
```

For the random-number generator specifically, the fix goes further than a
sentinel: the module-level `rng` global was removed entirely.
`generate_asset_master`, `generate_failure_history`, and
`generate_work_orders` now all take `rng: np.random.Generator` as a
required parameter, and `generate_all()` constructs exactly one generator
— seeded from the live `DGEN_CFG.random_seed` at the moment it's called —
and threads it through all three. This is strictly stronger than a
sentinel default would have been here, because a per-function `if rng is
None: rng = np.random.default_rng(DGEN_CFG.random_seed)` would have
silently created a *different* generator instance per function, breaking
the single continuous random stream that makes the whole dataset
reproducible from one seed.

The fix was applied to every instance of this pattern found across the
codebase, not just the three that were empirically confirmed broken — an
AST sweep (`ast.walk` over every function definition, flagging any default
argument that's an attribute access on a module-level name) found and
closed two more latent instances in `risk.py` (`consequence_score`'s
weight parameters and `deferred_risk_cost`'s `factor` parameter) that
nothing currently mutates after import, but would have broken silently the
moment a future CLI flag or config change tried to override them.

### 5.3 Regression tests

Each fix has a dedicated regression test that mutates the relevant config
singleton **mid-test**, calls the affected function twice with the
argument *omitted* both times, and asserts the two calls produce
genuinely different output — not just "doesn't raise an exception," which
is exactly the bar the original, broken code already cleared. See
`tests/test_weibull.py::TestHorizonDefaultResolvesLiveConfig`,
`tests/test_data_generator.py` (the whole file), and
`tests/test_risk.py::TestWeightDefaultsResolveLiveConfig`. The fixes were
also verified empirically against the real CLI a second time after the
fix, reproducing the exact diff-the-output-CSVs method that found the bug
in the first place, to confirm the actual `run_optimizer.py` entry point
— not just the underlying function in isolation — was fixed end to end.

## 6. Input Validation & Error Message Clarity

A few edge cases surfaced more bugs once the CLI was stress-tested with
deliberately unusual inputs rather than just the happy path:

**`--num-work-orders 0`** produced an empty, columnless `pd.DataFrame`
(zero iterations of the work-order generation loop never establishes any
columns), which then crashed much later — deep inside `pandas`, with a
`KeyError: 'priority'` traceback exposed directly to the CLI user, nowhere
near the actual cause. `generate_work_orders()` now validates `n > 0` up
front with a clear `ValueError`.

**A negative `--seed`** crashed with numpy's own internal validation
message ("expected non-negative integer"), with no indication of which
argument was at fault. `generate_all()` now validates the resolved seed
and raises an application-level message identifying the actual problem.

**CP-SAT's `INFEASIBLE` and `UNKNOWN` solver statuses** were sharing one
error message that always blamed "mandatory tasks exceed budget/hours."
That's correct for `INFEASIBLE` — a definitive proof no solution exists —
but actively misleading for `UNKNOWN`, which means the solver ran out of
time *before reaching any conclusion at all* and says nothing about
whether a solution exists. A perfectly feasible, generously-budgeted
problem given too short a `--timeout` returns `UNKNOWN`, and the old
message would have sent a confused user to increase the budget instead of
the timeout. The two statuses now produce distinct, accurate messages —
verified with a dedicated test that forces `UNKNOWN` via
`max_solve_seconds=0.0` on an easily-feasible instance and asserts the
message blames timeout, not budget (`tests/test_optimizer.py::TestUnknownVsInfeasibleErrorMessages`).

`run_optimizer.py`'s top-level exception handler also previously caught
only `RuntimeError`, so any `ValueError` raised by the new input-validation
checks above would still have surfaced as a raw, unhandled Python
traceback despite the underlying message being clean — the handler now
catches both, and the generic "Common causes: increase budget/hours"
footer that used to be appended unconditionally was removed, since it had
become actively contradictory for the `UNKNOWN` case (the error message
explicitly says "this does NOT mean your constraints are infeasible,"
immediately followed by a footer recommending budget/hours fixes anyway).

**Any of `total_budget`, `max_mech_hours`, `max_elec_hours`,
`max_inst_hours`, or `max_civil_hours` set to `0`** — a legitimate
configuration (a turnaround with genuinely no planned civil-craft work,
for instance) — crashed `_extract_results()` with an unhandled
`ZeroDivisionError` while computing the utilisation percentage, before the
solver even returned a result. A 0-capacity trade forces the matching
constraint `Σ hours_i · x_i ≤ 0`, which in turn forces `used` to also be
exactly 0 for any feasible solve, so 0/0 here always genuinely means
"0% utilised" rather than an undefined edge case. A `_safe_ratio()` helper
now returns that 0.0 directly instead of dividing. While fixing this, a
second, more subtle bug in the same code surfaced: the old `roi_ratio`
calculation guarded its own division by `max(budget_used, 1)` rather than
checking for zero — which doesn't crash, but when `budget_used` is
genuinely 0, silently returns the raw dollar value of `total_value`
mislabeled as a "ratio" (e.g. `roi_ratio == 5000.0` meaning $5,000, not
5000×). Both are now unified under the same `_safe_ratio()` helper.

**`export_to_excel()` and `generate_dashboard()`'s `out_path` parameters**
crashed with `AttributeError: 'str' object has no attribute 'parent'` if
a caller passed a plain string rather than a `pathlib.Path` — a completely
natural thing to do, since most file-path-accepting Python APIs are
duck-typed to accept either. Every test written for these two functions
up to that point had used the `tmp_path` pytest fixture, which always
yields a `Path`, so the gap was invisible despite 100% line coverage on
both files — a reminder that line coverage proves a line executed, not
that every realistic *input type* to that line was exercised. Both
parameters now coerce their input via `Path(out_path)`, and both also had
the same stale-default-argument pattern as the rest of §5
(`out_path: Path = REPORTS_DIR / "..."`, evaluated once at import time) —
fixed with the same `None`-sentinel pattern, alongside the same fix
applied to `load_work_orders`/`load_asset_master`/`load_failure_history`'s
`DATA_RAW`-derived defaults for full consistency, even though nothing
currently mutates `DATA_RAW`/`REPORTS_DIR`/`DASHBOARD_DIR` after import.

**`load_from_api()` imported `requests`**, a library never declared in
`requirements.txt` at all. The function is a real, documented extension
point (see "Connecting real CMMS data" in the README), not hypothetical
pseudocode, so a fresh `pip install -r requirements.txt` followed by an
attempt to actually use it would fail with an `ImportError` nowhere
mentioned in the dependency list. Added `requests` as a declared
dependency, moved it (and `sqlalchemy`, similarly previously
function-local "in case it's not installed" despite being a hard
requirement everywhere else in this codebase) to proper module-level
imports, and added both to the Docker healthcheck's import list alongside
the rest of the genuinely-required dependencies.

**The JSON audit log silently corrupted numeric and boolean type
fidelity.** `write_run_log()`'s `json.dump(..., default=str)` looks
harmless — it doesn't crash on anything — but numpy scalar types
(`np.int64`, `np.float64`, `np.bool_`, all extremely common in any dict
built from pandas/numpy reductions like `.sum()`/`.mean()`) aren't natively
JSON-serializable, so the `default=str` fallback silently stringified
them: `np.int64(222)` became the JSON string `"222"` rather than the
number `222`. Far more dangerously, `np.bool_(False)` became the string
`"False"` — which is **truthy** when reloaded and tested with `if value:`
in Python, silently inverting the original boolean's meaning for any
downstream consumer of the audit trail. In current usage this never
actually triggered, since `solver.py`'s summary dict is already
disciplined about casting every value to a native Python type before it
reaches the audit log — but `write_run_log()` is a general-purpose utility
that was safe only by accident, not by design, the moment any future
caller passed unconverted pandas/numpy output. A custom `default` hook now
converts numpy scalars via `.item()` first (preserving the correct JSON
type) and only falls back to `str()` for genuinely non-serializable
objects like `Path`.

it reports and is visible only to the reader as ordinary text with no
formula side effects.

**Dashboard HTML rendered user-controlled strings without escaping.**
`_build_table_rows()` interpolated `wo_id`, `asset_tag`, `area`,
`task_type`, `priority`, `decision`, and `risk_level` directly into the
HTML string with no `html.escape()` call. A real CMMS work-order whose
`area` field happened to contain `</td><script>alert("xss")</script>`
would execute as JavaScript in any browser opening the dashboard — not a
theoretical risk, since CMMS description and area fields are free-text
that operators edit directly. Every string column is now passed through
`html.escape()` before interpolation, and a dedicated regression test
injects a `<script>` tag and asserts it appears as `&lt;script&gt;` in
the output, not as executable markup.

**Excel spreadsheets rendered user-controlled strings without escaping.**
The classic "CSV/spreadsheet formula injection" vulnerability
(OWASP-listed): `openpyxl` writes string cell values verbatim, and Excel
treats any value beginning with `=`, `+`, `-`, or `@` as a formula to
execute. A work-order `wo_id` of `=HYPERLINK("http://evil.com","click")`
would silently open a browser window — or, with DDE formulas, execute
arbitrary commands — when the exported workbook was opened. A
`_sanitise_formula_injection()` helper now prefixes any formula-starting
value with a single apostrophe (Excel's conventional "treat as plain
text" escape, invisible to the reader) before every DataFrame is written
to the workbook.

**Three more division-by-zero risks in the reporting layer.** The Excel
export's `CapacityUtilization` sheet computed `used / capacity * 100`
without protecting against `capacity = 0`, and the `ByEquipmentClass`
sheet's ROI column used `total_value / total_cost.replace(0, 1)` —
the same `max(..., 1)` bug class already fixed in the solver: when
`total_cost = 0` it silently returned the raw dollar value of
`total_value` mislabeled as a ratio. The dashboard's KPI card computed
`tasks_selected / tasks_total * 100` without protecting against zero
tasks. All three now use the same `if denominator > 0 else 0.0` guard.

**`load_from_api()` logged API keys and mishandled non-JSON responses.**
Enterprise CMMS REST APIs frequently pass authentication as
`?api_key=...` query parameters. The URL was previously logged in full,
writing credentials to log sinks in plaintext; the endpoint is now
sanitised via `urlsplit()` with the query string stripped before logging.
Second: `resp.json()` raised `JSONDecodeError` when a proxy returned an
HTML error page (`401 Unauthorized` intercepted by a corporate proxy is
returned as an HTML login page, not a JSON error object) — `raise_for_status()` alone doesn't catch this because the HTTP status was
`200 OK` from the proxy. The function now catches `JSONDecodeError` and
re-raises as a `ValueError` with an actionable message.

**Three more division-by-zero risks, found by re-running every edge case
that had already been fixed once in solver.py.** The Excel export's
`CapacityUtilization` sheet computed `used / capacity * 100` with no guard
against `capacity = 0`; `ByEquipmentClass`'s ROI column used the same
`.replace(0, 1)` anti-pattern already fixed once in the solver — returning
the raw dollar value of `total_value` mislabeled as a ratio when
`total_cost = 0`; and the dashboard's KPI card divided
`tasks_selected / tasks_total` with no guard against zero tasks. All three
now use the same `if denominator > 0 else 0.0` pattern established
elsewhere. Separately, `build_criticality_matrix()` raised `ValueError:
cannot set a frame with no defined columns` on an empty DataFrame — fixed
with an early-return that produces a correctly-shaped zero-filled 5×5
matrix instead of crashing. A Weibull curve chart was also hardened to
clamp `beta`/`eta` away from zero before calling `weibull_min.sf()`,
since `fit_weibull()` guarantees safe values internally but a real CMMS
import via `load_from_db()`/`load_from_api()` could carry arbitrary
column values that bypass that guarantee.

**`Dim_Asset`'s docstring promised something the code didn't deliver.**
The Excel export's star-schema mirror sheet (built specifically so
Excel-only Power BI users get the same relationship model as the live
database — see `power_bi/README.md` Option C) claims in its own docstring
to "mirror src/db/schema.py" exactly. It didn't: `install_date` is a real
column on the database's `DimAsset` table, populated by the writer, but
was missing entirely from the Excel sheet's column list — a silent,
undocumented divergence between the two supposedly-equivalent Power BI
connection paths. The gap existed because the test fixture building sample
schedules for `test_export.py` never included an `install_date` column
either, so nothing ever exercised the path that would have caught it. Now
fixed, with the column also normalized to a plain string (matching the
database's `String(20)` representation) rather than left as a pandas
`Timestamp`, so a value looks identical regardless of which Power BI
connection path a user chooses.

## 7. Validation Philosophy

Every constraint in §3.3 has a dedicated property-based test in
`tests/test_optimizer.py` that holds across multiple random problem
instances and random seeds — not just one hand-picked example. The
philosophy: a scheduling tool that *usually* respects budget, or *usually*
includes mandatory safety tasks, is more dangerous than no tool at all,
because it will be trusted. The test suite exists to make "usually" into
"always, or it raises loudly."

The suite reaches 98% overall line coverage (369 tests across 11 test
modules), with every single module in `src/` at genuine 100% except the
known-unavoidable gaps documented below:
- `src/scenarios/runner.py` at 54%: `solve_scenario` calls the full ILP
  pipeline; covering it would require a multi-minute integration test that
  regenerates synthetic data, fits Weibull curves, and writes to a database.
  `build_config_for_scenario` — the logic that matters for correctness — is
  at 100%, with a dedicated regression test confirming the global `TA_CFG`
  singleton is never mutated.
- `src/db/writer.py` lines 102, 297–298: two defensive branches that only
  fire under specific concurrent race conditions (see §4.5).
- `src/modeling/weibull.py` lines 66, 68–70: the `fit_weibull` defensive
  branch where scipy's MLE returns non-finite output despite valid inputs
  (documented in prior rounds).

## 8. ERP Integration (SAP PM & IBM Maximo)

### 8.1 The integration problem

Production turnaround planning doesn't start from a CSV file. Work orders
live in SAP PM, IBM Maximo, or a comparable CMMS, and pulling them out
typically means navigating OData endpoints, OSLC paging, vendor-specific
field names, and authentication flows that bear no resemblance to the clean
canonical schema the optimizer actually works with. The gap between "vendor
JSON" and "what the pipeline needs" is where most integration projects lose
weeks to field-mapping spreadsheets.

`src/erp/` collapses that gap into two files with a clear contract: the
connector translates vendor-specific representations into the canonical
schema; the rest of the pipeline never learns which system the data came
from.

### 8.2 Mock server architecture (no ERP required)

`src/erp/mock_api.py` implements a realistic HTTP server with zero
additional dependencies — Python's standard-library `http.server` only —
that serves OData v2 JSON (SAP PM) and OSLC JSON (IBM Maximo) at the same
endpoint paths the real systems use:

```
GET /sap/opu/odata/sap/PM_WORKORDER/MaintenanceOrder  → {"d":{"results":[...]}}
GET /sap/opu/odata/sap/PM_WORKORDER/Equipment         → {"d":{"results":[...]}}
GET /maximo/oslc/os/mxwo                              → {"member":[...],"oslc:totalCount":N}
GET /maximo/oslc/os/mxasset                           → {"member":[...],"oslc:totalCount":N}
```

The server is managed as a context manager (`MockERPServer`) that binds to
an OS-assigned free port (port 0) so multiple test processes can run
concurrently without colliding. Payloads are generated once at module
import time using a seeded RNG, making every CI run deterministic.

The key design choice: the mock intentionally does NOT use `unittest.mock`
or `responses` to intercept `requests.get()` at the library layer. Instead,
it is a real HTTP server, meaning the connector's HTTP-error path, JSON
content-type validation, and connection-refused handling are all tested
against real network conditions rather than against a library patch that
cannot catch "my code is calling the wrong URL."

### 8.3 Field-mapping tables

Both connectors maintain explicit translation tables rather than relying on
string manipulation or format inference:

**SAP PM:**
- `MaintActivityType` codes (`PM01`–`PM07`) → canonical task types
- Priority codes (`1`–`4`) → `Critical / High / Medium / Low`
- Fields distributed across order + equipment master records → joined on `FunctLocId`

**IBM Maximo:**
- `WORKTYPE` codes (`INSPC`, `OVHUL`, …) → same canonical task types
- `WOPRIORITY` integer (`1`–`4`) → same priority vocabulary
- `ESTLABHRS` (total) → split 70/15/10/5 (mech/elec/inst/civil) when per-craft
  hours are absent (a real Maximo deployment detail — per-craft data lives in
  the LABOUR child table, which some API configurations omit)

### 8.4 What changes when connecting a real system

Only the `base_url` (and optionally the auth token). The connector
functions — `load_from_sap_pm()`, `load_from_maximo()`, `load_from_erp()`
— accept the same arguments; the rest of the pipeline is unchanged. The
mock server's test coverage verifies the connector handles the same
response envelopes a real system would return, so a developer can validate
their integration against the mock before ever requesting production API
credentials.

### 8.5 A subtle generator bug caught by the test

The initial Maximo mock generator computed per-craft hour fractions as:
```python
mech_frac = rng.uniform(0.4, 0.7)   # up to 0.7
elec_frac = rng.uniform(0.1, 0.3)   # up to 0.3
inst_frac = rng.uniform(0.05, 0.15) # up to 0.15
civil_frac = max(0.0, 1.0 - mech_frac - elec_frac - inst_frac)
```
The three independent fractions can sum to 1.15 — more than 100% — so the
`max(0.0, ...)` civil clamp fires, civil becomes 0, and the per-craft sum
exceeds `ESTLABHRS`. The test `test_maximo_wo_total_craft_hours_match_sum`
caught this: it asserts that `mech + elec + inst + civil == ESTLABHRS` for
every generated row. Fixed by narrowing the ranges so the worst-case sum is
0.85 and computing civil as `total_hrs - mech - elec - inst` from
already-rounded values, not from the raw fractional remainder.

## 9. Concurrent Scenario Management

### 9.1 The collaboration problem

When a single planner runs the optimizer with three different budgets and
compares the outputs, the existing `dim_run` + multi-run query design
(§4.1) already handles that. The collaboration problem is different: two
planners, simultaneously, each building their own parameter set, each
wanting to see the other's results, each able to accidentally overwrite the
other's work.

`src/scenarios/` adds a named, owned, lockable planning container —
`DimScenario` — that sits one level above `dim_run`. A scenario exists
before it's ever solved (a draft with chosen parameters), gets re-solved
multiple times as planners iterate (each solve appends a new `dim_run` row
tagged with `scenario_id`), and tracks its latest solve in `current_run_id`
so "compare A vs B" always means "compare the most recent result of each."

### 9.2 Two concurrency mechanisms protecting two different things

It's tempting to solve the concurrent-edit problem with either a lock or a
version token, but they protect different failure modes and both are needed:

**Pessimistic / advisory lock** (`status`, `locked_by`, `locked_at`):
Tells a second planner who is currently editing and since when. This is
purely informational — it cannot prevent two processes from both deciding
"the scenario is unlocked" before either acts, because between the check
and the write, another process can acquire the lock.

**Optimistic-concurrency token** (`version`): Closes the TOCTOU gap.
Every write — locking, unlocking, updating parameters — goes through a
single `UPDATE dim_scenario SET ... version = version + 1 WHERE
scenario_id = :id AND version = :expected` statement (`_compare_and_swap_update`).
If no row is updated (because `version` was already incremented by a
concurrent writer), `rowcount == 0`, and `ScenarioConflictError` is raised
before any application state changes. On all four supported backends (SQLite,
Postgres, MySQL, SQL Server) this is atomic at the database layer.

The concurrent-lock race test (`test_only_one_thread_wins_lock`) exercises
this empirically: two threads release a `threading.Barrier` simultaneously
and both call `lock_scenario()`. Exactly one wins; exactly one gets
`ScenarioLockedError` or `ScenarioConflictError`; the database row is
inspected afterward to confirm `version == 2` (not 3, which would indicate
a double write).

### 9.3 Why `current_run_id` is NOT a foreign key

`DimScenario.current_run_id` points at a `dim_run` row, but it is declared
as a plain `Integer` column rather than `ForeignKey("dim_run.run_id")`. The
reason is the creation-order problem: `dim_run.scenario_id` is already a
foreign key pointing at `dim_scenario.scenario_id` (creating the scenario
→ runs relationship). A foreign key in the other direction (scenario
→ current run) creates a cycle: neither table can be created first.

Postgres, MySQL, and SQL Server can resolve this with a deferred constraint
or `ALTER TABLE ... ADD CONSTRAINT` after both tables exist; SQLite's
`ALTER TABLE` cannot add foreign-key constraints to an existing table at
all. Rather than maintaining different schema-creation code for each
backend, the invariant is enforced at the application layer: `current_run_id`
is set in exactly one place — `runner.py::solve_scenario()`, immediately
after `write_results_to_db()` commits — and nowhere else.

### 9.4 `build_config_for_scenario` — the `None`-means-inherit guarantee

The runner's `build_config_for_scenario()` overlays a scenario's parameter
fields onto a copy of the baseline `TurnaroundConfig`. `None` means "inherit
from base_cfg, not zero it out" — a distinction that matters because `0.0`
is a valid (if unusual) budget value for the optimizer to solve with, and
there is no other sentinel available in a plain `Float` column to mean
"not set." The function reads each field explicitly:

```python
if scenario.budget_usd is not None:
    cfg.total_budget = scenario.budget_usd
```

This makes the None case provably correct — a scenario created without a
budget override produces the same result as not using the scenario system at
all. The regression test `test_none_budget_does_not_zero_it_out` enforces
this: it creates a scenario with `budget_usd=None`, calls
`build_config_for_scenario`, and asserts the resulting config's
`total_budget` equals the base config's value, not zero.

The same test also asserts that the global `TA_CFG` singleton is not mutated
by `build_config_for_scenario` — the same stale-default bug class as §5
would have been just as easy to introduce here by doing `TA_CFG.total_budget
= scenario.budget_usd` instead of `copy.copy(base_cfg)` first.

### 9.5 `compare_scenarios` — the diff semantics

`compare_scenarios(engine, sid_a, sid_b)` returns a `ScenarioComparison`
with four non-overlapping work-order sets:

```
added       = selected in B, deferred in A
removed     = selected in A, deferred in B
common_in   = selected in both
common_out  = deferred in both (appearing in either run's fact table)
```

The signed `delta` dict uses "B minus A" throughout, making it natural to
read: scenario B is the "new" scenario being evaluated against scenario A
as the baseline. A negative `delta_budget_usd` means B is a budget cut
relative to A. A positive `delta_tasks_selected` means B fits more work into
its budget. `compare_many_scenarios()` extends this to an arbitrary list of
scenario IDs, returning a summary DataFrame suitable for a Power BI
cross-scenario table or a notebook comparison view.
