"""
dashboard.py — Generate a standalone, executive-ready HTML dashboard.

Renders eight Plotly charts plus KPI cards into a single self-contained
HTML file.  Designed to mimic a Power BI report page for presentation.

Charts
------
1. KPI Cards        — 6 scalar KPI tiles
2. Budget Waterfall — cumulative cost of selected tasks ranked by ROI
3. Craft Utilisation — horizontal bar gauge per trade
4. Criticality Matrix — 5×5 heatmap with task counts
5. Risk Treemap     — deferred risk by equipment class
6. ROI Scatter      — cost vs net-value bubble per selected task
7. Weibull Curves   — reliability curves for top-5 critical equipment
8. Task Table       — interactive sortable task list
"""

from __future__ import annotations
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from scipy.stats import weibull_min

from src.optimization.solver import SolverResult
from src.modeling.risk import (
    build_criticality_matrix,
    CONSEQUENCE_NAMES,
)
from src.utils.config import DASHBOARD_DIR, TA_CFG
from src.utils.helpers import get_logger, timed

log = get_logger("reporting.dashboard")

# ─── Colour palette ───────────────────────────────────────────────────────────
C = {
    "bg": "#0d1117",
    "card": "#161b22",
    "border": "#30363d",
    "accent": "#f97316",  # orange
    "accent2": "#3b82f6",  # blue
    "success": "#22c55e",  # green
    "danger": "#ef4444",  # red
    "warn": "#eab308",  # yellow
    "text": "#e6edf3",
    "muted": "#8b949e",
    "risk_low": "#22c55e",
    "risk_med": "#eab308",
    "risk_high": "#f97316",
    "risk_crit": "#ef4444",
}

RISK_COLORS = {
    "LOW": C["risk_low"],
    "MEDIUM": C["risk_med"],
    "HIGH": C["risk_high"],
    "CRITICAL": C["risk_crit"],
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor=C["bg"],
    plot_bgcolor=C["card"],
    font=dict(color=C["text"], family="'Segoe UI', Arial, sans-serif"),
    margin=dict(l=40, r=20, t=40, b=40),
)


# ─── Individual chart builders ────────────────────────────────────────────────


def _fig_budget_waterfall(sel: pd.DataFrame, budget: float) -> go.Figure:
    """Cumulative budget consumption ranked by net-value (best ROI first)."""
    df = sel.sort_values("net_value_usd", ascending=False).copy()
    df["cum_cost"] = df["estimated_cost_usd"].cumsum()

    # Top-20 for readability
    top = df.head(30)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=top["wo_id"],
            y=top["estimated_cost_usd"],
            name="Task Cost",
            marker_color=C["accent2"],
            hovertemplate=("<b>%{x}</b><br>" "Cost: $%{y:,.0f}<br>" "<extra></extra>"),
        )
    )
    # Budget line
    fig.add_hline(
        y=budget,
        line_color=C["danger"],
        line_dash="dash",
        annotation_text=f"Budget ${budget/1e6:.1f}M",
        annotation_font_color=C["danger"],
    )

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Top 30 Tasks by Value (Cost per Task)", font_color=C["text"]),
        xaxis=dict(showticklabels=False, gridcolor=C["border"]),
        yaxis=dict(title="Cost (USD)", gridcolor=C["border"]),
    )
    return fig


def _fig_craft_utilisation(summary: dict) -> go.Figure:
    """Horizontal bar gauge showing capacity utilisation per trade."""
    trades = ["Mechanical", "Electrical", "Instrumentation", "Civil"]
    pcts = [
        summary["mech_utilisation"] * 100,
        summary["elec_utilisation"] * 100,
        summary["inst_utilisation"] * 100,
        summary["civil_utilisation"] * 100,
    ]
    used = [
        summary["mech_hours_used"],
        summary["elec_hours_used"],
        summary["inst_hours_used"],
        summary["civil_hours_used"],
    ]
    caps = [
        summary["max_mech_hours"],
        summary["max_elec_hours"],
        summary["max_inst_hours"],
        summary["max_civil_hours"],
    ]

    bar_colors = [C["success"] if p < 80 else C["warn"] if p < 95 else C["danger"] for p in pcts]

    fig = go.Figure()
    # Background bars (capacity)
    fig.add_trace(
        go.Bar(
            y=trades,
            x=[100] * 4,
            orientation="h",
            marker_color=C["border"],
            showlegend=False,
            hoverinfo="skip",
        )
    )
    # Utilisation bars
    fig.add_trace(
        go.Bar(
            y=trades,
            x=pcts,
            orientation="h",
            marker_color=bar_colors,
            text=[f"{u:,.0f} / {c:,.0f} h  ({p:.1f}%)" for u, c, p in zip(used, caps, pcts)],
            textposition="inside",
            textfont=dict(color="white", size=11),
            hovertemplate="<b>%{y}</b><br>Used: %{text}<extra></extra>",
            showlegend=False,
        )
    )

    fig.update_layout(
        **PLOTLY_LAYOUT,
        barmode="overlay",
        title=dict(text="Craft-Hour Capacity Utilisation", font_color=C["text"]),
        xaxis=dict(title="Utilisation (%)", range=[0, 105], gridcolor=C["border"]),
        yaxis=dict(gridcolor=C["border"]),
    )
    return fig


def _fig_criticality_matrix(sched: pd.DataFrame) -> go.Figure:
    """5×5 risk matrix heatmap with task counts."""
    pivot = build_criticality_matrix(sched)

    # Build risk-level color matrix
    RISK_Z = np.array(
        [
            [1, 1, 2, 2, 3],  # likelihood 1
            [1, 2, 2, 3, 3],
            [1, 2, 3, 3, 4],
            [2, 2, 3, 4, 4],
            [2, 3, 4, 4, 4],  # likelihood 5
        ]
    )

    z = pivot.values
    text = [[f"<b>{int(v)}</b>" for v in row] for row in z]

    fig = go.Figure(
        go.Heatmap(
            z=RISK_Z,
            x=[f"C{i}" for i in range(1, 6)],
            y=[f"L{i}" for i in range(1, 6)],
            text=text,
            texttemplate="%{text}",
            colorscale=[
                [0.0, C["risk_low"]],
                [0.33, C["risk_med"]],
                [0.66, C["risk_high"]],
                [1.0, C["risk_crit"]],
            ],
            showscale=False,
            hovertemplate=("Likelihood: %{y}<br>" "Consequence: %{x}<br>" "Tasks: %{text}<extra></extra>"),
            xgap=3,
            ygap=3,
        )
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Criticality Matrix (Likelihood × Consequence)", font_color=C["text"]),
        xaxis=dict(
            title="Consequence →",
            tickvals=list(range(5)),
            ticktext=[f"{i+1}: {CONSEQUENCE_NAMES[i+1]}" for i in range(5)],
            gridcolor=C["border"],
        ),
        yaxis=dict(
            title="↑ Likelihood",
            tickvals=list(range(5)),
            ticktext=[f"{i+1}" for i in range(5)],
            gridcolor=C["border"],
        ),
    )
    return fig


def _fig_risk_treemap(sched: pd.DataFrame) -> go.Figure:
    """Treemap of deferred risk cost by equipment class + risk level."""
    df = (
        sched.groupby(["asset_class", "risk_level"], observed=True)
        .agg(
            deferred_cost=("deferred_cost_usd", "sum"),
            tasks=("wo_id", "count"),
        )
        .reset_index()
    )
    df = df[df["deferred_cost"] > 0]

    fig = go.Figure(
        go.Treemap(
            labels=df["risk_level"] + "<br>" + df["asset_class"],
            parents=df["asset_class"],
            values=df["deferred_cost"],
            customdata=df[["tasks", "asset_class"]],
            hovertemplate=(
                "<b>%{label}</b><br>"
                "Deferred Risk: $%{value:,.0f}<br>"
                "Tasks: %{customdata[0]}<extra></extra>"
            ),
            marker=dict(
                colors=[RISK_COLORS.get(r, C["muted"]) for r in df["risk_level"]],
                line=dict(width=1, color=C["bg"]),
            ),
            textfont=dict(color="white"),
        )
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Deferred Risk Cost by Equipment Class", font_color=C["text"]),
    )
    return fig


def _fig_roi_scatter(sel: pd.DataFrame) -> go.Figure:
    """Cost vs net-value scatter coloured by risk level."""
    df = sel.copy()
    df["bubble_size"] = (df["risk_score"] * 4).clip(8, 40)

    fig = go.Figure()
    for level, grp in df.groupby("risk_level", observed=True):
        fig.add_trace(
            go.Scatter(
                x=grp["estimated_cost_usd"],
                y=grp["net_value_usd"],
                mode="markers",
                name=level,
                marker=dict(
                    color=RISK_COLORS.get(level, C["muted"]),
                    size=grp["bubble_size"],
                    opacity=0.75,
                    line=dict(width=0.5, color=C["bg"]),
                ),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Cost: $%{x:,.0f}<br>"
                    "Net Value: $%{y:,.0f}<br>"
                    "Risk: %{customdata[1]}<extra></extra>"
                ),
                customdata=grp[["wo_id", "risk_level"]].values,
            )
        )

    # Break-even line
    max_val = max(df["estimated_cost_usd"].max(), df["net_value_usd"].max())
    fig.add_trace(
        go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode="lines",
            name="Break-even",
            line=dict(color=C["muted"], dash="dot", width=1),
            showlegend=True,
        )
    )

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Cost vs Net Value — Selected Tasks", font_color=C["text"]),
        xaxis=dict(title="Task Cost (USD)", gridcolor=C["border"], type="log"),
        yaxis=dict(title="Net Value (USD)", gridcolor=C["border"], type="log"),
        legend=dict(bgcolor=C["card"], bordercolor=C["border"]),
    )
    return fig


def _fig_weibull_curves(sched: pd.DataFrame, top_n: int = 5) -> go.Figure:
    """Reliability curves for the top-N highest-risk equipment assets."""
    top = sched.sort_values("risk_score", ascending=False).drop_duplicates("asset_tag").head(top_n)

    t = np.linspace(0, 5000, 500)
    fig = go.Figure()

    for _, row in top.iterrows():
        beta = float(row["fitted_beta"]) if "fitted_beta" in row and not pd.isna(row["fitted_beta"]) else 2.0
        eta = float(row["fitted_eta"]) if "fitted_eta" in row and not pd.isna(row["fitted_eta"]) else 1825.0
        # Clamp to safe Weibull parameter ranges — weibull_min.sf() raises
        # ValueError for beta <= 0 or eta <= 0. The internal fit_weibull()
        # guarantees beta >= 1, but external CMMS data loaded via
        # load_from_db()/load_from_api() could carry arbitrary column values.
        beta = max(beta, 0.01)
        eta = max(eta, 1.0)
        R = weibull_min.sf(t, beta, scale=eta, loc=0) * 100

        fig.add_trace(
            go.Scatter(
                x=t,
                y=R,
                mode="lines",
                name=f"{row['asset_tag']} (β={beta:.2f}, η={eta:.0f}d)",
                hovertemplate=(
                    f"<b>{row['asset_tag']}</b><br>"
                    "Time: %{x:.0f} days<br>"
                    "Reliability: %{y:.1f}%<extra></extra>"
                ),
            )
        )
        # Current age marker
        age = float(row["age_days"])
        R_age = weibull_min.sf(age, beta, scale=eta, loc=0) * 100
        fig.add_trace(
            go.Scatter(
                x=[age],
                y=[R_age],
                mode="markers",
                marker=dict(size=10, symbol="x", color=C["danger"]),
                showlegend=False,
                hovertemplate=(
                    f"<b>{row['asset_tag']}</b><br>"
                    f"Current Age: {age:.0f} d<br>"
                    f"Current Reliability: {R_age:.1f}%<extra></extra>"
                ),
            )
        )

    fig.add_hline(
        y=10,
        line_dash="dash",
        line_color=C["warn"],
        annotation_text="10% Reliability (RUL boundary)",
        annotation_font_color=C["warn"],
    )

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(
            text=f"Weibull Reliability Curves — Top {top_n} Critical Assets",
            font_color=C["text"],
        ),
        xaxis=dict(title="Time (days)", gridcolor=C["border"]),
        yaxis=dict(title="Reliability (%)", range=[0, 105], gridcolor=C["border"]),
        legend=dict(bgcolor=C["card"], bordercolor=C["border"]),
    )
    return fig


def _fig_task_type_donut(sel: pd.DataFrame) -> go.Figure:
    """Donut chart of selected tasks by task type."""
    counts = sel["task_type"].value_counts()
    fig = go.Figure(
        go.Pie(
            labels=counts.index,
            values=counts.values,
            hole=0.55,
            marker=dict(
                colors=[
                    "#f97316",
                    "#3b82f6",
                    "#22c55e",
                    "#a855f7",
                    "#eab308",
                    "#ef4444",
                    "#6366f1",
                ],
                line=dict(color=C["bg"], width=2),
            ),
            textfont=dict(color=C["text"]),
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>",
        )
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Selected Tasks by Type", font_color=C["text"]),
        legend=dict(bgcolor=C["card"], bordercolor=C["border"]),
        annotations=[
            dict(
                text=f"{counts.sum()}<br>Tasks",
                x=0.5,
                y=0.5,
                font=dict(size=16, color=C["text"]),
                showarrow=False,
            )
        ],
    )
    return fig


# ─── Dashboard assembler ──────────────────────────────────────────────────────


def _kpi_html(kpis: dict) -> str:
    tasks_total = kpis["tasks_total"]
    sel_pct = kpis["tasks_selected"] / tasks_total * 100 if tasks_total > 0 else 0.0
    bud_pct = kpis["budget_utilisation"] * 100
    roi = kpis["roi_ratio"]

    def card(title, value, sub="", color=C["accent"]):
        return f"""
        <div class="kpi-card">
          <div class="kpi-title">{title}</div>
          <div class="kpi-value" style="color:{color}">{value}</div>
          <div class="kpi-sub">{sub}</div>
        </div>"""

    return "".join(
        [
            card(
                "Tasks Selected",
                f"{kpis['tasks_selected']:,} / {kpis['tasks_total']:,}",
                f"{sel_pct:.1f}% of work orders",
                C["accent2"],
            ),
            card(
                "Budget Utilised",
                f"${kpis['budget_used_usd']/1e6:.2f}M",
                f"{bud_pct:.1f}% of ${kpis['budget_usd']/1e6:.1f}M",
                C["accent"] if bud_pct < 90 else C["danger"],
            ),
            card("ROI Ratio", f"{roi:.1f}×", "Net Value / Budget Used", C["success"]),
            card(
                "Net Risk Value",
                f"${kpis['total_net_value_usd']/1e6:.2f}M",
                "Risk-adjusted net value",
                C["success"],
            ),
            card(
                "Risk Score Reduced",
                f"{kpis['total_risk_score_reduced']:,}",
                "Criticality-matrix units",
                C["warn"],
            ),
            card(
                "Solver Status",
                kpis["solver_status"],
                f"Solved in {kpis['solve_time_s']:.1f}s · OR-Tools CP-SAT",
                C["success"] if kpis["solver_status"] == "OPTIMAL" else C["warn"],
            ),
        ]
    )


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Turnaround Optimizer — Executive Dashboard</title>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body   {{ background: {bg}; color: {text}; font-family: 'Segoe UI', Arial, sans-serif; padding: 0; }}
    header {{ background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
              padding: 28px 40px; border-bottom: 2px solid {accent}; }}
    header h1 {{ font-size: 1.8rem; font-weight: 700; color: {accent}; letter-spacing: 1px; }}
    header p  {{ color: {muted}; font-size: 0.85rem; margin-top: 4px; }}
    header .meta {{ float: right; text-align: right; color: {muted}; font-size: 0.8rem; }}
    .container {{ padding: 28px 40px; }}
    /* KPI Cards */
    .kpi-grid {{ display: flex; gap: 16px; margin-bottom: 28px; flex-wrap: wrap; }}
    .kpi-card {{ flex: 1; min-width: 160px; background: {card}; border: 1px solid {border};
                 border-radius: 8px; padding: 20px 22px; }}
    .kpi-title {{ font-size: 0.72rem; text-transform: uppercase; letter-spacing: 1.2px;
                  color: {muted}; margin-bottom: 8px; }}
    .kpi-value {{ font-size: 1.9rem; font-weight: 700; line-height: 1; }}
    .kpi-sub   {{ font-size: 0.75rem; color: {muted}; margin-top: 6px; }}
    /* Chart grid */
    .chart-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }}
    .chart-grid.thirds {{ grid-template-columns: 1fr 1fr 1fr; }}
    .chart-card {{ background: {card}; border: 1px solid {border}; border-radius: 8px;
                   padding: 4px; overflow: hidden; }}
    .chart-card.full-width {{ grid-column: 1 / -1; }}
    /* Table */
    .table-wrap {{ background: {card}; border: 1px solid {border}; border-radius: 8px;
                   overflow: auto; max-height: 480px; margin-bottom: 28px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.8rem; }}
    th {{ position: sticky; top: 0; background: #1c2433; padding: 10px 14px;
          text-align: left; border-bottom: 1px solid {border}; color: {accent};
          cursor: pointer; user-select: none; white-space: nowrap; }}
    th:hover {{ background: #243040; }}
    td {{ padding: 8px 14px; border-bottom: 1px solid {border}; white-space: nowrap; }}
    tr:hover td {{ background: rgba(249,115,22,0.06); }}
    .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px;
              font-size: 0.7rem; font-weight: 600; }}
    .badge-critical {{ background: rgba(239,68,68,0.2);   color: #ef4444; }}
    .badge-high     {{ background: rgba(249,115,22,0.2);  color: #f97316; }}
    .badge-medium   {{ background: rgba(234,179,8,0.2);   color: #eab308; }}
    .badge-low      {{ background: rgba(34,197,94,0.2);   color: #22c55e; }}
    .badge-include  {{ background: rgba(59,130,246,0.2);  color: #3b82f6; }}
    .badge-defer    {{ background: rgba(139,148,158,0.2); color: #8b949e; }}
    footer {{ padding: 20px 40px; border-top: 1px solid {border}; color: {muted};
              font-size: 0.75rem; text-align: center; }}
    .sort-icon {{ margin-left: 4px; opacity: 0.5; }}
  </style>
</head>
<body>
<header>
  <div class="meta">
    Turnaround Date: {ta_date}<br>
    OR-Tools CP-SAT &nbsp;|&nbsp; Python 3.11<br>
    Generated: {gen_date}
  </div>
  <h1>🏭  PLANT SHUTDOWN TURNAROUND OPTIMIZER</h1>
  <p>Knapsack ILP · Weibull Reliability Analysis · Risk-Adjusted Scheduling &nbsp;|&nbsp;
     Budget: ${budget_m:.1f}M &nbsp;|&nbsp; {n_tasks} Work Orders</p>
</header>

<div class="container">

  <!-- KPI Cards -->
  <div class="kpi-grid">{kpi_html}</div>

  <!-- Row 1: Budget breakdown + Craft utilisation -->
  <div class="chart-grid">
    <div class="chart-card">{chart_waterfall}</div>
    <div class="chart-card">{chart_craft}</div>
  </div>

  <!-- Row 2: Criticality matrix + Treemap -->
  <div class="chart-grid">
    <div class="chart-card">{chart_matrix}</div>
    <div class="chart-card">{chart_treemap}</div>
  </div>

  <!-- Row 3: ROI scatter + Donut + Weibull -->
  <div class="chart-grid thirds">
    <div class="chart-card">{chart_scatter}</div>
    <div class="chart-card">{chart_donut}</div>
    <div class="chart-card">{chart_weibull}</div>
  </div>

  <!-- Work Order Table -->
  <h2 style="margin: 24px 0 12px; font-size:1rem; color:{accent}; letter-spacing:1px;">
    📋 OPTIMIZED WORK ORDER SCHEDULE</h2>
  <div class="table-wrap">
    <table id="wo-table">
      <thead>
        <tr>
          <th onclick="sortTable(0)">WO ID<span class="sort-icon">↕</span></th>
          <th onclick="sortTable(1)">Asset<span class="sort-icon">↕</span></th>
          <th onclick="sortTable(2)">Area<span class="sort-icon">↕</span></th>
          <th onclick="sortTable(3)">Type<span class="sort-icon">↕</span></th>
          <th onclick="sortTable(4)">Priority<span class="sort-icon">↕</span></th>
          <th onclick="sortTable(5)">Decision<span class="sort-icon">↕</span></th>
          <th onclick="sortTable(6)">Cost ($)<span class="sort-icon">↕</span></th>
          <th onclick="sortTable(7)">Net Value ($)<span class="sort-icon">↕</span></th>
          <th onclick="sortTable(8)">P(Fail)<span class="sort-icon">↕</span></th>
          <th onclick="sortTable(9)">Risk Score<span class="sort-icon">↕</span></th>
          <th onclick="sortTable(10)">Risk Level<span class="sort-icon">↕</span></th>
        </tr>
      </thead>
      <tbody>{table_rows}</tbody>
    </table>
  </div>

</div><!-- /container -->

<footer>Turnaround Knapsack Optimizer &nbsp;·&nbsp; OR-Tools CP-SAT ILP &nbsp;·&nbsp;
Weibull Reliability Analysis &nbsp;·&nbsp; Anthropic Claude — For demonstration purposes</footer>

<script>
let sortAsc = {{}};
function sortTable(col) {{
  const tbl = document.getElementById('wo-table');
  const rows = Array.from(tbl.querySelectorAll('tbody tr'));
  sortAsc[col] = !sortAsc[col];
  rows.sort((a, b) => {{
    const av = a.cells[col].dataset.val ?? a.cells[col].textContent.replace(/[$,]/g,'');
    const bv = b.cells[col].dataset.val ?? b.cells[col].textContent.replace(/[$,]/g,'');
    const an = parseFloat(av), bn = parseFloat(bv);
    const cmp = isNaN(an) ? av.localeCompare(bv) : an - bn;
    return sortAsc[col] ? cmp : -cmp;
  }});
  const tbody = tbl.querySelector('tbody');
  rows.forEach(r => tbody.appendChild(r));
}}
</script>
</body>
</html>"""


def _build_table_rows(sched: pd.DataFrame, max_rows: int = 250) -> str:
    """
    Render HTML table rows for the top `max_rows` tasks.

    Every user-controlled string value is passed through html.escape()
    before being interpolated into the HTML template. Without this, a real
    CMMS work-order description or area field containing '<script>' or
    similar would execute as JavaScript in the browser — a real risk when
    loading data from a production system rather than the synthetic
    generator. Numeric values interpolated into data-val attributes are
    already safe because they're formatted as float/int strings by Python's
    format spec, which never produces angle brackets or quotes.
    """
    import html

    rows_html = []
    for _, r in sched.head(max_rows).iterrows():
        rl = html.escape(str(r.get("risk_level", "LOW")).upper())
        dec = html.escape(str(r.get("decision", "DEFER")).upper())
        pri = html.escape(str(r.get("priority", "Low")).title())

        rl_cls = f"badge-{rl.lower()}"
        dec_cls = "badge-include" if dec == "INCLUDE" else "badge-defer"
        pri_cls = f"badge-{pri.lower()}"

        fp = r.get("failure_prob", 0.0)
        rs = r.get("risk_score", 0)
        cost = r.get("estimated_cost_usd", 0)
        val = r.get("net_value_usd", 0)

        rows_html.append(f"""
        <tr>
          <td>{html.escape(str(r.wo_id))}</td>
          <td>{html.escape(str(r.asset_tag))}</td>
          <td>{html.escape(str(r.area))}</td>
          <td>{html.escape(str(r.task_type))}</td>
          <td><span class="badge {pri_cls}">{pri}</span></td>
          <td><span class="badge {dec_cls}">{dec}</span></td>
          <td data-val="{cost:.0f}">${cost:>12,.0f}</td>
          <td data-val="{val:.0f}">${val:>12,.0f}</td>
          <td data-val="{fp:.4f}">{fp:.3f}</td>
          <td data-val="{rs}">{rs}</td>
          <td><span class="badge {rl_cls}">{rl}</span></td>
        </tr>""")
    return "\n".join(rows_html)


def _to_div(fig: go.Figure, height: int = 340) -> str:
    fig.update_layout(height=height)
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True})


# ─── Main entry point ─────────────────────────────────────────────────────────


@timed
def generate_dashboard(
    result: SolverResult,
    out_path: str | Path | None = None,
) -> Path:
    """
    Render the full executive dashboard to a standalone HTML file.

    `out_path` accepts either a `str` or `Path` and is coerced to `Path`
    internally — see export_to_excel()'s docstring in src/reporting/export.py
    for why a bare string used to crash here too (`.parent.mkdir(...)` and
    `.write_text(...)` both require a true Path object). Defaults to `None`,
    resolved against `DASHBOARD_DIR` inside the function body rather than
    baked into the signature at import time.
    """
    from datetime import datetime

    if out_path is None:
        out_path = DASHBOARD_DIR / "turnaround_dashboard.html"
    out_path = Path(out_path)

    sched = result.schedule
    sel = result.selected_schedule
    kpis = result.summary

    log.info("Generating executive dashboard (%d tasks) …", len(sched))

    charts = {
        "chart_waterfall": _to_div(_fig_budget_waterfall(sel, kpis["budget_usd"])),
        "chart_craft": _to_div(_fig_craft_utilisation(kpis)),
        "chart_matrix": _to_div(_fig_criticality_matrix(sched)),
        "chart_treemap": _to_div(_fig_risk_treemap(sched)),
        "chart_scatter": _to_div(_fig_roi_scatter(sel)),
        "chart_donut": _to_div(_fig_task_type_donut(sel)),
        "chart_weibull": _to_div(_fig_weibull_curves(sched)),
    }

    html = HTML_TEMPLATE.format(
        **{k: v for k, v in C.items()},
        ta_date=TA_CFG.turnaround_date,
        gen_date=datetime.now().strftime("%Y-%m-%d %H:%M"),
        budget_m=TA_CFG.total_budget / 1e6,
        n_tasks=len(sched),
        kpi_html=_kpi_html(kpis),
        table_rows=_build_table_rows(sched.sort_values("decision")),
        **charts,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    log.info("✅  Dashboard saved → %s  (%d KB)", out_path, len(html) // 1024)
    return out_path
