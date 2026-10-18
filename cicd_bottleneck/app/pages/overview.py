"""Executive overview."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app import charts
from app.components import html, insight_callout, kpi_row, missing_file, section_header, styled_table
from app.data import ablation, alerts, apply_filters, hours_at_risk, importance, metrics, predictions, project_medians


def _spark(values: list[float], color: str) -> str:
    if len(values) < 2:
        return ""
    lo, hi = min(values), max(values)
    span = hi - lo or 1
    width, height = 96, 28
    points = []
    for index, value in enumerate(values):
        x = index / (len(values) - 1) * width
        y = height - ((value - lo) / span) * (height - 4) - 2
        points.append(f"{x:.1f},{y:.1f}")
    return f'<svg width="{width}" height="{height}" aria-hidden="true"><polyline fill="none" stroke="{color}" stroke-width="2" points="{" ".join(points)}"/></svg>'


def _tone(rate: float) -> tuple[str, str]:
    if rate >= 0.45:
        return "#EF4444", "HIGH"
    if rate >= 0.35:
        return "#F59E0B", "MEDIUM"
    return "#22C55E", "LOW"


def render(filters: dict) -> None:
    frame = predictions()
    metric = metrics()
    if frame is None:
        missing_file("data/processed/predictions.csv", "python -m src.train")
        return
    view = apply_filters(frame, filters)
    section_header("Executive overview", "Where bottleneck risk is concentrating in the current filter.")
    if view is None or view.empty:
        html(insight_callout("NO MATCHES", "No runs match the current filters."))
        return
    alert_frame = apply_filters(alerts(), filters)
    alert_count = 0 if alert_frame is None else len(alert_frame)
    f1 = metric["best_f1"] * 100 if metric else 0
    model_name = metric["best_model_display"] if metric else "Model pending"
    medians = project_medians(frame)
    hours = hours_at_risk(view, medians)
    rate = float(view["true_label"].mean())
    kpi_row(
        [
            {"label": "Runs analysed", "value": len(view), "decimals": 0, "text": f"{len(view):,}", "note": "In the current filter"},
            {"label": "Bottleneck rate", "value": round(rate * 100, 1), "decimals": 1, "suffix": "%", "text": f"{rate:.1%}", "note": "True label, this view"},
            {"label": "Best-model F1", "value": round(f1, 1), "decimals": 1, "suffix": "%", "text": f"{f1:.1f}%", "note": f"{model_name} · held-out test"},
            {"label": "Active alerts", "value": alert_count, "decimals": 0, "text": f"{alert_count:,}", "note": "Confirmed by both rules"},
            {"label": "Hours at risk", "value": round(hours, 1), "decimals": 1, "text": f"{hours:.1f}", "note": "Excess over project median"},
        ]
    )
    notes = _stories(view, importance(), metric)
    kickers = ["SIGNAL", "HOTSPOT", "INTERACTION"]
    html('<div class="story">' + "".join(insight_callout(kickers[i], text) for i, text in enumerate(notes)) + "</div>")
    left, right = st.columns((3, 2), gap="medium")
    with left:
        charts.show(charts.weekly_bottleneck(view, _weekly_title(view)))
    with right:
        charts.show(charts.risk_donut(view, "Risk bands in this view"))
    section_header("Project league", "Bottleneck rate with a weekly sparkline. Colour is paired with a text label.")
    _league(view)


def _weekly_title(view: pd.DataFrame) -> str:
    weekly = view.set_index("timestamp")["true_label"].resample("W-MON").mean().dropna()
    if len(weekly) < 2:
        return "Weekly bottleneck rate"
    delta = weekly.iloc[-1] - weekly.iloc[0]
    direction = "rose" if delta > 0.01 else "eased" if delta < -0.01 else "held steady"
    return f"Weekly bottleneck rate {direction} by {abs(delta):.0%} across this view"


def _stories(view: pd.DataFrame, table: pd.DataFrame | None, metric: dict | None) -> list[str]:
    notes = []
    if table is not None and len(table):
        behavioral = table[table["group"] == "behavioral"].sort_values("mdi_pct", ascending=False)
        share = float(table.loc[table["group"] == "behavioral", "mdi_pct"].sum())
        if len(behavioral) >= 2:
            notes.append(f"{behavioral.iloc[0]['label']} and {behavioral.iloc[1]['label']} lead the behavior signal. Together, behavioral features hold {share:.0f}% of Random Forest MDI.")
    latest = view["timestamp"].max()
    window = view[view["timestamp"] >= latest - pd.Timedelta(days=28)]
    if len(window):
        rates = window.groupby("project_name")["true_label"].mean().sort_values(ascending=False)
        notes.append(f"{rates.index[0]} has the highest bottleneck rate in the last 28 days of this view ({rates.iloc[0]:.0%}).")
    churn_cut = float(view["code_churn"].quantile(0.75))
    latency_cut = float(view["pr_review_latency_hours"].quantile(0.75))
    zone = (view["code_churn"] >= churn_cut) & (view["pr_review_latency_hours"] >= latency_cut)
    if zone.sum() > 15 and (~zone).sum() > 15 and view.loc[~zone, "true_label"].mean() > 0:
        ratio = view.loc[zone, "true_label"].mean() / view.loc[~zone, "true_label"].mean()
        notes.append(f"Runs above the 75th percentile of both churn and review latency are bottlenecks {ratio:.1f}× as often as the rest of this view.")
    if metric and len(notes) < 3:
        notes.append(f"Held-out F1 for {metric['best_model_display']} is {metric['best_f1']:.1%}. Rank models on F1, not accuracy.")
    while len(notes) < 3:
        notes.append("Adjust the global filters to focus the story on one team, platform, or risk band.")
    return notes[:3]


def _league(view: pd.DataFrame) -> None:
    rows = []
    for project_id, part in view.groupby("project_id"):
        name = part["project_name"].iloc[0]
        weekly = part.set_index("timestamp")["true_label"].resample("W-MON").mean().dropna()
        rate = float(part["true_label"].mean())
        color, label = _tone(rate)
        rows.append(
            (
                rate,
                [
                    f"{name}<br><span class='muted'>{project_id}</span>",
                    f"{len(part):,}",
                    f"<span style='color:{color}'>{label}</span> {rate:.0%}",
                    f"{part['bottleneck_probability'].mean():.0%}",
                    _spark(weekly.tolist(), color),
                ],
            )
        )
    rows = [row for _rate, row in sorted(rows, key=lambda item: item[0], reverse=True)]
    styled_table(["Project", "Runs", "Bottleneck rate", "Mean risk", "Trend"], rows)
