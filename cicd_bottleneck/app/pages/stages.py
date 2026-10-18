"""Pipeline stage performance."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app import charts
from app.components import html, insight_callout, missing_file, section_header, styled_table
from app.data import alerts, apply_filters, predictions
from config import STAGE_LABELS


def render(filters: dict) -> None:
    frame = predictions()
    if frame is None:
        missing_file("data/processed/predictions.csv", "python -m src.train")
        return
    view = apply_filters(frame, filters)
    section_header("Stage performance", "A stage is marked only when it stays slow for three runs and the model agrees.")
    if view is None or view.empty:
        html(insight_callout("NO MATCHES", "No runs match the current filters."))
        return
    projects = sorted(view["project_name"].unique())
    project_name = st.selectbox("Project", projects)
    stage_label = st.radio("Stage", list(STAGE_LABELS.values()), horizontal=True)
    stage = {label: key for key, label in STAGE_LABELS.items()}[stage_label]
    project = view[view["project_name"] == project_name].sort_values("timestamp")
    alert_frame = apply_filters(alerts(), filters)
    if alert_frame is not None and len(alert_frame):
        alert_frame = alert_frame[alert_frame["project_name"] == project_name]
    slow = project[stage]
    roll = slow.shift(1).rolling(20, min_periods=10).mean()
    latest = float(slow.iloc[-1]) if len(slow) else 0.0
    baseline = float(roll.dropna().iloc[-1]) if roll.notna().any() else 0.0
    if baseline > 0:
        gap = latest / baseline - 1
        title = f"{stage_label} is {gap:+.0%} versus its 20-run mean"
    else:
        title = f"{stage_label} duration for {project_name}"
    charts.show(charts.stage_timeseries(project, alert_frame, stage, title))
    charts.show(charts.small_multiples(project, f"All four stages · {project_name}"))
    left, right = st.columns(2, gap="medium")
    with left:
        charts.show(charts.waterfall(view, "Average time split by stage in this filter"))
    with right:
        charts.show(charts.violin_platform(view, "Total runtime by CI platform"))
    section_header("Latest anomaly windows", "Confirmed alerts: z > 2 for at least three consecutive runs.")
    if alert_frame is None or alert_frame.empty:
        html(insight_callout("QUIET", "No confirmed alerts in this filter. Generate them with python -m src.detect if the file is missing."))
        return
    latest_alerts = alert_frame.sort_values("timestamp", ascending=False).head(12)
    styled_table(
        ["When", "Project", "Stage", "z-score", "Window", "Probability", "Severity"],
        [
            [
                pd.Timestamp(row.timestamp).strftime("%d %b %H:%M"),
                row.project_name,
                STAGE_LABELS.get(row.stage, row.stage),
                f"{row.z_score:.2f}",
                str(int(row.anomaly_window_runs)),
                f"{row.ml_probability:.0%}",
                str(row.severity_label).upper(),
            ]
            for row in latest_alerts.itertuples(index=False)
        ],
    )
