"""Alert center."""

from __future__ import annotations

from datetime import timedelta

import pandas as pd
import streamlit as st

from app.components import html, alert_card, insight_callout, kpi_row, missing_file, section_header
from app.data import alerts, apply_filters
from config import STAGE_LABELS


def _when(stamp: pd.Timestamp, now: pd.Timestamp) -> str:
    hours = (now - stamp).total_seconds() / 3600
    if hours < 1:
        return f"{max(1, int(hours * 60))}m ago"
    if hours < 48:
        return f"{int(hours)}h ago"
    return f"{int(hours / 24)}d ago"


def _render_list(frame: pd.DataFrame, now: pd.Timestamp) -> None:
    ordered = frame.sort_values("timestamp", ascending=False).head(30)
    for row in ordered.to_dict(orient="records"):
        alert_card(row, _when(pd.Timestamp(row["timestamp"]), now))


def render(filters: dict) -> None:
    frame = alerts()
    section_header("Alert center", "Confirmed only when probability and a persistent stage spike agree.")
    if frame is None:
        missing_file("data/processed/alerts.csv", "python -m src.detect")
        return
    view = apply_filters(frame, filters)
    if view is None or view.empty:
        html(insight_callout("QUIET", "No alerts match the current filters."))
        return
    html(
        """
        <div class="panel rule">
          <span>ML probability &gt; 0.65</span><strong>AND</strong>
          <span>stage z-score &gt; 2.0</span><strong>FOR</strong>
          <span>≥ 3 consecutive runs</span><strong>→</strong>
          <span>confirmed alert</span>
        </div>
        """
    )
    severities = st.multiselect("Severity", ["high", "medium", "low"], default=["high", "medium", "low"])
    stages = st.multiselect("Stage", list(STAGE_LABELS), default=list(STAGE_LABELS), format_func=lambda key: STAGE_LABELS[key])
    view = view[view["severity_label"].isin(severities) & view["stage"].isin(stages)]
    counts = view["severity_label"].value_counts()
    kpi_row(
        [
            {"label": "High", "value": int(counts.get("high", 0)), "text": str(int(counts.get("high", 0))), "note": "Critical"},
            {"label": "Medium", "value": int(counts.get("medium", 0)), "text": str(int(counts.get("medium", 0))), "note": "Watch"},
            {"label": "Low", "value": int(counts.get("low", 0)), "text": str(int(counts.get("low", 0))), "note": "Confirmed, milder"},
            {"label": "In view", "value": len(view), "text": str(len(view)), "note": "After filters"},
        ]
    )
    st.download_button("Download alerts CSV", view.to_csv(index=False), file_name="pipelineiq_alerts.csv", mime="text/csv")
    live = st.toggle("Simulate live feed", key="live_feed")
    now = view["timestamp"].max()
    if live:
        _live(view, now)
    else:
        _render_list(view, now)


@st.fragment(run_every=timedelta(seconds=1.2))
def _live(view: pd.DataFrame, now: pd.Timestamp) -> None:
    ordered = view.sort_values("timestamp")
    cursor = st.session_state.get("live_k", 1) + 1
    if cursor > len(ordered):
        cursor = 1
    st.session_state["live_k"] = cursor
    _render_list(ordered.head(cursor), now)
