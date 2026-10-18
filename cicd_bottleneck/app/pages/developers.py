"""Developer behavior analytics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from app import charts
from app.components import html, insight_callout, missing_file, section_header, styled_table
from app.data import apply_filters, predictions

_METRICS = {
    "Commit frequency": "commit_frequency",
    "Code churn": "code_churn",
    "PR review latency": "pr_review_latency_hours",
    "Risk score": "bottleneck_probability",
}


def render(filters: dict) -> None:
    frame = predictions()
    if frame is None:
        missing_file("data/processed/predictions.csv", "python -m src.train")
        return
    view = apply_filters(frame, filters)
    section_header("Developer behavior", "The paper's claim: risk shows up in how people commit and review, before the pipeline finishes.")
    if view is None or view.empty:
        html(insight_callout("NO MATCHES", "No developers match the current filters."))
        return
    choice = st.radio("Heatmap metric", list(_METRICS), horizontal=True, label_visibility="collapsed")
    heat = _top_developers(view, 25)
    charts.show(charts.heatmap(heat, _METRICS[choice], f"{choice} by developer and week"))
    board = _leaderboard(view)
    section_header("Risk leaderboard", "Developers with at least 8 runs, ranked by mean predicted risk.")
    if board.empty:
        html(insight_callout("THIN SAMPLE", "Not enough runs per developer in this filter."))
        return
    left, right = st.columns((3, 2), gap="medium")
    with left:
        styled_table(
            ["Developer", "Risk", "Avg churn", "Avg review h", "Historical success"],
            [
                [row.developer_id, f"{row.risk:.0%}", f"{row.churn:,.0f}", f"{row.latency:.1f}", f"{row.history:.0%}"]
                for row in board.itertuples(index=False)
            ],
        )
    selected = st.selectbox("Drill into a developer", board["developer_id"].tolist())
    person = view[view["developer_id"] == selected]
    team = view
    with right:
        charts.show(charts.radar(["Risk", "Churn", "Review wait", "Commit rate", "Build fragility"], _radar_values(person, team), f"{selected} versus this view"))
    html(insight_callout("COACHING", _coaching(person, team)))
    zone_rate, base_rate = _zone_rates(view)
    title = "Code churn against review latency"
    if base_rate > 0 and zone_rate is not None:
        title = f"High churn plus slow reviews: {zone_rate / base_rate:.1f}× the bottleneck rate"
    charts.show(charts.scatter_interaction(view, title))


def _top_developers(view: pd.DataFrame, limit: int) -> pd.DataFrame:
    ranking = view.groupby("developer_id")["bottleneck_probability"].mean().sort_values(ascending=False)
    if len(ranking) <= limit:
        return view
    keep = set(ranking.head(limit).index)
    return view[view["developer_id"].isin(keep)]


def _leaderboard(view: pd.DataFrame) -> pd.DataFrame:
    grouped = view.groupby("developer_id").agg(
        runs=("run_id", "size"),
        risk=("bottleneck_probability", "mean"),
        churn=("code_churn", "mean"),
        latency=("pr_review_latency_hours", "mean"),
        history=("historical_build_success_ratio", "mean"),
    )
    board = grouped[grouped["runs"] >= 8].sort_values("risk", ascending=False).head(10).reset_index()
    return board


def _scale(value: float, series: pd.Series) -> float:
    lo, hi = float(series.min()), float(series.max())
    if hi - lo < 1e-9:
        return 0.5
    return float(np.clip((value - lo) / (hi - lo), 0, 1))


def _radar_values(person: pd.DataFrame, team: pd.DataFrame) -> list[float]:
    return [
        _scale(person["bottleneck_probability"].mean(), team.groupby("developer_id")["bottleneck_probability"].mean()),
        _scale(person["code_churn"].mean(), team.groupby("developer_id")["code_churn"].mean()),
        _scale(person["pr_review_latency_hours"].mean(), team.groupby("developer_id")["pr_review_latency_hours"].mean()),
        _scale(person["commit_frequency"].mean(), team.groupby("developer_id")["commit_frequency"].mean()),
        _scale(1 - person["historical_build_success_ratio"].mean(), 1 - team.groupby("developer_id")["historical_build_success_ratio"].mean()),
    ]


def _coaching(person: pd.DataFrame, team: pd.DataFrame) -> str:
    def z(column: str, higher_is_risk: bool = True) -> float:
        std = float(team[column].std() or 1)
        delta = (float(person[column].mean()) - float(team[column].mean())) / std
        return delta if higher_is_risk else -delta

    drivers = [
        ("code churn", z("code_churn")),
        ("PR review latency", z("pr_review_latency_hours")),
        ("commit frequency", z("commit_frequency")),
        ("weak historical build success", z("historical_build_success_ratio", higher_is_risk=False)),
    ]
    drivers.sort(key=lambda item: item[1], reverse=True)
    risky = person[person["bottleneck_probability"] > 0.65]
    share = len(risky) / len(person) if len(person) else 0
    first, second = drivers[0], drivers[1]
    return (
        f"{person['developer_id'].iloc[0]} sits {first[1]:+.1f}σ on {first[0]} and {second[1]:+.1f}σ on {second[0]} "
        f"versus this filtered team. {share:.0%} of their runs are in the red risk band. "
        f"Start with the larger of those two gaps."
    )


def _zone_rates(view: pd.DataFrame) -> tuple[float | None, float]:
    zone = (view["code_churn"] >= view["code_churn"].quantile(0.75)) & (
        view["pr_review_latency_hours"] >= view["pr_review_latency_hours"].quantile(0.75)
    )
    if zone.sum() < 10 or (~zone).sum() < 10:
        return None, float(view["true_label"].mean())
    return float(view.loc[zone, "true_label"].mean()), float(view.loc[~zone, "true_label"].mean())
