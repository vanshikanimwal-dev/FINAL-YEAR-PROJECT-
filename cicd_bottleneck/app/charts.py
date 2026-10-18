"""Plotly figures. Every chart uses the shared dark theme."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app.components import plotly_theme
from config import FEATURE_LABELS, MODEL_DISPLAY, STAGE_LABELS

_RISK_COLORS = {"green": "#22C55E", "amber": "#F59E0B", "red": "#EF4444"}


def show(fig: go.Figure) -> None:
    """Draw a figure with our theme, not Streamlit's default Plotly skin."""
    import streamlit as st

    st.plotly_chart(
        fig,
        width="stretch",
        theme=None,
        config={"displayModeBar": False, "responsive": True},
    )


def apply_theme(fig: go.Figure, title: str | None = None, height: int = 380) -> go.Figure:
    fig.update_layout(**plotly_theme(), height=height)
    fig.update_layout(title={"text": title or "", "font": {"size": 16, "color": "#E5E7EB"}, "x": 0})
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.14)", zerolinecolor="rgba(148,163,184,0.14)", tickfont={"color": "#94A3B8"})
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.14)", zerolinecolor="rgba(148,163,184,0.14)", tickfont={"color": "#94A3B8"})
    return fig


def weekly_bottleneck(frame: pd.DataFrame, title: str) -> go.Figure:
    tmp = frame.sort_values("timestamp").set_index("timestamp")
    weekly = tmp["true_label"].resample("W-MON").mean().dropna()
    smooth = weekly.rolling(4, min_periods=1).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=weekly.index, y=weekly, fill="tozeroy", name="Weekly rate", line={"color": "#2DD4BF", "width": 2}, hovertemplate="%{x|%d %b %Y}<br>Bottleneck rate %{y:.1%}<extra></extra>"))
    fig.add_trace(go.Scatter(x=smooth.index, y=smooth, name="4-week average", line={"color": "#6366F1", "width": 2, "dash": "dot"}, hovertemplate="4-week average %{y:.1%}<extra></extra>"))
    fig.update_yaxes(tickformat=".0%")
    return apply_theme(fig, title, 360)


def risk_donut(frame: pd.DataFrame, title: str) -> go.Figure:
    counts = frame["risk_band"].value_counts()
    labels = [band.title() for band in counts.index]
    colors = [_RISK_COLORS.get(str(band), "#94A3B8") for band in counts.index]
    fig = go.Figure(go.Pie(labels=labels, values=counts.values, hole=0.62, marker={"colors": colors}, hovertemplate="%{label}<br>%{value} runs (%{percent})<extra></extra>"))
    fig.update_layout(showlegend=True)
    return apply_theme(fig, title, 360)


def heatmap(frame: pd.DataFrame, metric: str, title: str) -> go.Figure:
    tmp = frame.copy()
    stamp = tmp["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)
    tmp["week"] = stamp.dt.to_period("W-MON").dt.start_time
    pivot = tmp.pivot_table(index="developer_id", columns="week", values=metric, aggfunc="mean")
    fig = go.Figure(go.Heatmap(z=pivot.values, x=[pd.Timestamp(col).strftime("%d %b") for col in pivot.columns], y=list(pivot.index), colorscale=[[0, "#121A33"], [0.45, "#0F766E"], [1, "#2DD4BF"]], hovertemplate="Developer %{y}<br>%{x}<br>%{z:.2f}<extra></extra>", colorbar={"tickfont": {"color": "#94A3B8"}}))
    height = max(360, 22 * len(pivot.index) + 120)
    return apply_theme(fig, title, height)


def scatter_interaction(frame: pd.DataFrame, title: str) -> go.Figure:
    sample = frame.sample(min(len(frame), 1200), random_state=42) if len(frame) else frame
    size = np.clip(sample["total_execution_time_sec"] / max(sample["total_execution_time_sec"].median(), 1) * 10, 6, 26)
    fig = go.Figure(go.Scatter(x=sample["code_churn"], y=sample["pr_review_latency_hours"], mode="markers", marker={"size": size, "color": sample["bottleneck_probability"], "colorscale": [[0, "#22C55E"], [0.45, "#F59E0B"], [1, "#EF4444"]], "showscale": True, "colorbar": {"title": "Risk", "tickfont": {"color": "#94A3B8"}}, "line": {"width": 0}}, hovertemplate="Churn %{x:.0f}<br>Review %{y:.1f} h<br>Risk %{marker.color:.0%}<extra></extra>"))
    if len(sample):
        fig.add_vline(x=float(frame["code_churn"].quantile(0.75)), line_dash="dot", line_color="#6366F1")
        fig.add_hline(y=float(frame["pr_review_latency_hours"].quantile(0.75)), line_dash="dot", line_color="#6366F1")
    fig.update_xaxes(title="Code churn (lines)")
    fig.update_yaxes(title="PR review latency (hours)")
    return apply_theme(fig, title, 460)


def radar(labels: list[str], values: list[float], title: str) -> go.Figure:
    fig = go.Figure(go.Scatterpolar(r=values + values[:1], theta=labels + labels[:1], fill="toself", line={"color": "#2DD4BF"}, fillcolor="rgba(45,212,191,0.25)"))
    fig.update_layout(polar={"bgcolor": "rgba(0,0,0,0)", "radialaxis": {"range": [0, 1], "gridcolor": "#1F2A4D", "tickfont": {"color": "#94A3B8"}}, "angularaxis": {"gridcolor": "#1F2A4D", "tickfont": {"color": "#E5E7EB"}}})
    return apply_theme(fig, title, 380)


def stage_timeseries(frame: pd.DataFrame, alerts: pd.DataFrame | None, stage: str, title: str) -> go.Figure:
    df = frame.sort_values("timestamp")
    series = df[stage]
    rolling = series.shift(1).rolling(20, min_periods=10)
    mean, std = rolling.mean(), rolling.std()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["timestamp"], y=mean + 2 * std, line={"width": 0}, showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=df["timestamp"], y=mean - 2 * std, fill="tonexty", fillcolor="rgba(45,212,191,0.12)", line={"width": 0}, name="±2σ band", hovertemplate="Band %{y:.0f}s<extra></extra>"))
    fig.add_trace(go.Scatter(x=df["timestamp"], y=mean, name="Rolling mean (20)", line={"color": "#6366F1", "width": 2}, hovertemplate="Mean %{y:.0f}s<extra></extra>"))
    fig.add_trace(go.Scatter(x=df["timestamp"], y=series, name="Duration", line={"color": "#2DD4BF", "width": 1.4}, hovertemplate="%{x|%d %b %H:%M}<br>%{y:.0f}s<extra></extra>"))
    if alerts is not None and len(alerts):
        marks = df.merge(alerts.loc[alerts["stage"] == stage, ["run_id", "z_score", "ml_probability"]], on="run_id", how="inner")
        if len(marks):
            fig.add_trace(go.Scatter(x=marks["timestamp"], y=marks[stage], mode="markers", name="Confirmed alert", marker={"color": "#EF4444", "size": 9, "symbol": "diamond"}, customdata=np.column_stack([marks["z_score"], marks["ml_probability"]]), hovertemplate="Alert<br>z=%{customdata[0]:.2f}<br>probability %{customdata[1]:.0%}<extra></extra>"))
    fig.update_yaxes(title="Seconds")
    return apply_theme(fig, title, 420)


def small_multiples(frame: pd.DataFrame, title: str) -> go.Figure:
    stages = list(STAGE_LABELS)
    fig = make_subplots(rows=2, cols=2, subplot_titles=[STAGE_LABELS[s] for s in stages])
    df = frame.sort_values("timestamp")
    for index, stage in enumerate(stages):
        row, col = divmod(index, 2)
        mean = df[stage].shift(1).rolling(20, min_periods=5).mean()
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df[stage], line={"color": "#2DD4BF", "width": 1}, name=stage, showlegend=False, hovertemplate="%{y:.0f}s<extra></extra>"), row=row + 1, col=col + 1)
        fig.add_trace(go.Scatter(x=df["timestamp"], y=mean, line={"color": "#6366F1", "width": 1.5}, showlegend=False, hovertemplate="mean %{y:.0f}s<extra></extra>"), row=row + 1, col=col + 1)
    return apply_theme(fig, title, 520)


def waterfall(frame: pd.DataFrame, title: str) -> go.Figure:
    means = [float(frame[stage].mean()) for stage in STAGE_LABELS]
    fig = go.Figure(go.Waterfall(x=[STAGE_LABELS[s] for s in STAGE_LABELS], y=means, measure=["relative"] * 4, connector={"line": {"color": "#1F2A4D"}}, increasing={"marker": {"color": "#2DD4BF"}}, text=[f"{value:.0f}s" for value in means], textposition="outside", hovertemplate="%{x}<br>%{y:.0f}s<extra></extra>"))
    fig.update_yaxes(title="Average seconds")
    return apply_theme(fig, title, 380)


def violin_platform(frame: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    for platform, color in zip(sorted(frame["platform"].unique()), ["#2DD4BF", "#6366F1", "#F59E0B"]):
        part = frame.loc[frame["platform"] == platform, "total_execution_time_sec"]
        fig.add_trace(go.Violin(y=part, name=platform, box_visible=True, meanline_visible=True, line_color=color, fillcolor="rgba(0,0,0,0)", hovertemplate="%{y:.0f}s<extra>" + platform + "</extra>"))
    fig.update_yaxes(title="Total execution (seconds)")
    return apply_theme(fig, title, 400)


def grouped_metrics(comparison: pd.DataFrame, title: str) -> go.Figure:
    metrics = ["accuracy", "precision", "recall", "f1"]
    fig = go.Figure()
    for name, label in MODEL_DISPLAY.items():
        row = comparison.loc[comparison["model"] == name].iloc[0]
        fig.add_trace(go.Bar(name=label, x=[m.title() for m in metrics], y=[row[m] * 100 for m in metrics], hovertemplate="%{x} %{y:.1f}%<extra>" + label + "</extra>"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="Score (%)", range=[0, 100])
    return apply_theme(fig, title, 420)


def curve_chart(metrics: dict, kind: str, title: str) -> go.Figure:
    fig = go.Figure()
    for name, label in MODEL_DISPLAY.items():
        payload = metrics["models"][name]
        if kind == "roc":
            fig.add_trace(go.Scatter(x=payload["fpr"], y=payload["tpr"], name=f"{label} AUC {payload['roc_auc']:.3f}", mode="lines", hovertemplate="FPR %{x:.2f}<br>TPR %{y:.2f}<extra></extra>"))
        elif kind == "pr":
            fig.add_trace(go.Scatter(x=payload["recall_curve"], y=payload["precision_curve"], name=f"{label} AP {payload['pr_auc']:.3f}", mode="lines", hovertemplate="Recall %{x:.2f}<br>Precision %{y:.2f}<extra></extra>"))
        else:
            fig.add_trace(go.Scatter(x=payload["calibration_mean_predicted"], y=payload["calibration_fraction_positive"], name=label, mode="lines+markers", hovertemplate="Predicted %{x:.2f}<br>Observed %{y:.2f}<extra></extra>"))
    if kind == "roc":
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line={"dash": "dash", "color": "#94A3B8"}))
    if kind == "calibration":
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect", line={"dash": "dash", "color": "#94A3B8"}))
    return apply_theme(fig, title, 420)


def confusion(metrics: dict, model: str, title: str) -> go.Figure:
    matrix = metrics["models"][model]["confusion_matrix"]
    fig = go.Figure(go.Heatmap(z=matrix, x=["Pred normal", "Pred bottleneck"], y=["Actual normal", "Actual bottleneck"], text=matrix, texttemplate="%{text}", colorscale=[[0, "#121A33"], [1, "#2DD4BF"]], hovertemplate="%{y} / %{x}<br>%{z}<extra></extra>", showscale=False))
    return apply_theme(fig, title, 380)


def importance_bars(table: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    for column, label in (("mdi_pct", "MDI"), ("permutation_pct", "Permutation"), ("shap_pct", "SHAP")):
        if column not in table or table[column].isna().all():
            continue
        fig.add_trace(go.Bar(name=label, y=table["label"], x=table[column], orientation="h", hovertemplate="%{y}<br>%{x:.1f}%<extra>" + label + "</extra>"))
    fig.update_layout(barmode="group")
    fig.update_xaxes(title="Share of importance (%)")
    return apply_theme(fig, title, 460)


def ablation_bars(table: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure(go.Bar(x=table["setting"], y=table["f1"] * 100, marker_color="#2DD4BF", text=[f"{v * 100:.1f}%" for v in table["f1"]], textposition="outside", hovertemplate="%{x}<br>F1 %{y:.1f}%<extra></extra>"))
    fig.update_yaxes(title="Test F1 (%)", range=[0, 100])
    return apply_theme(fig, title, 380)


def tau_tradeoff(report: dict, title: str) -> go.Figure:
    sweep = report.get("tau_sweep") or []
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[row["tau"] for row in sweep], y=[(row["false_alarm_rate"] or 0) * 100 for row in sweep], name="False-alarm rate", mode="lines+markers", line={"color": "#EF4444"}, hovertemplate="τ %{x:.2f}<br>False alarms %{y:.1f}%<extra></extra>"))
    fig.add_trace(go.Scatter(x=[row["tau"] for row in sweep], y=[(row["detection_rate"] or 0) * 100 for row in sweep], name="Detection rate", mode="lines+markers", line={"color": "#2DD4BF"}, hovertemplate="τ %{x:.2f}<br>Detection %{y:.1f}%<extra></extra>"))
    fig.add_hline(y=8, line_dash="dash", line_color="#F59E0B", annotation_text="8% target", annotation_font_color="#F59E0B")
    fig.add_vline(x=0.65, line_dash="dot", line_color="#6366F1")
    fig.update_yaxes(title="Rate (%)")
    fig.update_xaxes(title="Probability threshold τ")
    return apply_theme(fig, title, 400)


def gauge(value: float, title: str, height: int = 230) -> go.Figure:
    color = "#22C55E" if value < 0.4 else "#F59E0B" if value <= 0.65 else "#EF4444"
    fig = go.Figure(go.Indicator(mode="gauge+number", value=value * 100, number={"suffix": "%", "font": {"color": "#F8FAFC", "family": "JetBrains Mono"}}, title={"text": title, "font": {"color": "#94A3B8", "size": 13}}, gauge={"axis": {"range": [0, 100], "tickcolor": "#94A3B8"}, "bar": {"color": color}, "bgcolor": "#0B1020", "borderwidth": 0, "steps": [{"range": [0, 40], "color": "rgba(34,197,94,0.18)"}, {"range": [40, 65], "color": "rgba(245,158,11,0.20)"}, {"range": [65, 100], "color": "rgba(239,68,68,0.22)"}]}))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=height, margin={"l": 16, "r": 16, "t": 48, "b": 8}, font={"color": "#E5E7EB"})
    return fig


def sensitivity(labels: list[str], deltas: list[float], title: str) -> go.Figure:
    colors = ["#EF4444" if value > 0 else "#22C55E" for value in deltas]
    fig = go.Figure(go.Bar(y=labels, x=deltas, orientation="h", marker_color=colors, hovertemplate="%{y}<br>%{x:+.1%} risk vs cohort median<extra></extra>"))
    fig.update_xaxes(title="Change in bottleneck probability", tickformat="+.0%")
    return apply_theme(fig, title, 380)
