"""Predictive queue and the what-if simulator."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app import charts
from app.components import html, insight_callout, missing_file, section_header
from app.data import apply_filters, model_bundle, predictions
from config import FEATURE_LABELS

_PRESETS = ("Healthy team", "Big-bang commit", "Review backlog")


def _predict(model, scaler, features: list[str], values: dict[str, float]) -> float:
    row = pd.DataFrame([{name: values[name] for name in features}])
    return float(model.predict_proba(scaler.transform(row))[0, 1])


def _effects(model, scaler, features: list[str], values: dict[str, float], medians: dict[str, float]) -> dict[str, float]:
    base = _predict(model, scaler, features, values)
    try:
        import shap

        if hasattr(model, "estimators_") or hasattr(model, "feature_importances_"):
            frame = pd.DataFrame([{name: values[name] for name in features}])
            transformed = scaler.transform(frame)
            explanation = shap.TreeExplainer(model).shap_values(transformed)
            array = explanation
            if isinstance(array, list):
                array = array[-1]
            if getattr(array, "ndim", 1) == 3:
                array = array[:, :, -1]
            return {feature: float(array.reshape(-1)[index]) for index, feature in enumerate(features)}
    except Exception:
        pass
    deltas = {}
    for feature in features:
        alt = dict(values)
        alt[feature] = medians[feature]
        deltas[feature] = base - _predict(model, scaler, features, alt)
    return deltas


def _recommend(model, scaler, features: list[str], values: dict[str, float], medians: dict[str, float]) -> str:
    base = _predict(model, scaler, features, values)
    scenarios: list[tuple[str, dict[str, float]]] = []
    if values["code_churn"] > medians["code_churn"] * 1.3:
        alt = dict(values)
        alt["code_churn"] = max(values["code_churn"] / 3, medians["code_churn"] * 0.6)
        scenarios.append(("Split this commit into smaller changes", alt))
    if values["pr_review_latency_hours"] > 24:
        alt = dict(values)
        alt["pr_review_latency_hours"] = 24
        scenarios.append(("Set a 24-hour review SLA", alt))
    if values["code_churn"] > medians["code_churn"] and values["pr_review_latency_hours"] > 24:
        alt = dict(values)
        alt["code_churn"] = max(values["code_churn"] / 3, medians["code_churn"] * 0.6)
        alt["pr_review_latency_hours"] = 24
        scenarios.append(("Split this commit and set a 24h review SLA", alt))
    if values["historical_build_success_ratio"] < 0.75:
        alt = dict(values)
        alt["historical_build_success_ratio"] = min(0.95, values["historical_build_success_ratio"] + 0.15)
        alt["build_success_ratio"] = min(1.0, max(values["build_success_ratio"], 0.8))
        scenarios.append(("Stabilize recent builds before adding scope", alt))
    best_text, best_prob = None, base
    for text, alt in scenarios:
        prob = _predict(model, scaler, features, alt)
        if prob < best_prob:
            best_text, best_prob = text, prob
    if best_text and base - best_prob >= 0.02:
        return f"{best_text} to cut risk from {base:.0%} to {best_prob:.0%}."
    return f"Risk is {base:.0%}. None of the behavior edits moves it by more than 2 points. Keep the change small and reviewed inside a day."


def _bounds(frame: pd.DataFrame, feature: str) -> tuple[float, float, float]:
    if feature in {"build_success_ratio", "historical_build_success_ratio"}:
        return 0.0, 1.0, 0.01
    low = float(frame[feature].quantile(0.01))
    high = float(frame[feature].quantile(0.99))
    if high <= low:
        high = low + 1
    step = 1.0 if feature == "commit_frequency" else max((high - low) / 100, 0.1)
    return low, high, step


def _preset_values(frame: pd.DataFrame, name: str, features: list[str]) -> dict[str, float]:
    q = frame[features].quantile
    if name == "Healthy team":
        raw = {
            "commit_frequency": q(0.40)["commit_frequency"],
            "code_churn": q(0.35)["code_churn"],
            "pr_review_latency_hours": q(0.25)["pr_review_latency_hours"],
            "build_success_ratio": 0.9,
            "test_execution_duration_sec": q(0.40)["test_execution_duration_sec"],
            "deployment_delay_min": q(0.25)["deployment_delay_min"],
            "historical_build_success_ratio": 0.9,
            "build_duration_sec": q(0.40)["build_duration_sec"],
        }
    elif name == "Big-bang commit":
        raw = {
            "commit_frequency": q(0.80)["commit_frequency"],
            "code_churn": q(0.95)["code_churn"],
            "pr_review_latency_hours": q(0.90)["pr_review_latency_hours"],
            "build_success_ratio": 0.5,
            "test_execution_duration_sec": q(0.70)["test_execution_duration_sec"],
            "deployment_delay_min": q(0.80)["deployment_delay_min"],
            "historical_build_success_ratio": 0.55,
            "build_duration_sec": q(0.70)["build_duration_sec"],
        }
    else:
        raw = {feature: float(frame[feature].median()) for feature in features}
        raw["pr_review_latency_hours"] = float(q(0.95)["pr_review_latency_hours"])
    return {feature: float(raw[feature]) for feature in features}


def render(filters: dict) -> None:
    frame = predictions()
    bundle = model_bundle()
    section_header("Predictive risk", "Score the queue, then move the behavior sliders and watch the saved model respond.")
    if frame is None or bundle is None:
        missing_file("models/best_model.joblib", "python -m src.train")
        return
    view = apply_filters(frame, filters)
    if view is None or view.empty:
        html(insight_callout("NO MATCHES", "No runs match the current filters."))
        return
    model, scaler, features = bundle
    queue = view.sort_values("timestamp").tail(8)
    section_header("Live queue", "Eight most recent runs in this filter.")
    columns = st.columns(4)
    for index, row in enumerate(queue.itertuples(index=False)):
        with columns[index % 4]:
            charts.show(charts.gauge(float(row.bottleneck_probability), f"{row.project_name} · {row.run_id[-4:]}", 200))
    medians = {feature: float(frame[feature].median()) for feature in features}
    html('<div class="section"><h2>What-if simulator</h2><p>Sliders are in original units. The saved scaler and model score every change.</p></div>')
    for name in _PRESETS:
        st.button(name, key=f"preset_{name}", on_click=_apply_preset, args=(frame, name, features))
    values = {}
    left, right = st.columns(2, gap="medium")
    for index, feature in enumerate(features):
        low, high, step = _bounds(frame, feature)
        key = f"sim_{feature}"
        if key not in st.session_state:
            st.session_state[key] = float(min(max(medians[feature], low), high))
        column = left if index % 2 == 0 else right
        with column:
            values[feature] = float(st.slider(FEATURE_LABELS.get(feature, feature), min_value=float(low), max_value=float(high), step=float(step), key=key))
    probability = _predict(model, scaler, features, values)
    effects = _effects(model, scaler, features, values, medians)
    gcol, scol = st.columns((2, 3), gap="medium")
    with gcol:
        charts.show(charts.gauge(probability, "Bottleneck probability"))
        html(insight_callout("RECOMMENDATION", _recommend(model, scaler, features, values, medians)))
    with scol:
        labels = [FEATURE_LABELS.get(feature, feature) for feature in features]
        deltas = [effects[feature] for feature in features]
        charts.show(charts.sensitivity(labels, deltas, "What is pushing risk up or down versus the cohort median"))


def _apply_preset(frame: pd.DataFrame, name: str, features: list[str]) -> None:
    for feature, value in _preset_values(frame, name, features).items():
        low, high, _step = _bounds(frame, feature)
        st.session_state[f"sim_{feature}"] = float(min(max(value, low), high))
