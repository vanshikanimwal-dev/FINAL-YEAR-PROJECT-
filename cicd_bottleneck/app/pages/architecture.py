"""Method, dataset, and limitations."""

from __future__ import annotations

import streamlit as st

from app.components import html, insight_callout, section_header, styled_table
from app.data import apply_filters, predictions
from config import FEATURE_LABELS

_LAYERS = [
    ("1 · Collect", "Jenkins, GitHub Actions, GitLab CI, Git, webhooks. Each row is one pipeline run."),
    ("2 · Process", "Log parse, UTC clocks, webhook dedupe, train-only median imputation, train-only min-max scaling."),
    ("3 · Model", "Logistic regression, RBF SVM, gradient boosting, and random forest, plus a hybrid stack of all four. The saved model is the best held-out F1."),
    ("4 · Detect", "Alert only if probability > 0.65 and a stage stays 2σ hot for 3 runs."),
    ("5 · Decide", "This dashboard. Filters, developer coaching, and a live what-if on the saved model."),
]

_DEFS = {
    "commit_frequency": "Rolling 7-day commit count. Poisson.",
    "code_churn": "Lines added plus deleted. Log-normal, heavy tail.",
    "pr_review_latency_hours": "Hours from submission to approval. Log-normal.",
    "build_success_ratio": "Successes in the last 10 builds, on a 0.1 grid.",
    "test_execution_duration_sec": "This run's unit plus integration time.",
    "deployment_delay_min": "Minutes between the project baseline and deploy start. Gamma.",
    "historical_build_success_ratio": "Developer's long-run success rate. Beta, plus a little noise.",
    "build_duration_sec": "This run's compilation stage, the timing column.",
}


def render(filters: dict) -> None:
    section_header("Architecture and method", "Five layers, from raw events to a decision a team can act on.")
    cards = "".join(f'<div class="layer-card"><strong>{title}</strong><span class="muted">{body}</span></div>' for title, body in _LAYERS)
    html(f'<div class="layers">{cards}</div>')
    for title, body in _LAYERS:
        with st.expander(title):
            st.write(body)
    frame = predictions()
    view = apply_filters(frame, filters) if frame is not None else None
    if view is not None and len(view):
        rate = float(view["true_label"].mean())
        html(
            insight_callout(
                "DATASET",
                f"This filter shows {len(view):,} of the simulated runs across {view['project_id'].nunique()} projects. "
                f"Bottleneck rate here is {rate:.1%}. The label is 1 when total time exceeds 1.5× that project's own median.",
            )
        )
    section_header("The seven features", "Medians below are computed from the current filter, not hard-coded.")
    if view is not None and len(view):
        rows = []
        for feature, text in _DEFS.items():
            if feature not in view.columns:
                continue
            rows.append([FEATURE_LABELS.get(feature, feature), f"{view[feature].median():.2f}", text])
        styled_table(["Feature", "Median in view", "Definition"], rows)
    section_header("Limitations", "Say these out loud if the jury asks what would break in production.")
    html(
        insight_callout(
            "HONEST LIMITS",
            "The runs are simulated, so generalization to a real Jenkins or GitHub organization is unproven. "
            "The label uses each project's full-sample median, which you only know after the fact. "
            "Developer behavior will drift. Seven features omit review text, flaky tests, and queue contention. "
            "Future work: real GitHub and GitLab logs, sequence models, NLP on commits and review comments, and online updates.",
        )
    )
