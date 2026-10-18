"""Model credibility: metrics, importance, ablation, detection."""

from __future__ import annotations

import streamlit as st

from app import charts
from app.components import html, insight_callout, missing_file, section_header, styled_table
from app.data import ablation, comparison, detection, importance, metrics
from config import MODEL_DISPLAY


def render(filters: dict) -> None:
    del filters
    section_header("Model lab", "Every number on this page is read from the report files, not typed in.")
    payload = metrics()
    table = comparison()
    if payload is None or table is None:
        missing_file("reports/metrics.json", "python -m src.evaluate")
        return
    winner = payload["best_model_display"]
    charts.show(charts.grouped_metrics(table, f"{winner} leads on held-out F1 ({payload['best_f1'] * 100:.1f}%)"))
    styled_table(
        ["Model", "Accuracy", "Precision", "Recall", "F1", "F1 mean ± std", "Paper F1"],
        [
            [
                row.display_name,
                f"{row.accuracy * 100:.1f}%",
                f"{row.precision * 100:.1f}%",
                f"{row.recall * 100:.1f}%",
                f"{row.f1 * 100:.1f}%",
                f"{row.f1_mean * 100:.1f}% ± {row.f1_std * 100:.1f}",
                f"{row.paper_f1:.1f}%",
            ]
            for row in table.itertuples(index=False)
        ],
    )
    st.caption("Paper percentages are a published reference. This study did not tune the simulator to match them.")
    roc, pr, matrix, calibration = st.tabs(["ROC", "Precision-recall", "Confusion", "Calibration"])
    with roc:
        charts.show(charts.curve_chart(payload, "roc", "ROC on the 960-run test set"))
    with pr:
        charts.show(charts.curve_chart(payload, "pr", "Precision-recall. F1 is the ranking metric because the classes are uneven."))
    with matrix:
        choice = st.selectbox("Model", list(MODEL_DISPLAY), format_func=lambda key: MODEL_DISPLAY[key])
        charts.show(charts.confusion(payload, choice, f"Confusion matrix · {MODEL_DISPLAY[choice]}"))
    with calibration:
        charts.show(charts.curve_chart(payload, "calibration", "Reliability of predicted probabilities"))

    imp = importance()
    if imp is None:
        missing_file("reports/feature_importance.csv", "python -m src.explain")
    else:
        section_header("What the model uses", "MDI can favor high-cardinality features. Permutation and SHAP are the check.")
        charts.show(charts.importance_bars(imp, "MDI, permutation, and SHAP on the same feature order"))
        shares = imp.groupby("group")["mdi_pct"].sum()
        behavioral = float(shares.get("behavioral", 0))
        timing = float(shares.get("timing", 0))
        delay = float(shares.get("deployment_delay", 0))
        html(
            f"""
            <div class="story">
              {insight_callout("BEHAVIOR", f"{behavioral:.1f}% of Random Forest MDI")}
              {insight_callout("TIMING", f"{timing:.1f}% from build and test duration")}
              {insight_callout("DELAY", f"{delay:.1f}% from deployment delay, reported on its own")}
            </div>
            """
        )
    ab = ablation()
    if ab is not None:
        drop = float(ab.loc[ab["setting"] == "all_features", "f1"].iloc[0] - ab.loc[ab["setting"] == "behavior_only", "f1"].iloc[0])
        charts.show(charts.ablation_bars(ab, f"Removing timing features changes test F1 by {-drop * 100:+.1f} points versus behavior-only"))

    report = detection()
    if report is None:
        missing_file("reports/detection_report.json", "python -m src.detect")
        return
    section_header("Detection layer", report.get("evaluation_scope", ""))
    far = report.get("false_alarm_rate")
    det = report.get("detection_rate")
    left, right = st.columns((1, 2), gap="medium")
    with left:
        if far is None:
            html(insight_callout("NO TEST ALERTS", "The persistence rule did not confirm any held-out alerts."))
        else:
            met = "meets" if report.get("target_met") else "misses"
            charts.show(charts.gauge(min(far, 1), f"False-alarm rate · {met} the 8% target"))
            html(insight_callout("DETECTION", f"Held-out detection rate {0 if det is None else det:.1%}. Lead time {report.get('mean_lead_time_hours') or 0:.2f} h."))
    with right:
        charts.show(charts.tau_tradeoff(report, "False alarms versus detection as τ moves from 0.50 to 0.90"))
