"""Stage-level alerts that fire only when the model and the clock agree.

A stage is a confirmed bottleneck when bottleneck probability exceeds tau
and its duration stays more than 2 standard deviations above the previous
20-run mean for at least three consecutive runs.
"""

from __future__ import annotations

import json
import warnings
from typing import Any

import joblib

warnings.filterwarnings(
    "ignore",
    message=r"The `probability` parameter was deprecated.*",
    category=FutureWarning,
)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    ALERTS_PATH,
    DETECTION_REPORT_PATH,
    DETECT_MIN_HISTORY,
    DETECT_PERSISTENCE,
    DETECT_TAU,
    DETECT_TAU_GRID,
    DETECT_WINDOW,
    DETECT_Z_THRESHOLD,
    FALSE_ALARM_TARGET,
    FEATURE_LABELS,
    FIGURES,
    FIGURE_DPI,
    MODEL_FEATURES,
    PREDICTIONS_PATH,
    SCALER_PATH,
    STAGE_COLUMNS,
    STAGE_LABELS,
    get_logger,
)
from src.features import load_split

logger = get_logger("detect")


def _stage_scores(durations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z_scores = np.full(len(durations), np.nan)
    streaks = np.zeros(len(durations), dtype=int)
    streak = 0
    for index, value in enumerate(durations):
        history = durations[max(0, index - DETECT_WINDOW) : index]
        if len(history) >= DETECT_MIN_HISTORY and float(np.std(history)) > 1e-8:
            z_scores[index] = (value - float(history.mean())) / float(np.std(history))
        if np.isfinite(z_scores[index]) and z_scores[index] > DETECT_Z_THRESHOLD:
            streak += 1
        else:
            streak = 0
        streaks[index] = streak
    return z_scores, streaks


def _candidate_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    """One row per (run, stage) that satisfies the persistence rule."""
    frames: list[pd.DataFrame] = []
    ordered = predictions.sort_values(["project_id", "timestamp"]).reset_index(drop=True)
    for project_id, project in ordered.groupby("project_id", sort=False):
        project = project.reset_index(drop=True)
        for stage in STAGE_COLUMNS:
            z_scores, streaks = _stage_scores(project[stage].to_numpy(dtype=float))
            keep = streaks >= DETECT_PERSISTENCE
            if not np.any(keep):
                continue
            chunk = project.loc[keep].copy()
            chunk["stage"] = stage
            chunk["z_score"] = z_scores[keep]
            chunk["anomaly_window_runs"] = streaks[keep]
            chunk["project_position"] = np.flatnonzero(keep)
            frames.append(chunk)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _contributions(candidates: pd.DataFrame, train_median: np.ndarray) -> list[str]:
    """One-at-a-time drop toward the training median. Positive raises risk."""
    if candidates.empty:
        return []
    model = joblib.load(config_model())
    scaler = joblib.load(SCALER_PATH)
    matrix = candidates[MODEL_FEATURES].to_numpy(dtype=float)
    base = model.predict_proba(scaler.transform(matrix))[:, 1]
    deltas = np.zeros_like(matrix)
    for column in range(matrix.shape[1]):
        tweaked = matrix.copy()
        tweaked[:, column] = train_median[column]
        deltas[:, column] = base - model.predict_proba(scaler.transform(tweaked))[:, 1]
    encoded: list[str] = []
    for row in deltas:
        order = np.argsort(-np.abs(row))[:3]
        encoded.append(
            ";".join(f"{MODEL_FEATURES[index]}:{row[index]:+.4f}" for index in order)
        )
    return encoded


def config_model() -> Any:
    from config import BEST_MODEL_PATH

    return BEST_MODEL_PATH


def _severity(z_score: float, probability: float) -> str:
    if z_score >= 3.5 or (z_score >= 3.0 and probability >= 0.85):
        return "high"
    if z_score >= 2.5 or probability >= 0.75:
        return "medium"
    return "low"


def _reason(row: pd.Series) -> str:
    parts = []
    for piece in str(row["top_3_contributing_features"]).split(";"):
        feature = piece.split(":")[0]
        parts.append(FEATURE_LABELS.get(feature, feature))
    drivers = ", ".join(parts) if parts else "the recorded behavior features"
    return (
        f"{STAGE_LABELS[row['stage']]} stayed {row['z_score']:.1f} standard deviations "
        f"above its previous {DETECT_WINDOW}-run baseline for {int(row['anomaly_window_runs'])} "
        f"consecutive runs. Bottleneck probability is {row['ml_probability']:.0%}. "
        f"The largest local contributions come from {drivers}."
    )


def _lead_hours(predictions: pd.DataFrame, alerts: pd.DataFrame, tau: float) -> float | None:
    if alerts.empty:
        return None
    leads: list[float] = []
    ordered = predictions.sort_values(["project_id", "timestamp"])
    for project_id, project in ordered.groupby("project_id", sort=False):
        project = project.reset_index(drop=True)
        times = pd.to_datetime(project["timestamp"], utc=True)
        probs = project["bottleneck_probability"].to_numpy(dtype=float)
        positions = {run_id: index for index, run_id in enumerate(project["run_id"])}
        project_alerts = alerts[(alerts["project_id"] == project_id) & (alerts["true_label"] == 1)]
        for run_id in project_alerts["run_id"].unique():
            index = positions.get(run_id)
            if index is None:
                continue
            cursor = index
            while cursor - 1 >= 0 and probs[cursor - 1] > tau:
                cursor -= 1
            leads.append((times.iloc[index] - times.iloc[cursor]).total_seconds() / 3600.0)
    if not leads:
        return None
    return float(np.mean(leads))


def _rates(alerts: pd.DataFrame, test_runs: pd.DataFrame) -> dict[str, float | int | None]:
    test_ids = set(test_runs["run_id"])
    scoped = alerts[alerts["run_id"].isin(test_ids)] if not alerts.empty else alerts
    positives = test_runs[test_runs["true_label"] == 1]
    if scoped.empty:
        false_alarm = None
        detected = 0
    else:
        false_alarm = float((scoped["true_label"] == 0).mean())
        detected = int(scoped.loc[scoped["true_label"] == 1, "run_id"].nunique())
    detection = float(detected / len(positives)) if len(positives) else None
    return {
        "n_alerts": int(len(scoped)),
        "false_alarm_rate": false_alarm,
        "detection_rate": detection,
    }


def detect() -> pd.DataFrame:
    predictions = pd.read_csv(PREDICTIONS_PATH)
    split = load_split()
    test_runs = predictions[predictions["run_id"].isin(set(split["test_run_ids"]))].copy()
    train_rows = predictions[predictions["run_id"].isin(set(split["train_run_ids"]))]
    train_median = train_rows[MODEL_FEATURES].median().to_numpy(dtype=float)

    candidates = _candidate_frame(predictions)
    if candidates.empty:
        logger.warning("Persistence rule produced no candidate stages.")
        empty = pd.DataFrame(
            columns=[
                "alert_id",
                "project_id",
                "project_name",
                "run_id",
                "developer_id",
                "platform",
                "timestamp",
                "stage",
                "z_score",
                "anomaly_window_runs",
                "ml_probability",
                "top_3_contributing_features",
                "severity_label",
                "plain_language_reason",
                "true_label",
            ]
        )
        empty.to_csv(ALERTS_PATH, index=False)
        report = {
            "tau": DETECT_TAU,
            "z_threshold": DETECT_Z_THRESHOLD,
            "persistence_runs": DETECT_PERSISTENCE,
            "window": DETECT_WINDOW,
            "evaluation_scope": "held-out test runs; rolling history uses each project's full timeline",
            "n_alerts_all_runs": 0,
            "n_alerts_test": 0,
            "false_alarm_rate": None,
            "detection_rate": 0.0,
            "mean_lead_time_hours": None,
            "false_alarm_target": FALSE_ALARM_TARGET,
            "target_met": False,
            "tau_sweep": [],
            "lead_time_definition": (
                "Mean hours, over true-positive test alerts, from the earliest preceding "
                "run whose bottleneck probability already exceeded tau to the alerted run."
            ),
        }
        DETECTION_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return empty

    sweep = []
    for tau in DETECT_TAU_GRID:
        subset = candidates[candidates["bottleneck_probability"] > tau].copy()
        subset["true_label"] = subset["true_label"]
        rates = _rates(subset, test_runs)
        sweep.append({"tau": tau, **rates})
        logger.info(
            "tau %.2f test alerts %s false-alarm %s detection %s",
            tau,
            rates["n_alerts"],
            rates["false_alarm_rate"],
            rates["detection_rate"],
        )

    confirmed = candidates[candidates["bottleneck_probability"] > DETECT_TAU].copy()
    confirmed = confirmed.sort_values(["timestamp", "project_id", "stage"]).reset_index(drop=True)
    if not confirmed.empty:
        confirmed["top_3_contributing_features"] = _contributions(confirmed, train_median)
        confirmed["ml_probability"] = confirmed["bottleneck_probability"]
        confirmed["severity_label"] = [
            _severity(float(z), float(p))
            for z, p in zip(confirmed["z_score"], confirmed["ml_probability"])
        ]
        confirmed["plain_language_reason"] = confirmed.apply(_reason, axis=1)
        confirmed["alert_id"] = [f"alert_{index + 1:06d}" for index in range(len(confirmed))]
        confirmed["z_score"] = confirmed["z_score"].round(3)
    else:
        confirmed["top_3_contributing_features"] = []
        confirmed["ml_probability"] = []
        confirmed["severity_label"] = []
        confirmed["plain_language_reason"] = []
        confirmed["alert_id"] = []

    alerts = confirmed[
        [
            "alert_id",
            "project_id",
            "project_name",
            "run_id",
            "developer_id",
            "platform",
            "timestamp",
            "stage",
            "z_score",
            "anomaly_window_runs",
            "ml_probability",
            "top_3_contributing_features",
            "severity_label",
            "plain_language_reason",
            "true_label",
        ]
    ].copy()
    alerts.to_csv(ALERTS_PATH, index=False)
    logger.info("Wrote %s (%d alerts).", ALERTS_PATH, len(alerts))

    test_alerts = alerts[alerts["run_id"].isin(set(test_runs["run_id"]))]
    primary = _rates(alerts, test_runs)
    lead = _lead_hours(predictions, test_alerts, DETECT_TAU)
    false_alarm = primary["false_alarm_rate"]
    target_met = bool(false_alarm is not None and false_alarm < FALSE_ALARM_TARGET)
    report = {
        "tau": DETECT_TAU,
        "z_threshold": DETECT_Z_THRESHOLD,
        "persistence_runs": DETECT_PERSISTENCE,
        "window": DETECT_WINDOW,
        "evaluation_scope": "held-out test runs; rolling history uses each project's full timeline",
        "n_alerts_all_runs": int(len(alerts)),
        "n_alerts_test": primary["n_alerts"],
        "false_alarm_rate": false_alarm,
        "detection_rate": primary["detection_rate"],
        "mean_lead_time_hours": lead,
        "false_alarm_target": FALSE_ALARM_TARGET,
        "target_met": target_met,
        "alerts_by_stage": alerts["stage"].value_counts().to_dict() if not alerts.empty else {},
        "alerts_by_severity": alerts["severity_label"].value_counts().to_dict() if not alerts.empty else {},
        "tau_sweep": sweep,
        "lead_time_definition": (
            "Mean hours, over true-positive test alerts, from the earliest preceding run "
            "whose bottleneck probability already exceeded tau to the alerted run. "
            "Zero means probability first crossed tau on the alerted run."
        ),
        "false_alarm_definition": (
            "Share of confirmed alerts on held-out runs whose true label is normal, "
            "divided by all confirmed alerts on held-out runs."
        ),
    }
    DETECTION_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info(
        "Test false-alarm rate=%s (target < %.0f%%: %s), detection rate=%s, mean lead=%s h",
        false_alarm,
        FALSE_ALARM_TARGET * 100,
        "MET" if target_met else "NOT MET",
        primary["detection_rate"],
        "n/a" if lead is None else f"{lead:.2f}",
    )

    fig, ax = plt.subplots(figsize=(7.6, 5))
    taus = [row["tau"] for row in sweep]
    far = [0 if row["false_alarm_rate"] is None else row["false_alarm_rate"] * 100 for row in sweep]
    det = [0 if row["detection_rate"] is None else row["detection_rate"] * 100 for row in sweep]
    ax.plot(taus, far, marker="o", color="#DC2626", label="False-alarm rate")
    ax.plot(taus, det, marker="o", color="#0F766E", label="Detection rate")
    ax.axhline(FALSE_ALARM_TARGET * 100, color="#94A3B8", linestyle="--", label="8% false-alarm target")
    ax.axvline(DETECT_TAU, color="#4F46E5", linestyle=":", label="Operating tau 0.65")
    ax.set_xlabel("Probability threshold τ")
    ax.set_ylabel("Rate (%)")
    ax.set_title("Detection trade-off on held-out runs")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "tau_tradeoff.png", dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return alerts


def main() -> None:
    detect()


if __name__ == "__main__":
    main()
