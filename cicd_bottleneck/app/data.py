"""Load the Part 1 contract and apply the global filters."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from config import (
    ABLATION_PATH,
    ALERTS_PATH,
    BEST_MODEL_PATH,
    DETECTION_REPORT_PATH,
    FEATURE_IMPORTANCE_PATH,
    FEATURE_ORDER_PATH,
    METRICS_PATH,
    MODEL_COMPARISON_PATH,
    PREDICTIONS_PATH,
    SCALER_PATH,
)


def _mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else -1.0


@st.cache_data(show_spinner=False)
def load_predictions(_mtime: float) -> pd.DataFrame | None:
    if not PREDICTIONS_PATH.exists():
        return None
    frame = pd.read_csv(PREDICTIONS_PATH)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame


@st.cache_data(show_spinner=False)
def load_alerts(_mtime: float, _pred_mtime: float) -> pd.DataFrame | None:
    if not ALERTS_PATH.exists():
        return None
    alerts = pd.read_csv(ALERTS_PATH)
    if alerts.empty:
        return alerts
    alerts["timestamp"] = pd.to_datetime(alerts["timestamp"], utc=True)
    if PREDICTIONS_PATH.exists() and "risk_band" not in alerts.columns:
        preds = pd.read_csv(PREDICTIONS_PATH, usecols=["run_id", "risk_band"])
        alerts = alerts.merge(preds, on="run_id", how="left")
    return alerts


@st.cache_data(show_spinner=False)
def load_json(_mtime: float, path_str: str) -> dict | None:
    path = Path(path_str)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_table(_mtime: float, path_str: str) -> pd.DataFrame | None:
    path = Path(path_str)
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_resource(show_spinner=False)
def load_model(_model_mtime: float, _scaler_mtime: float) -> tuple | None:
    if not (BEST_MODEL_PATH.exists() and SCALER_PATH.exists() and FEATURE_ORDER_PATH.exists()):
        return None
    model = joblib.load(BEST_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    features = json.loads(FEATURE_ORDER_PATH.read_text(encoding="utf-8"))
    return model, scaler, features


def predictions() -> pd.DataFrame | None:
    return load_predictions(_mtime(PREDICTIONS_PATH))


def alerts() -> pd.DataFrame | None:
    return load_alerts(_mtime(ALERTS_PATH), _mtime(PREDICTIONS_PATH))


def metrics() -> dict | None:
    return load_json(_mtime(METRICS_PATH), str(METRICS_PATH))


def detection() -> dict | None:
    return load_json(_mtime(DETECTION_REPORT_PATH), str(DETECTION_REPORT_PATH))


def importance() -> pd.DataFrame | None:
    return load_table(_mtime(FEATURE_IMPORTANCE_PATH), str(FEATURE_IMPORTANCE_PATH))


def ablation() -> pd.DataFrame | None:
    return load_table(_mtime(ABLATION_PATH), str(ABLATION_PATH))


def comparison() -> pd.DataFrame | None:
    return load_table(_mtime(MODEL_COMPARISON_PATH), str(MODEL_COMPARISON_PATH))


def model_bundle():
    return load_model(_mtime(BEST_MODEL_PATH), _mtime(SCALER_PATH))


def apply_filters(frame: pd.DataFrame | None, filters: dict) -> pd.DataFrame | None:
    """Apply the sidebar filters. Missing columns are skipped so alerts still work."""
    if frame is None:
        return None
    if frame.empty:
        return frame
    out = frame
    if filters.get("projects") and "project_id" in out.columns:
        out = out[out["project_id"].isin(filters["projects"])]
    if filters.get("developers") and "developer_id" in out.columns:
        out = out[out["developer_id"].isin(filters["developers"])]
    if filters.get("platforms") and "platform" in out.columns:
        out = out[out["platform"].isin(filters["platforms"])]
    if filters.get("risk_bands") and "risk_band" in out.columns:
        out = out[out["risk_band"].isin(filters["risk_bands"])]
    if filters.get("start") is not None and "timestamp" in out.columns and len(out):
        start = pd.Timestamp(filters["start"], tz="UTC")
        end = pd.Timestamp(filters["end"], tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        stamp = pd.to_datetime(out["timestamp"], utc=True)
        out = out[(stamp >= start) & (stamp <= end)]
    return out


def project_medians(frame: pd.DataFrame) -> pd.Series:
    return frame.groupby("project_id")["total_execution_time_sec"].median()


def hours_at_risk(filtered: pd.DataFrame, medians: pd.Series) -> float:
    excess = filtered["total_execution_time_sec"] - filtered["project_id"].map(medians)
    mask = (filtered["predicted_label"] == 1) & (excess > 0)
    return float(excess[mask].sum() / 3600.0)


def active_filter_count(filters: dict, defaults: dict) -> int:
    count = 0
    for key in ("projects", "developers", "platforms", "risk_bands"):
        if set(filters.get(key) or []) != set(defaults.get(key) or []):
            count += 1
    if filters.get("start") != defaults.get("start") or filters.get("end") != defaults.get("end"):
        count += 1
    return count


def report_html(filtered: pd.DataFrame, metrics_payload: dict | None, alert_count: int) -> str:
    rate = float(filtered["true_label"].mean()) if len(filtered) else 0.0
    f1 = "n/a"
    model = "n/a"
    if metrics_payload:
        model = metrics_payload.get("best_model_display", "n/a")
        f1 = f"{metrics_payload.get('best_f1', 0) * 100:.1f}%"
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>PipelineIQ report</title>
    <style>body{{font-family:Inter,sans-serif;background:#0B1020;color:#E5E7EB;padding:32px}}
    .card{{background:#121A33;border:1px solid #1F2A4D;border-radius:14px;padding:16px;margin-bottom:12px}}</style>
    </head><body><h1>PipelineIQ</h1><p>Filtered view exported from the dashboard.</p>
    <div class="card">Runs in view: {len(filtered):,}<br>Bottleneck rate: {rate:.1%}<br>
    Active alerts in view: {alert_count:,}<br>Best model: {model} · held-out F1 {f1}</div>
    </body></html>"""
