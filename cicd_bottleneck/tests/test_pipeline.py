"""Checks against the generated study, not a toy fixture."""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from config import (
    ABLATION_PATH,
    ALERTS_PATH,
    BEST_MODEL_PATH,
    CLEAN_RUNS,
    DETECTION_REPORT_PATH,
    FEATURE_IMPORTANCE_PATH,
    FEATURE_ORDER_PATH,
    FIGURES,
    IMPUTE_MEDIANS_PATH,
    METRICS_PATH,
    MODEL_COMPARISON_PATH,
    MODEL_FEATURES,
    PRE_IMPUTE_RUNS,
    PREDICTIONS_PATH,
    RAW_RUNS,
    SCALER_PATH,
    SPLIT_PATH,
)
from src.generate_data import assign_bottleneck_labels


def test_catalog_and_raw_quality() -> None:
    raw = pd.read_csv(RAW_RUNS)
    assert raw["run_id"].nunique() == 4800
    assert len(raw) == 4896
    assert raw["project_id"].nunique() == 12
    assert raw[MODEL_FEATURES].isna().any().any()


def test_clean_shape_ratio_and_label_rule() -> None:
    clean = pd.read_csv(CLEAN_RUNS)
    assert len(clean) == 4800
    assert clean["run_id"].is_unique
    assert clean["project_id"].nunique() == 12
    rate = float(clean["bottleneck"].mean())
    assert 0.32 <= rate <= 0.38
    expected = assign_bottleneck_labels(clean)
    mismatch = expected.to_numpy() != clean["bottleneck"].to_numpy()
    if mismatch.any():
        medians = clean.groupby("project_id")["total_execution_time_sec"].transform("median")
        gap = (clean.loc[mismatch, "total_execution_time_sec"] - 1.5 * medians[mismatch]).abs()
        assert float(gap.max()) < 1e-3


def test_split_is_disjoint_and_scaler_uses_train_only() -> None:
    clean = pd.read_csv(CLEAN_RUNS)
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    train = set(split["train_indices"])
    test = set(split["test_indices"])
    assert train.isdisjoint(test)
    assert len(train) == 3840 and len(test) == 960
    scaler = joblib.load(SCALER_PATH)
    order = json.loads(FEATURE_ORDER_PATH.read_text(encoding="utf-8"))
    train_x = clean.iloc[sorted(train)][order]
    assert np.allclose(scaler.data_min_, train_x.min().to_numpy())
    assert np.allclose(scaler.data_max_, train_x.max().to_numpy())


def test_imputation_medians_come_from_training_rows() -> None:
    pre = pd.read_csv(PRE_IMPUTE_RUNS)
    clean = pd.read_csv(CLEAN_RUNS)
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    medians = json.loads(IMPUTE_MEDIANS_PATH.read_text(encoding="utf-8"))
    train_idx = split["train_indices"]
    for column, median in medians.items():
        observed = pre.iloc[train_idx][column].dropna()
        assert abs(float(observed.median()) - float(median)) < 1e-6
        missing = pre[column].isna()
        if missing.any():
            assert np.allclose(clean.loc[missing, column].astype(float), float(median))


def test_contract_files_and_prediction_schema() -> None:
    for path in (
        PREDICTIONS_PATH,
        ALERTS_PATH,
        BEST_MODEL_PATH,
        SCALER_PATH,
        FEATURE_ORDER_PATH,
        METRICS_PATH,
        MODEL_COMPARISON_PATH,
        FEATURE_IMPORTANCE_PATH,
        ABLATION_PATH,
        DETECTION_REPORT_PATH,
    ):
        assert path.exists(), path
    assert any(FIGURES.glob("*.png"))
    preds = pd.read_csv(PREDICTIONS_PATH)
    assert len(preds) == 4800
    for column in (
        "run_id",
        "project_id",
        "developer_id",
        "platform",
        "timestamp",
        *MODEL_FEATURES,
        "build_compilation",
        "unit_testing",
        "integration_testing",
        "deployment",
        "total_execution_time_sec",
        "true_label",
        "predicted_label",
        "bottleneck_probability",
        "risk_band",
    ):
        assert column in preds.columns
    assert set(preds["risk_band"]).issubset({"green", "amber", "red"})
    alerts = pd.read_csv(ALERTS_PATH)
    for column in (
        "alert_id",
        "project_id",
        "run_id",
        "timestamp",
        "stage",
        "z_score",
        "anomaly_window_runs",
        "ml_probability",
        "top_3_contributing_features",
        "severity_label",
        "plain_language_reason",
    ):
        assert column in alerts.columns
