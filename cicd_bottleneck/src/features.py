"""Feature validation: correlation, RFE, and the stratified split.

The split function lives here and is called by preprocessing *before*
imputation, so medians and the scaler see training rows only.
"""

from __future__ import annotations

import json
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from config import (
    CLEAN_RUNS,
    CORE_FEATURES,
    CORR_FLAG,
    FEATURE_VALIDATION_PATH,
    FIGURES,
    FIGURE_DPI,
    MODEL_FEATURES,
    RFE_N_FEATURES,
    SEED,
    SPLIT_PATH,
    get_logger,
)

logger = get_logger("features")


def make_split(frame: pd.DataFrame, seed: int = SEED) -> dict[str, Any]:
    """80/20 stratified split. Returns row positions into ``frame``."""
    positions = np.arange(len(frame))
    train_idx, test_idx = train_test_split(
        positions,
        test_size=0.20,
        random_state=seed,
        stratify=frame["bottleneck"].to_numpy(),
    )
    train_idx = sorted(int(i) for i in train_idx)
    test_idx = sorted(int(i) for i in test_idx)
    payload = {
        "seed": seed,
        "train_indices": train_idx,
        "test_indices": test_idx,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "train_run_ids": frame.iloc[train_idx]["run_id"].tolist(),
        "test_run_ids": frame.iloc[test_idx]["run_id"].tolist(),
    }
    return payload


def save_split(payload: dict[str, Any]) -> None:
    SPLIT_PATH.write_text(json.dumps(payload), encoding="utf-8")
    logger.info("Wrote %s (train=%d, test=%d).", SPLIT_PATH, payload["n_train"], payload["n_test"])


def load_split() -> dict[str, Any]:
    return json.loads(SPLIT_PATH.read_text(encoding="utf-8"))


def _heatmap(corr: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", font="DejaVu Sans")
    fig, ax = plt.subplots(figsize=(9.5, 8))
    sns.heatmap(
        corr,
        ax=ax,
        cmap="RdBu_r",
        center=0,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 8},
        square=True,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Pearson correlation of model features (training split)")
    fig.tight_layout()
    path = FIGURES / "correlation_heatmap.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Wrote %s", path)


def validate() -> dict[str, Any]:
    frame = pd.read_csv(CLEAN_RUNS)
    payload = make_split(frame, SEED)
    save_split(payload)
    train = frame.iloc[payload["train_indices"]]
    corr = train[MODEL_FEATURES].corr(method="pearson")
    _heatmap(corr)

    flagged: list[dict[str, float | str]] = []
    columns = list(corr.columns)
    for i, left in enumerate(columns):
        for right in columns[i + 1 :]:
            value = float(corr.loc[left, right])
            if abs(value) > CORR_FLAG:
                flagged.append({"feature_a": left, "feature_b": right, "r": value})
                logger.info("High correlation |r|=%.3f between %s and %s", abs(value), left, right)
    if not flagged:
        logger.info("No feature pair exceeded |r| > %.2f.", CORR_FLAG)

    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(train[CORE_FEATURES])
    y_train = train["bottleneck"].to_numpy()
    estimator = RandomForestClassifier(
        n_estimators=120,
        max_depth=8,
        min_samples_leaf=4,
        random_state=SEED,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    rfe = RFE(estimator=estimator, n_features_to_select=RFE_N_FEATURES, step=1)
    rfe.fit(x_train, y_train)
    ranking = [
        {"feature": feature, "rank": int(rank), "selected": bool(keep)}
        for feature, rank, keep in sorted(
            zip(CORE_FEATURES, rfe.ranking_, rfe.support_),
            key=lambda item: item[1],
        )
    ]
    for row in ranking:
        logger.info(
            "RFE %-32s rank=%d selected=%s",
            row["feature"],
            row["rank"],
            row["selected"],
        )

    report = {
        "correlation_flag": CORR_FLAG,
        "high_correlation_pairs": flagged,
        "rfe_base_estimator": "RandomForestClassifier",
        "rfe_n_features_to_select": RFE_N_FEATURES,
        "rfe_scope": "seven core features; build_duration_sec is kept for the timing ablation and is not dropped here",
        "rfe_ranking": ranking,
        "features_kept": [row["feature"] for row in ranking if row["selected"]],
        "note": (
            "RFE is diagnostic. Classifiers train on the full eight-feature set "
            "so the leakage ablation stays interpretable."
        ),
    }
    FEATURE_VALIDATION_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Wrote %s", FEATURE_VALIDATION_PATH)
    return report


def main() -> None:
    validate()


if __name__ == "__main__":
    main()
