"""Train four binary classifiers and keep the one with the best test F1.

Hyperparameters are chosen by 5-fold stratified GridSearchCV on the training
split only. The SVM search uses ``probability=False`` (F1 only needs labels)
and the winning settings are refit with ``probability=True``.
"""

from __future__ import annotations

import json
import warnings
from typing import Any

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings(
    "ignore",
    message=r"The `probability` parameter was deprecated.*",
    category=FutureWarning,
)
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC

from config import (
    BEST_MODEL_PATH,
    BEST_PARAMS_PATH,
    CLEAN_RUNS,
    CV_FOLDS,
    MODEL_DISPLAY,
    MODEL_FEATURES,
    MODELS,
    N_JOBS,
    PARAM_GRIDS,
    PREDICT_THRESHOLD,
    PREDICTIONS_PATH,
    RISK_AMBER_MAX,
    RISK_GREEN_MAX,
    SCALER_PATH,
    SEED,
    get_logger,
)
from src.features import load_split

logger = get_logger("train")


def risk_band(probability: float) -> str:
    """Map a bottleneck probability onto the dashboard's three bands."""
    if probability < RISK_GREEN_MAX:
        return "green"
    if probability <= RISK_AMBER_MAX:
        return "amber"
    return "red"


def build_estimator(name: str, seed: int, probability: bool = True) -> Any:
    """Return an unfitted estimator. SVM probability is optional during search."""
    if name == "logistic_regression":
        return LogisticRegression(
            max_iter=800,
            class_weight="balanced",
            random_state=seed,
            solver="lbfgs",
        )
    if name == "svm_rbf":
        return SVC(
            kernel="rbf",
            probability=probability,
            class_weight="balanced",
            random_state=seed,
            cache_size=1000,
        )
    if name == "gradient_boosting":
        return GradientBoostingClassifier(random_state=seed)
    if name == "random_forest":
        return RandomForestClassifier(
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=1,
        )
    raise KeyError(name)


def tune_model(
    name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
) -> tuple[Any, dict[str, Any], float]:
    """Grid-search one model. Returns the fitted estimator, params, and CV F1."""
    search_seed_model = build_estimator(name, seed, probability=False)
    grid = GridSearchCV(
        estimator=search_seed_model,
        param_grid=PARAM_GRIDS[name],
        scoring="f1",
        cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed),
        n_jobs=N_JOBS,
        refit=True,
        error_score="raise",
    )
    logger.info("Tuning %s (%d candidates, %d folds)...", name, len(grid.param_grid) and _grid_size(PARAM_GRIDS[name]), CV_FOLDS)
    grid.fit(x_train, y_train)
    params = dict(grid.best_params_)
    if name == "svm_rbf":
        model = build_estimator(name, seed, probability=True)
        model.set_params(**params)
        model.fit(x_train, y_train)
    else:
        model = grid.best_estimator_
    logger.info(
        "%s best CV F1 %.4f with %s",
        MODEL_DISPLAY[name],
        float(grid.best_score_),
        params,
    )
    return model, params, float(grid.best_score_)


def _grid_size(grid: dict[str, list]) -> int:
    size = 1
    for values in grid.values():
        size *= len(values)
    return size


def load_matrices(seed: int = SEED) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any]:
    """Return the cleaned frame, scaled matrices, labels, and the seed-42 scaler."""
    frame = pd.read_csv(CLEAN_RUNS)
    split = load_split()
    if split["seed"] != seed and seed == SEED:
        logger.warning("split.json seed is %s, expected %s", split["seed"], seed)
    scaler = joblib.load(SCALER_PATH)
    scaled = scaler.transform(frame[MODEL_FEATURES])
    y = frame["bottleneck"].to_numpy(dtype=int)
    train_idx = np.array(split["train_indices"])
    test_idx = np.array(split["test_indices"])
    return frame, scaled, y, train_idx, test_idx, scaler


def train(seed: int = SEED) -> dict[str, Any]:
    frame, scaled, y, train_idx, test_idx, _scaler = load_matrices(seed)
    x_train, y_train = scaled[train_idx], y[train_idx]
    x_test, y_test = scaled[test_idx], y[test_idx]

    fitted: dict[str, Any] = {}
    params: dict[str, Any] = {}
    cv_scores: dict[str, float] = {}
    test_f1: dict[str, float] = {}
    from sklearn.metrics import f1_score

    for name in PARAM_GRIDS:
        model, best_params, cv_f1 = tune_model(name, x_train, y_train, seed)
        fitted[name] = model
        params[name] = best_params
        cv_scores[name] = cv_f1
        predicted = model.predict(x_test)
        test_f1[name] = float(f1_score(y_test, predicted))
        joblib.dump(model, MODELS / f"{name}.joblib")
        logger.info("%s held-out F1 %.4f", MODEL_DISPLAY[name], test_f1[name])

    best_name = max(test_f1, key=test_f1.get)
    best_model = fitted[best_name]
    joblib.dump(best_model, BEST_MODEL_PATH)
    payload = {
        "best_model": best_name,
        "best_model_display": MODEL_DISPLAY[best_name],
        "selection_metric": "held-out test F1",
        "test_f1": test_f1,
        "cv_f1": cv_scores,
        "params": params,
        "seed": seed,
        "caveat": (
            "The winning model was chosen on the same test split used for the "
            "headline F1, so that single number is mildly optimistic. Multi-seed "
            "means in metrics.json are the uncertainty check."
        ),
    }
    BEST_PARAMS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Best model by test F1: %s (%.4f)", MODEL_DISPLAY[best_name], test_f1[best_name])

    probabilities = best_model.predict_proba(scaled)[:, 1]
    predictions = frame[
        [
            "run_id",
            "project_id",
            "project_name",
            "team_type",
            "developer_id",
            "platform",
            "trigger",
            "branch",
            "timestamp",
            *MODEL_FEATURES,
            *["build_compilation", "unit_testing", "integration_testing", "deployment"],
            "total_execution_time_sec",
        ]
    ].copy()
    predictions["true_label"] = y
    predictions["bottleneck_probability"] = probabilities
    predictions["predicted_label"] = (probabilities >= PREDICT_THRESHOLD).astype(int)
    predictions["risk_band"] = [risk_band(float(p)) for p in probabilities]
    predictions.to_csv(PREDICTIONS_PATH, index=False)
    logger.info("Wrote %s for %d runs.", PREDICTIONS_PATH, len(predictions))
    return payload


def main() -> None:
    train(SEED)


if __name__ == "__main__":
    main()
