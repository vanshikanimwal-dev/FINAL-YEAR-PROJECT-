"""Held-out metrics, curves, and a five-seed repeat of the experiment."""

from __future__ import annotations

import json
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import MinMaxScaler

from config import (
    CLEAN_RUNS,
    FIGURES,
    FIGURE_DPI,
    METRICS_PATH,
    MODEL_COMPARISON_PATH,
    MODEL_DISPLAY,
    MODEL_FEATURES,
    MODELS,
    MULTI_SEEDS,
    PAPER_METRICS_PCT,
    SEED,
    get_logger,
)
from src.features import make_split
from src.train import HYBRID_NAME, fit_hybrid, tune_model

logger = get_logger("evaluate")

METRIC_NAMES = ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")


def _save(fig: plt.Figure, name: str) -> None:
    path = FIGURES / name
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Wrote %s", path)


def _downsample(values: np.ndarray, limit: int = 80) -> list[float]:
    array = np.asarray(values, dtype=float)
    if len(array) <= limit:
        return [float(v) for v in array]
    index = np.linspace(0, len(array) - 1, limit).astype(int)
    return [float(v) for v in array[index]]


def score_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, Any]:
    """Point metrics plus curves for one fitted model on one test fold."""
    precision_c, recall_c, _ = precision_recall_curve(y_true, y_prob)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=8, strategy="quantile")
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "fpr": _downsample(fpr),
        "tpr": _downsample(tpr),
        "precision_curve": _downsample(precision_c),
        "recall_curve": _downsample(recall_c),
        "calibration_fraction_positive": [float(v) for v in prob_true],
        "calibration_mean_predicted": [float(v) for v in prob_pred],
    }


def _matrices_for_seed(frame: pd.DataFrame, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Refit scaling inside the seed so other splits do not reuse seed-42 bounds."""
    split = make_split(frame, seed)
    train_idx = np.array(split["train_indices"])
    test_idx = np.array(split["test_indices"])
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(frame.iloc[train_idx][MODEL_FEATURES])
    x_test = scaler.transform(frame.iloc[test_idx][MODEL_FEATURES])
    y_train = frame.iloc[train_idx]["bottleneck"].to_numpy(dtype=int)
    y_test = frame.iloc[test_idx]["bottleneck"].to_numpy(dtype=int)
    return x_train, y_train, x_test, y_test


def _score_model(model: Any, x_test: np.ndarray, y_test: np.ndarray) -> dict[str, Any]:
    prob = model.predict_proba(x_test)[:, 1]
    pred = model.predict(x_test)
    return score_predictions(y_test, pred, prob)


def _repeat_seeds(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Full grid search on every seed except 42, which is scored from disk."""
    from config import PARAM_GRIDS

    per_seed: dict[str, list[dict[str, float]]] = {name: [] for name in MODEL_DISPLAY}
    for seed in MULTI_SEEDS:
        logger.info("Multi-seed experiment, seed %d", seed)
        if seed == SEED:
            from src.train import load_matrices

            _frame, scaled, y, _train_idx, test_idx, _scaler = load_matrices(SEED)
            x_test_ref = scaled[test_idx]
            y_test_ref = y[test_idx]
            for name in MODEL_DISPLAY:
                model = joblib.load(MODELS / f"{name}.joblib")
                metrics = _score_model(model, x_test_ref, y_test_ref)
                per_seed[name].append({metric: float(metrics[metric]) for metric in METRIC_NAMES})
                logger.info("seed %d %s F1 %.4f", seed, name, metrics["f1"])
            continue

        x_train, y_train, x_test, y_test = _matrices_for_seed(frame, seed)
        fitted: dict[str, Any] = {}
        params: dict[str, Any] = {}
        for name in PARAM_GRIDS:
            model, best_params, _cv = tune_model(name, x_train, y_train, seed)
            fitted[name] = model
            params[name] = best_params
            metrics = _score_model(model, x_test, y_test)
            per_seed[name].append({metric: float(metrics[metric]) for metric in METRIC_NAMES})
            logger.info("seed %d %s F1 %.4f", seed, name, metrics["f1"])
        hybrid = fit_hybrid(fitted, x_train, y_train, seed, params)
        metrics = _score_model(hybrid, x_test, y_test)
        per_seed[HYBRID_NAME].append({metric: float(metrics[metric]) for metric in METRIC_NAMES})
        logger.info("seed %d %s F1 %.4f", seed, HYBRID_NAME, metrics["f1"])

    summary: dict[str, dict[str, float]] = {}
    for name, rows in per_seed.items():
        block: dict[str, float] = {}
        for metric in METRIC_NAMES:
            values = np.array([row[metric] for row in rows], dtype=float)
            block[f"{metric}_mean"] = float(values.mean())
            block[f"{metric}_std"] = float(values.std(ddof=1))
        summary[name] = block
    return summary


def _plot_comparison(models: dict[str, dict[str, Any]]) -> None:
    sns.set_theme(style="whitegrid", font="DejaVu Sans")
    metrics = ["accuracy", "precision", "recall", "f1"]
    names = list(MODEL_DISPLAY)
    x = np.arange(len(metrics))
    width = 0.15
    fig, ax = plt.subplots(figsize=(11, 5.2))
    colors = ["#0F766E", "#4F46E5", "#D97706", "#DC2626", "#0891B2"]
    center = (len(names) - 1) / 2
    for i, name in enumerate(names):
        vals = [models[name][metric] * 100 for metric in metrics]
        ax.bar(x + (i - center) * width, vals, width, label=MODEL_DISPLAY[name], color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1"])
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Held-out test metrics (seed 42)")
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, "model_comparison.png")


def _plot_roc(models: dict[str, dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6))
    for name, payload in models.items():
        ax.plot(payload["fpr"], payload["tpr"], label=f"{MODEL_DISPLAY[name]} (AUC {payload['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#94A3B8", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves on the held-out test set")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    _save(fig, "roc_curves.png")


def _plot_pr(models: dict[str, dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6))
    for name, payload in models.items():
        ax.plot(
            payload["recall_curve"],
            payload["precision_curve"],
            label=f"{MODEL_DISPLAY[name]} (AP {payload['pr_auc']:.3f})",
        )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curves on the held-out test set")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    _save(fig, "pr_curves.png")


def _plot_confusion(models: dict[str, dict[str, Any]]) -> None:
    names = list(MODEL_DISPLAY)
    columns = 3
    rows = int(np.ceil(len(names) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 3.5 * rows))
    flat = np.atleast_1d(axes).ravel()
    for ax, name in zip(flat, names):
        matrix = np.array(models[name]["confusion_matrix"])
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(MODEL_DISPLAY[name])
        ax.set_xticklabels(["Normal", "Bottleneck"])
        ax.set_yticklabels(["Normal", "Bottleneck"], rotation=0)
    for ax in flat[len(names):]:
        ax.axis("off")
    fig.suptitle("Confusion matrices (test set)")
    fig.tight_layout()
    _save(fig, "confusion_matrices.png")


def _plot_calibration(models: dict[str, dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#94A3B8", linewidth=1)
    for name, payload in models.items():
        ax.plot(
            payload["calibration_mean_predicted"],
            payload["calibration_fraction_positive"],
            marker="o",
            label=MODEL_DISPLAY[name],
        )
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of bottlenecks")
    ax.set_title("Calibration curves")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    _save(fig, "calibration_curves.png")


def _plot_paper(models: dict[str, dict[str, Any]]) -> None:
    names = [name for name in MODEL_DISPLAY if name in PAPER_METRICS_PCT]
    ours = [models[name]["f1"] * 100 for name in names]
    paper = [PAPER_METRICS_PCT[name]["f1"] for name in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.bar(x - 0.18, paper, 0.36, label="Paper F1", color="#94A3B8")
    ax.bar(x + 0.18, ours, 0.36, label="This study F1", color="#0F766E")
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_DISPLAY[name] for name in names], rotation=15)
    ax.set_ylabel("F1 (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Paper vs this study (reference only, not a tuning target)")
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, "paper_vs_ours.png")


def evaluate() -> dict[str, Any]:
    frame = pd.read_csv(CLEAN_RUNS)
    from src.train import load_matrices

    _frame, scaled, y, _train_idx, test_idx, _scaler = load_matrices(SEED)
    y_test = y[test_idx]
    x_test = scaled[test_idx]
    detailed: dict[str, Any] = {}
    for name in MODEL_DISPLAY:
        model = joblib.load(MODELS / f"{name}.joblib")
        prob = model.predict_proba(x_test)[:, 1]
        pred = model.predict(x_test)
        detailed[name] = score_predictions(y_test, pred, prob)
        detailed[name]["display_name"] = MODEL_DISPLAY[name]
        logger.info(
            "%s acc %.3f prec %.3f rec %.3f f1 %.3f auc %.3f",
            name,
            detailed[name]["accuracy"],
            detailed[name]["precision"],
            detailed[name]["recall"],
            detailed[name]["f1"],
            detailed[name]["roc_auc"],
        )

    multi = _repeat_seeds(frame)
    best_name = max(detailed, key=lambda name: detailed[name]["f1"])
    best_params = json.loads((MODELS / "best_params.json").read_text(encoding="utf-8")) if (MODELS / "best_params.json").exists() else {}
    deployed_name = best_params.get("deployed_model", best_name)
    if deployed_name not in detailed:
        deployed_name = best_name

    sns.set_theme(style="whitegrid", font="DejaVu Sans")
    _plot_comparison(detailed)
    _plot_roc(detailed)
    _plot_pr(detailed)
    _plot_confusion(detailed)
    _plot_calibration(detailed)
    _plot_paper(detailed)

    rows = []
    for name in MODEL_DISPLAY:
        row: dict[str, Any] = {"model": name, "display_name": MODEL_DISPLAY[name]}
        for metric in METRIC_NAMES:
            row[metric] = detailed[name][metric]
            row[f"{metric}_mean"] = multi[name][f"{metric}_mean"]
            row[f"{metric}_std"] = multi[name][f"{metric}_std"]
        if name in PAPER_METRICS_PCT:
            for metric, value in PAPER_METRICS_PCT[name].items():
                row[f"paper_{metric}"] = value
        rows.append(row)
    pd.DataFrame(rows).to_csv(MODEL_COMPARISON_PATH, index=False)

    # Drop bulky curves from the logged preview but keep them in metrics.json
    # so the dashboard can redraw ROC / PR / calibration without recomputing.
    payload = {
        "primary_seed": SEED,
        "seeds": list(MULTI_SEEDS),
        "n_test": int(len(y_test)),
        "best_model": best_name,
        "best_model_display": MODEL_DISPLAY[best_name],
        "best_f1": detailed[best_name]["f1"],
        "deployed_model": deployed_name,
        "deployed_model_display": MODEL_DISPLAY[deployed_name],
        "deployed_f1": detailed[deployed_name]["f1"],
        "models": detailed,
        "multi_seed": multi,
        "paper_percent": PAPER_METRICS_PCT,
        "tuning": best_params,
        "notes": [
            "Primary ranking metric is positive-class F1.",
            "The deployed predictor is a stack: logistic regression blends out-of-fold scores from logistic regression, RBF SVM, gradient boosting, and random forest.",
            "Seed 42 metrics describe the saved models. The other four seeds repeat the full grid search and the stack on their own stratified splits and scalers.",
            "Paper percentages are a reference, not a target. Differences are expected because the data are simulated.",
        ],
    }
    METRICS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Wrote %s and %s", METRICS_PATH, MODEL_COMPARISON_PATH)
    return payload


def main() -> None:
    evaluate()


if __name__ == "__main__":
    main()
