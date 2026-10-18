"""Importances, behavioral-versus-infrastructure share, and a leakage ablation."""

from __future__ import annotations

import json
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler

from config import (
    ABLATION_PATH,
    BEHAVIORAL_FEATURES,
    BEST_PARAMS_PATH,
    CLEAN_RUNS,
    FEATURE_GROUPS,
    FEATURE_IMPORTANCE_PATH,
    FEATURE_LABELS,
    FIGURES,
    FIGURE_DPI,
    MODEL_FEATURES,
    MODELS,
    SEED,
    TIMING_FEATURES,
    get_logger,
)
from src.features import load_split
from src.train import build_estimator

logger = get_logger("explain")


def _save(fig: plt.Figure, name: str) -> None:
    path = FIGURES / name
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Wrote %s", path)


def _percent(values: np.ndarray) -> np.ndarray:
    """Turn a non-negative importance vector into percentages that sum to 100."""
    cleaned = np.clip(np.asarray(values, dtype=float), 0, None)
    if float(cleaned.sum()) <= 0:
        cleaned = np.abs(np.asarray(values, dtype=float))
    total = float(cleaned.sum())
    if total <= 0:
        return np.zeros_like(cleaned)
    return 100.0 * cleaned / total


def _shap_values(model: Any, x_test: np.ndarray) -> np.ndarray | None:
    try:
        import shap
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.warning("SHAP unavailable: %s", exc)
        return None
    try:
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(x_test)
        if isinstance(values, list):
            values = values[1] if len(values) > 1 else values[0]
        array = np.asarray(values)
        if array.ndim == 3:
            array = array[:, :, 1] if array.shape[2] > 1 else array[:, :, 0]
        return np.abs(array).mean(axis=0)
    except Exception as exc:
        logger.warning("SHAP failed, continuing without it: %s", exc)
        return None


def _refit(name: str, params: dict[str, Any], x_train: np.ndarray, y_train: np.ndarray) -> Any:
    model = build_estimator(name, SEED, probability=True)
    model.set_params(**params)
    model.fit(x_train, y_train)
    return model


def explain() -> pd.DataFrame:
    frame = pd.read_csv(CLEAN_RUNS)
    split = load_split()
    train_idx = np.array(split["train_indices"])
    test_idx = np.array(split["test_indices"])
    y_train = frame.iloc[train_idx]["bottleneck"].to_numpy(dtype=int)
    y_test = frame.iloc[test_idx]["bottleneck"].to_numpy(dtype=int)

    scaler = joblib.load(MODELS / "scaler.joblib")
    x_test = scaler.transform(frame.iloc[test_idx][MODEL_FEATURES])
    forest = joblib.load(MODELS / "random_forest.joblib")
    best_meta = json.loads(BEST_PARAMS_PATH.read_text(encoding="utf-8"))
    best_name = best_meta["best_model"]
    best_model = joblib.load(MODELS / "best_model.joblib")

    mdi = _percent(np.asarray(forest.feature_importances_, dtype=float))
    perm = permutation_importance(
        best_model,
        x_test,
        y_test,
        n_repeats=5,
        random_state=SEED,
        scoring="f1",
        n_jobs=-1,
    )
    perm_pct = _percent(perm.importances_mean)
    shap_mean = _shap_values(forest, x_test)
    shap_pct = _percent(shap_mean) if shap_mean is not None else np.full(len(MODEL_FEATURES), np.nan)

    table = pd.DataFrame(
        {
            "feature": MODEL_FEATURES,
            "label": [FEATURE_LABELS[name] for name in MODEL_FEATURES],
            "group": [FEATURE_GROUPS[name] for name in MODEL_FEATURES],
            "mdi": np.asarray(forest.feature_importances_, dtype=float),
            "mdi_pct": mdi,
            "permutation_mean": perm.importances_mean,
            "permutation_std": perm.importances_std,
            "permutation_pct": perm_pct,
            "shap_mean_abs": shap_mean if shap_mean is not None else np.nan,
            "shap_pct": shap_pct,
        }
    )
    table.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
    logger.info("Wrote %s", FEATURE_IMPORTANCE_PATH)
    for _, row in table.sort_values("mdi_pct", ascending=False).iterrows():
        logger.info("MDI %-32s %5.1f%%", row["feature"], row["mdi_pct"])

    shares = table.groupby("group")["mdi_pct"].sum().to_dict()
    logger.info(
        "MDI shares — behavioral %.1f%% | timing %.1f%% | deployment delay %.1f%%",
        shares.get("behavioral", 0.0),
        shares.get("timing", 0.0),
        shares.get("deployment_delay", 0.0),
    )

    subsets = {
        "all_features": MODEL_FEATURES,
        "no_timing": [feature for feature in MODEL_FEATURES if feature not in TIMING_FEATURES],
        "behavior_only": list(BEHAVIORAL_FEATURES),
    }
    ablation_rows = []
    full_f1 = None
    params = best_meta["params"][best_name]
    for setting, columns in subsets.items():
        subset_scaler = MinMaxScaler()
        x_sub_train = subset_scaler.fit_transform(frame.iloc[train_idx][columns])
        x_sub_test = subset_scaler.transform(frame.iloc[test_idx][columns])
        model = _refit(best_name, params, x_sub_train, y_train)
        f1 = float(f1_score(y_test, model.predict(x_sub_test)))
        if setting == "all_features":
            full_f1 = f1
        ablation_rows.append(
            {
                "setting": setting,
                "model": best_name,
                "n_features": len(columns),
                "features": ",".join(columns),
                "f1": f1,
            }
        )
        logger.info("Ablation %-16s F1 %.4f", setting, f1)
    ablation = pd.DataFrame(ablation_rows)
    ablation["delta_f1_vs_all"] = ablation["f1"] - float(full_f1)
    ablation.to_csv(ABLATION_PATH, index=False)

    sns.set_theme(style="whitegrid", font="DejaVu Sans")
    melted = table.melt(
        id_vars=["label"],
        value_vars=[column for column in ("mdi_pct", "permutation_pct", "shap_pct") if table[column].notna().any()],
        var_name="method",
        value_name="percent",
    )
    melted["method"] = melted["method"].map(
        {"mdi_pct": "MDI", "permutation_pct": "Permutation", "shap_pct": "SHAP"}
    )
    fig, ax = plt.subplots(figsize=(11, 5.5))
    sns.barplot(data=melted, x="label", y="percent", hue="method", ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Share of importance (%)")
    ax.set_title("Feature importance: MDI vs permutation vs SHAP")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(frameon=False, title="")
    fig.tight_layout()
    _save(fig, "feature_importance.png")

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    sns.barplot(data=ablation, x="setting", y="f1", color="#0F766E", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Test F1")
    ax.set_xlabel("")
    ax.set_title(f"Leakage ablation ({best_meta['best_model_display']})")
    fig.tight_layout()
    _save(fig, "ablation.png")
    return table


def main() -> None:
    explain()


if __name__ == "__main__":
    main()
