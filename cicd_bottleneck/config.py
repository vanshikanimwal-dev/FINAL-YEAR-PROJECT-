"""Project-wide constants for the CI/CD bottleneck study.

Assumptions live here so every script and the dashboard share one source of
truth: seed, paths, feature lists, model grids, and detection thresholds.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED: int = 42
MULTI_SEEDS: tuple[int, ...] = (42, 7, 13, 21, 99)
N_JOBS: int = -1
CV_FOLDS: int = 5
FIGURE_DPI: int = 200

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT: Path = Path(__file__).resolve().parent
DATA_RAW: Path = ROOT / "data" / "raw"
DATA_PROCESSED: Path = ROOT / "data" / "processed"
MODELS: Path = ROOT / "models"
REPORTS: Path = ROOT / "reports"
FIGURES: Path = REPORTS / "figures"

for _path in (DATA_RAW, DATA_PROCESSED, MODELS, REPORTS, FIGURES):
    _path.mkdir(parents=True, exist_ok=True)

RAW_RUNS: Path = DATA_RAW / "runs_raw.csv"
CLEAN_RUNS: Path = DATA_PROCESSED / "runs_clean.csv"
PRE_IMPUTE_RUNS: Path = DATA_PROCESSED / "runs_pre_impute.csv"
SPLIT_PATH: Path = DATA_PROCESSED / "split.json"
IMPUTE_MEDIANS_PATH: Path = DATA_PROCESSED / "imputation_medians.json"
PREDICTIONS_PATH: Path = DATA_PROCESSED / "predictions.csv"
ALERTS_PATH: Path = DATA_PROCESSED / "alerts.csv"

SCALER_PATH: Path = MODELS / "scaler.joblib"
BEST_MODEL_PATH: Path = MODELS / "best_model.joblib"
FEATURE_ORDER_PATH: Path = MODELS / "feature_order.json"
BEST_PARAMS_PATH: Path = MODELS / "best_params.json"

METRICS_PATH: Path = REPORTS / "metrics.json"
MODEL_COMPARISON_PATH: Path = REPORTS / "model_comparison.csv"
FEATURE_IMPORTANCE_PATH: Path = REPORTS / "feature_importance.csv"
ABLATION_PATH: Path = REPORTS / "ablation.csv"
DETECTION_REPORT_PATH: Path = REPORTS / "detection_report.json"
FEATURE_VALIDATION_PATH: Path = REPORTS / "feature_validation.json"
CLEANING_REPORT_PATH: Path = REPORTS / "cleaning_report.json"
SUMMARY_PATH: Path = REPORTS / "RESULTS_SUMMARY.md"

# ---------------------------------------------------------------------------
# Dataset design
# ---------------------------------------------------------------------------
N_RUNS: int = 4800
TARGET_BOTTLENECK_RATE: float = 0.35
BOTTLENECK_RATE_TOLERANCE: float = 0.015
LABEL_MEDIAN_MULTIPLIER: float = 1.5
DUPLICATE_FRACTION: float = 0.02
MISSING_CELL_FRACTION: float = 0.03
# Fixed signal-to-noise ratio. The residual scale is what bisection calibrates
# so the positive class lands near 35%. SNR is a modelling assumption, not a
# fit to the paper's published accuracy table.
SIGNAL_TO_NOISE: float = 2.2
TRAIN_FRACTION: float = 0.80
TEST_SIZE: int = 960
TRAIN_SIZE: int = 3840

# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
BEHAVIORAL_FEATURES: list[str] = [
    "commit_frequency",
    "code_churn",
    "pr_review_latency_hours",
    "build_success_ratio",
    "historical_build_success_ratio",
]
TIMING_FEATURES: list[str] = [
    "test_execution_duration_sec",
    "build_duration_sec",
]
PROCESS_FEATURES: list[str] = [
    "deployment_delay_min",
]
CORE_FEATURES: list[str] = [
    "commit_frequency",
    "code_churn",
    "pr_review_latency_hours",
    "build_success_ratio",
    "test_execution_duration_sec",
    "deployment_delay_min",
    "historical_build_success_ratio",
]
MODEL_FEATURES: list[str] = CORE_FEATURES + ["build_duration_sec"]

FEATURE_LABELS: dict[str, str] = {
    "commit_frequency": "Commit frequency",
    "code_churn": "Code churn",
    "pr_review_latency_hours": "PR review latency",
    "build_success_ratio": "Recent build success",
    "test_execution_duration_sec": "Test duration",
    "deployment_delay_min": "Deployment delay",
    "historical_build_success_ratio": "Historical build success",
    "build_duration_sec": "Build duration",
}

FEATURE_GROUPS: dict[str, str] = {
    "commit_frequency": "behavioral",
    "code_churn": "behavioral",
    "pr_review_latency_hours": "behavioral",
    "build_success_ratio": "behavioral",
    "historical_build_success_ratio": "behavioral",
    "test_execution_duration_sec": "timing",
    "build_duration_sec": "timing",
    "deployment_delay_min": "deployment_delay",
}

STAGE_COLUMNS: list[str] = [
    "build_compilation",
    "unit_testing",
    "integration_testing",
    "deployment",
]
STAGE_LABELS: dict[str, str] = {
    "build_compilation": "Build compilation",
    "unit_testing": "Unit testing",
    "integration_testing": "Integration testing",
    "deployment": "Deployment",
}

IMPUTE_COLUMNS: list[str] = MODEL_FEATURES + STAGE_COLUMNS

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
MODEL_DISPLAY: dict[str, str] = {
    "logistic_regression": "Logistic Regression",
    "svm_rbf": "SVM (RBF)",
    "gradient_boosting": "Gradient Boosting",
    "random_forest": "Random Forest",
}

PARAM_GRIDS: dict[str, dict[str, list]] = {
    "logistic_regression": {"C": [0.1, 1.0, 10.0]},
    "svm_rbf": {"C": [1.0, 10.0], "gamma": ["scale"]},
    "gradient_boosting": {
        "n_estimators": [160],
        "learning_rate": [0.06, 0.12],
        "max_depth": [2, 3],
    },
    "random_forest": {
        "n_estimators": [280],
        "max_depth": [8, 16],
        "min_samples_leaf": [2, 5],
    },
}

# Paper table, percent. Reference only — never a training target.
PAPER_METRICS_PCT: dict[str, dict[str, float]] = {
    "logistic_regression": {"accuracy": 81.4, "precision": 79.6, "recall": 77.3, "f1": 78.4},
    "svm_rbf": {"accuracy": 85.7, "precision": 84.2, "recall": 82.1, "f1": 83.1},
    "gradient_boosting": {"accuracy": 92.6, "precision": 91.9, "recall": 91.4, "f1": 91.7},
    "random_forest": {"accuracy": 94.2, "precision": 93.8, "recall": 92.6, "f1": 93.2},
}

RISK_GREEN_MAX: float = 0.40
RISK_AMBER_MAX: float = 0.65
PREDICT_THRESHOLD: float = 0.50

# ---------------------------------------------------------------------------
# Detection layer
# ---------------------------------------------------------------------------
DETECT_WINDOW: int = 20
DETECT_MIN_HISTORY: int = 10
DETECT_Z_THRESHOLD: float = 2.0
DETECT_PERSISTENCE: int = 3
DETECT_TAU: float = 0.65
DETECT_TAU_GRID: tuple[float, ...] = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90)
FALSE_ALARM_TARGET: float = 0.08
RFE_N_FEATURES: int = 5
CORR_FLAG: float = 0.80

# ---------------------------------------------------------------------------
# Project catalog. Counts sum to exactly 4,800.
# ---------------------------------------------------------------------------
PROJECT_SPECS: list[dict[str, object]] = [
    {"project_id": "proj_01", "project_name": "Atlas", "team_type": "single", "n_runs": 200, "n_devs": 6, "platform": "Jenkins"},
    {"project_id": "proj_02", "project_name": "Beacon", "team_type": "single", "n_runs": 220, "n_devs": 7, "platform": "GitHub Actions"},
    {"project_id": "proj_03", "project_name": "Cobalt", "team_type": "single", "n_runs": 240, "n_devs": 8, "platform": "GitLab CI"},
    {"project_id": "proj_04", "project_name": "Drift", "team_type": "single", "n_runs": 250, "n_devs": 5, "platform": "Jenkins"},
    {"project_id": "proj_05", "project_name": "Ember", "team_type": "single", "n_runs": 260, "n_devs": 9, "platform": "GitHub Actions"},
    {"project_id": "proj_06", "project_name": "Forge", "team_type": "single", "n_runs": 270, "n_devs": 8, "platform": "GitLab CI"},
    {"project_id": "proj_07", "project_name": "Grove", "team_type": "single", "n_runs": 280, "n_devs": 10, "platform": "Jenkins"},
    {"project_id": "proj_08", "project_name": "Harbor", "team_type": "single", "n_runs": 280, "n_devs": 12, "platform": "GitHub Actions"},
    {"project_id": "proj_09", "project_name": "Ion", "team_type": "multi", "n_runs": 650, "n_devs": 16, "platform": "Jenkins"},
    {"project_id": "proj_10", "project_name": "Jade", "team_type": "multi", "n_runs": 700, "n_devs": 18, "platform": "GitHub Actions"},
    {"project_id": "proj_11", "project_name": "Kite", "team_type": "multi", "n_runs": 720, "n_devs": 20, "platform": "GitLab CI"},
    {"project_id": "proj_12", "project_name": "Lumen", "team_type": "multi", "n_runs": 730, "n_devs": 25, "platform": "GitLab CI"},
]

PLATFORMS: tuple[str, ...] = ("Jenkins", "GitHub Actions", "GitLab CI")
TRIGGERS: tuple[str, ...] = ("commit", "pull_request")
BRANCHES: tuple[str, ...] = ("main", "develop", "feature")
PLATFORM_COMPILE_MULT: dict[str, float] = {
    "Jenkins": 1.14,
    "GitHub Actions": 0.92,
    "GitLab CI": 1.00,
}


def get_logger(name: str) -> logging.Logger:
    """Return a stream logger configured once per name."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger
