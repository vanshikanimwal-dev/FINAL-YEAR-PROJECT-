"""Feature-processing layer: parse, clock-normalise, dedupe, impute, scale.

Scaling is fit on the training split and stored in ``models/scaler.joblib``.
``runs_clean.csv`` keeps imputed values in original units so the dashboard
can talk about lines of code and hours. That split of responsibilities is
deliberate: the model consumes scaled inputs, the narrative stays readable.
"""

from __future__ import annotations

import json
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from config import (
    CLEANING_REPORT_PATH,
    CLEAN_RUNS,
    DATA_PROCESSED,
    FEATURE_ORDER_PATH,
    IMPUTE_COLUMNS,
    IMPUTE_MEDIANS_PATH,
    MODEL_FEATURES,
    PRE_IMPUTE_RUNS,
    RAW_RUNS,
    SCALER_PATH,
    get_logger,
)
from src.features import make_split, save_split

logger = get_logger("preprocess")

_LEVEL = re.compile(r"level=([A-Z]+)")
_STAGE = re.compile(r"stage=([a-z_]+)")
_EXIT = re.compile(r"exit=(\d+)")
_EPOCH = re.compile(r"^\d{9,12}$")


def parse_log(text: object) -> tuple[str | None, str | None, float | None]:
    """Pull level, stage, and exit code out of a free-text log line."""
    raw = "" if text is None or (isinstance(text, float) and np.isnan(text)) else str(text)
    level = _LEVEL.search(raw)
    stage = _STAGE.search(raw)
    exit_code = _EXIT.search(raw)
    return (
        level.group(1) if level else None,
        stage.group(1) if stage else None,
        float(exit_code.group(1)) if exit_code else None,
    )


def parse_timestamp(value: object) -> pd.Timestamp:
    """Normalise the four messy timestamp encodings to UTC."""
    text = str(value).strip()
    if _EPOCH.fullmatch(text):
        return pd.to_datetime(int(text), unit="s", utc=True)
    if text.endswith(" UTC"):
        text = text[:-4]
    parsed = pd.to_datetime(text, utc=True)
    if parsed.tzinfo is None:
        parsed = parsed.tz_localize("UTC")
    return parsed


def preprocess() -> pd.DataFrame:
    raw = pd.read_csv(RAW_RUNS)
    n_raw = len(raw)
    parsed_logs = raw["log_text"].map(parse_log)
    raw["log_level"] = [item[0] for item in parsed_logs]
    raw["log_stage"] = [item[1] for item in parsed_logs]
    raw["log_exit_code"] = [item[2] for item in parsed_logs]
    logger.info(
        "Parsed logs: %d/%d rows yielded a level.",
        int(raw["log_level"].notna().sum()),
        n_raw,
    )

    timestamps = raw["timestamp"].map(parse_timestamp)
    raw["timestamp"] = pd.DatetimeIndex(timestamps)
    n_before = len(raw)
    clean = raw.drop_duplicates(subset=["run_id"], keep="first").copy()
    n_dupes_removed = n_before - len(clean)
    clean = clean.sort_values(["project_id", "timestamp"]).reset_index(drop=True)
    clean["timestamp"] = pd.DatetimeIndex(clean["timestamp"]).tz_convert("UTC")
    logger.info("Removed %d webhook-retry duplicates. %d unique runs remain.", n_dupes_removed, len(clean))

    pre_impute = clean.copy()
    pre_impute["timestamp"] = pre_impute["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    pre_impute.to_csv(PRE_IMPUTE_RUNS, index=False)

    split = make_split(clean)
    save_split(split)
    train_idx = split["train_indices"]

    medians: dict[str, float] = {}
    imputed_cells = 0
    for column in IMPUTE_COLUMNS:
        observed = clean.iloc[train_idx][column].dropna()
        if observed.empty:
            raise RuntimeError(f"Training split has no observed values for {column}.")
        median = float(observed.median())
        medians[column] = median
        missing = clean[column].isna()
        imputed_cells += int(missing.sum())
        clean.loc[missing, column] = median
    IMPUTE_MEDIANS_PATH.write_text(json.dumps(medians, indent=2), encoding="utf-8")
    logger.info("Imputed %d cells using training-split medians.", imputed_cells)

    scaler = MinMaxScaler()
    scaler.fit(clean.iloc[train_idx][MODEL_FEATURES])
    joblib.dump(scaler, SCALER_PATH)
    FEATURE_ORDER_PATH.write_text(json.dumps(MODEL_FEATURES, indent=2), encoding="utf-8")
    logger.info("Fit MinMaxScaler on %d training rows. Wrote %s", len(train_idx), SCALER_PATH)

    clean["timestamp"] = pd.DatetimeIndex(clean["timestamp"]).strftime("%Y-%m-%dT%H:%M:%SZ")
    clean.to_csv(CLEAN_RUNS, index=False)
    try:
        clean.to_parquet(DATA_PROCESSED / "runs_clean.parquet", index=False)
    except Exception as exc:  # pragma: no cover - parquet is a convenience copy
        logger.warning("Parquet copy skipped: %s", exc)

    report = {
        "raw_rows": n_raw,
        "rows_removed_as_duplicates": n_dupes_removed,
        "rows_after_dedup": len(clean),
        "cells_imputed": imputed_cells,
        "imputation": "per-column median fit on the training split only",
        "scaling": "MinMaxScaler fit on the training split only; runs_clean.csv stays in original units",
        "timestamps": "normalised to UTC",
        "medians": medians,
    }
    CLEANING_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Cleaning report: removed %d rows, imputed %d cells.", n_dupes_removed, imputed_cells)
    return clean


def main() -> None:
    preprocess()


if __name__ == "__main__":
    main()
