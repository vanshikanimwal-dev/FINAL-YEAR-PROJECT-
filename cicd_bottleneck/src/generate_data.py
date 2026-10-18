"""Simulate CI/CD runs whose bottlenecks come from developer behavior.

There is no public dataset that links developer behavior to run-level
bottleneck labels, so this module draws 4,800 runs for 12 projects from
explicit distributions and a non-linear structural model.

Distributions (documented assumption):
- commit_frequency: Poisson. A 7-day count is a non-negative integer.
- code_churn: log-normal. Line changes are heavy-tailed.
- pr_review_latency_hours: log-normal. Reviews wait, then land in a burst.
- build_success_ratio: Binomial(10, p) / 10. It is a fraction of the last
  10 builds, so it lives on a 0.1 grid.
- deployment_delay_min: Gamma. Delays are positive and right-skewed.
- historical_build_success_ratio: Beta latent mean per developer, plus a
  little run-level noise, clipped to (0, 1).
- stage times: log-normal multipliers on a project baseline.

Causal story: a within-project behavior score combines main effects with a
three-way interaction (high churn is costly mainly when review latency is
high and historical success is low). That score stretches the *deployment*
stage. Compile and test times mostly reflect platform and mild churn, so a
model cannot recover the label by adding the timing features. An AR(1)
session shock makes the same developer look consistently risky for a few
consecutive runs, which is what the persistence rule later looks for.

The label is retrospective: bottleneck = 1 when total time exceeds 1.5 times
that project's own median. A bisection on the residual scale, at a fixed
signal-to-noise ratio, targets a 35% positive rate.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import bisect
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

import config
from config import (
    BOTTLENECK_RATE_TOLERANCE,
    DUPLICATE_FRACTION,
    LABEL_MEDIAN_MULTIPLIER,
    MISSING_CELL_FRACTION,
    MODEL_FEATURES,
    N_RUNS,
    PLATFORM_COMPILE_MULT,
    PROJECT_SPECS,
    RAW_RUNS,
    SEED,
    SIGNAL_TO_NOISE,
    STAGE_COLUMNS,
    TARGET_BOTTLENECK_RATE,
    get_logger,
)

logger = get_logger("generate_data")

BEHAVIOR_ONLY = [
    "commit_frequency",
    "code_churn",
    "pr_review_latency_hours",
    "build_success_ratio",
    "deployment_delay_min",
    "historical_build_success_ratio",
]


def assign_bottleneck_labels(frame: pd.DataFrame) -> pd.Series:
    """Apply the paper's 1.5× project-median rule. Ties on the boundary stay 0."""
    medians = frame.groupby("project_id")["total_execution_time_sec"].transform("median")
    labels = (frame["total_execution_time_sec"] > LABEL_MEDIAN_MULTIPLIER * medians).astype(int)
    return labels


def _zscore(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    std = float(array.std())
    if std < 1e-8:
        return np.zeros_like(array)
    return (array - array.mean()) / std


def _timestamps(n: int, rng: np.random.Generator, origin_shift_days: int) -> list[pd.Timestamp]:
    """Business-hours-weighted timestamps across roughly six months."""
    start = pd.Timestamp("2025-01-06") + pd.Timedelta(days=int(origin_shift_days))
    weekdays: list[pd.Timestamp] = []
    cursor = start
    while len(weekdays) < 140:
        if cursor.weekday() < 5:
            weekdays.append(cursor)
        cursor += pd.Timedelta(days=1)
    chosen = rng.choice(len(weekdays), size=n, replace=True)
    stamps: list[pd.Timestamp] = []
    for index in chosen:
        day = weekdays[int(index)]
        if rng.random() < 0.07:
            day = day + pd.Timedelta(days=int(rng.integers(1, 3)))
        roll = float(rng.random())
        if roll < 0.78:
            hour = float(rng.uniform(9.0, 17.5))
        elif roll < 0.93:
            hour = float(rng.uniform(17.5, 20.5))
        else:
            hour = float(rng.uniform(8.0, 9.0))
        minute = float(rng.uniform(0.0, 60.0))
        stamps.append(day + pd.to_timedelta(hour, unit="h") + pd.to_timedelta(minute, unit="m"))
    stamps.sort()
    for i in range(1, len(stamps)):
        if stamps[i] <= stamps[i - 1]:
            stamps[i] = stamps[i - 1] + pd.Timedelta(minutes=1)
    return stamps


def _developer_roster(spec: dict[str, Any], rng: np.random.Generator, id_start: int) -> list[dict[str, Any]]:
    roster = []
    n_devs = int(spec["n_devs"])
    for offset in range(n_devs):
        discipline = float(rng.beta(2.2, 2.2))
        hist_mean = float(np.clip(0.48 + 0.44 * discipline, 0.08, 0.97))
        roster.append(
            {
                "developer_id": f"dev_{id_start + offset:03d}",
                "discipline": discipline,
                "hist_mean": hist_mean,
            }
        )
    return roster


def _assign_sessions(roster: list[dict[str, Any]], n: int, rng: np.random.Generator) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Sample developers in short sessions and attach a shared stress shock."""
    weights = np.array([0.55 + (1.0 - dev["discipline"]) for dev in roster], dtype=float)
    weights /= weights.sum()
    chosen: list[dict[str, Any]] = []
    stress = np.zeros(n, dtype=float)
    cursor = 0
    while cursor < n:
        dev = roster[int(rng.choice(len(roster), p=weights))]
        length = int(rng.integers(2, 6))
        shock = float(rng.normal(0.0, 1.0))
        end = min(n, cursor + length)
        for i in range(cursor, end):
            chosen.append(dev)
            stress[i] = shock
        cursor = end
    return chosen, stress


def _one_project(spec: dict[str, Any], rng: np.random.Generator, id_start: int, run_start: int) -> tuple[pd.DataFrame, int, int]:
    n = int(spec["n_runs"])
    roster = _developer_roster(spec, rng, id_start)
    developers, stress = _assign_sessions(roster, n, rng)
    stamps = _timestamps(n, rng, origin_shift_days=id_start % 11)
    primary = str(spec["platform"])
    others = [p for p in PLATFORM_COMPILE_MULT if p != primary]

    rows: list[dict[str, Any]] = []
    for i in range(n):
        dev = developers[i]
        discipline = float(dev["discipline"])
        shock = float(stress[i])
        platform = primary if rng.random() < 0.70 else str(rng.choice(others))
        trigger = "pull_request" if rng.random() < 0.62 else "commit"
        branch_roll = rng.random()
        if branch_roll < 0.58:
            branch = "feature"
        elif branch_roll < 0.82:
            branch = "develop"
        else:
            branch = "main"
        commit_frequency = int(rng.poisson(lam=1.6 + 6.2 * (1.0 - discipline) + (1.3 if spec["team_type"] == "multi" else 0.0)))
        commit_frequency = int(np.clip(commit_frequency + (2 if shock > 0.8 else 0), 0, 45))
        churn = float(rng.lognormal(mean=np.log(150) + 1.05 * (1.0 - discipline) + 0.28 * shock, sigma=0.82))
        churn = float(np.clip(churn, 1.0, 20000.0))
        latency_mu = np.log(5.0) + 1.20 * (1.0 - discipline) + 0.32 * max(shock, 0.0)
        if trigger == "pull_request":
            latency_mu += 0.45
        latency = float(np.clip(rng.lognormal(latency_mu, 0.70), 0.2, 240.0))
        success_p = float(np.clip(0.40 + 0.52 * discipline - 0.06 * max(shock, 0.0), 0.05, 0.98))
        build_success = float(rng.binomial(10, success_p) / 10.0)
        delay = float(rng.gamma(2.2, 5.2 * (1.20 - 0.65 * discipline)))
        delay = float(np.clip(delay * np.exp(0.18 * max(shock, 0.0)), 0.4, 480.0))
        historical = float(np.clip(rng.normal(dev["hist_mean"], 0.018), 0.02, 0.99))
        rows.append(
            {
                "project_id": spec["project_id"],
                "project_name": spec["project_name"],
                "team_type": spec["team_type"],
                "run_id": f"run_{run_start + i:06d}",
                "developer_id": dev["developer_id"],
                "platform": platform,
                "trigger": trigger,
                "branch": branch,
                "timestamp": stamps[i],
                "commit_frequency": commit_frequency,
                "code_churn": churn,
                "pr_review_latency_hours": latency,
                "build_success_ratio": build_success,
                "deployment_delay_min": delay,
                "historical_build_success_ratio": historical,
                "project_base": 720.0 if spec["team_type"] == "single" else 1500.0,
                "base_jitter": float(rng.uniform(0.85, 1.20)),
            }
        )
    frame = pd.DataFrame(rows)
    frame["zc"] = _zscore(np.log1p(frame["code_churn"].to_numpy()))
    frame["zr"] = _zscore(np.log1p(frame["pr_review_latency_hours"].to_numpy()))
    frame["zf"] = _zscore(frame["commit_frequency"].to_numpy(dtype=float))
    frame["zh"] = _zscore(frame["historical_build_success_ratio"].to_numpy())
    frame["zb"] = _zscore(frame["build_success_ratio"].to_numpy())
    frame["zd"] = _zscore(np.log1p(frame["deployment_delay_min"].to_numpy()))
    low_hist = _zscore(1.0 - frame["historical_build_success_ratio"].to_numpy())
    low_success = _zscore(1.0 - frame["build_success_ratio"].to_numpy())
    interaction = np.clip(frame["zc"].to_numpy(), 0, None) * np.clip(frame["zr"].to_numpy(), 0, None) * np.clip(low_hist, 0, None)
    interaction2 = np.clip(frame["zc"].to_numpy(), 0, None) * np.clip(frame["zd"].to_numpy(), 0, None) * np.clip(low_hist, 0, None)
    score = (
        0.28 * frame["zc"].to_numpy()
        + 0.24 * frame["zr"].to_numpy()
        + 0.08 * frame["zf"].to_numpy()
        - 0.18 * frame["zh"].to_numpy()
        - 0.10 * frame["zb"].to_numpy()
        + 0.12 * frame["zd"].to_numpy()
        + 1.05 * interaction
        + 0.40 * interaction2
    )
    frame["s"] = _zscore(score)
    frame["low_success_z"] = low_success
    eps = rng.normal(0.0, 1.0, size=n)
    ar = np.zeros(n, dtype=float)
    for t in range(n):
        ar[t] = eps[t] if t == 0 else 0.62 * ar[t - 1] + 0.38 * eps[t]
    frame["eps_ar"] = _zscore(ar)
    for stage_i in range(4):
        frame[f"e{stage_i}"] = rng.normal(0.0, 1.0, size=n)
    return frame, id_start + len(roster), run_start + n


def _materialize(frame: pd.DataFrame, noise_scale: float) -> pd.DataFrame:
    """Turn behavior scores into stage times. `noise_scale` is the bisected knob."""
    out = frame.copy()
    gain = SIGNAL_TO_NOISE * noise_scale
    s = out["s"].to_numpy(dtype=float)
    shared = np.exp(0.22 * gain * s)
    behavior = np.exp(gain * s + noise_scale * out["eps_ar"].to_numpy(dtype=float))
    base = out["project_base"].to_numpy(dtype=float) * out["base_jitter"].to_numpy(dtype=float)
    plat = out["platform"].map(PLATFORM_COMPILE_MULT).to_numpy(dtype=float)
    zc = out["zc"].to_numpy(dtype=float)
    low_success = out["low_success_z"].to_numpy(dtype=float)
    e0 = out["e0"].to_numpy(dtype=float)
    e1 = out["e1"].to_numpy(dtype=float)
    e2 = out["e2"].to_numpy(dtype=float)
    e3 = out["e3"].to_numpy(dtype=float)

    build = base * 0.16 * plat * shared * np.exp(0.28 * noise_scale * e0)
    unit = base * 0.15 * np.exp(0.10 * zc) * np.exp(0.30 * noise_scale * e1)
    integ = base * 0.15 * np.exp(0.16 * low_success) * np.exp(0.30 * noise_scale * e2)
    deploy = base * 0.54 * behavior * np.exp(0.22 * noise_scale * e3)
    stages = np.column_stack([build, unit, integ, deploy])
    stages = np.clip(stages, 1.0, None)
    out["build_compilation"] = np.round(stages[:, 0], 3)
    out["unit_testing"] = np.round(stages[:, 1], 3)
    out["integration_testing"] = np.round(stages[:, 2], 3)
    out["deployment"] = np.round(stages[:, 3], 3)
    out["total_execution_time_sec"] = np.round(out[STAGE_COLUMNS].sum(axis=1), 3)
    out["build_duration_sec"] = out["build_compilation"]
    out["test_execution_duration_sec"] = np.round(out["unit_testing"] + out["integration_testing"], 3)
    out["bottleneck"] = assign_bottleneck_labels(out)
    return out


def _positive_rate(frame: pd.DataFrame, noise_scale: float) -> float:
    labeled = _materialize(frame, noise_scale)
    return float(labeled["bottleneck"].mean())


def _calibrate(frame: pd.DataFrame) -> float:
    """Bisect the residual scale until the positive rate is about 35%."""

    def objective(scale: float) -> float:
        return _positive_rate(frame, scale) - TARGET_BOTTLENECK_RATE

    low, high = 0.15, 2.5
    rate_low, rate_high = objective(low), objective(high)
    expand = 0
    while rate_low > 0 and low > 0.02 and expand < 6:
        low *= 0.5
        rate_low = objective(low)
        expand += 1
    expand = 0
    while rate_high < 0 and high < 8 and expand < 6:
        high *= 1.6
        rate_high = objective(high)
        expand += 1
    if rate_low > 0 or rate_high < 0:
        grid = np.linspace(0.05, 4.0, 24)
        rates = [(float(g), _positive_rate(frame, float(g))) for g in grid]
        best = min(rates, key=lambda item: abs(item[1] - TARGET_BOTTLENECK_RATE))
        logger.warning("Bisection bracket did not change sign. Closest scale %.4f rate %.4f", best[0], best[1])
        return best[0]
    scale = float(bisect(objective, low, high, xtol=0.01))
    return scale


def _format_timestamp(ts: pd.Timestamp, rng: np.random.Generator) -> str:
    """Encode one UTC instant in one of four messy but parseable forms."""
    utc = pd.Timestamp(ts).tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    roll = rng.random()
    if roll < 0.40:
        return utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    if roll < 0.65:
        shifted = utc.tz_convert("Asia/Kolkata")
        return shifted.strftime("%Y-%m-%dT%H:%M:%S%z")[:-2] + ":" + shifted.strftime("%z")[-2:]
    if roll < 0.85:
        return utc.strftime("%d %b %Y %H:%M:%S UTC")
    return str(int(utc.timestamp()))


def _log_line(row: pd.Series, rng: np.random.Generator) -> str:
    stages = {name: float(row[name]) for name in STAGE_COLUMNS}
    dominant = max(stages, key=stages.get)
    fail_p = 0.08 + 0.25 * (1.0 - float(row["build_success_ratio"]))
    level = "ERROR" if rng.random() < fail_p else "INFO"
    exit_code = 1 if level == "ERROR" else 0
    return (
        f"[{row['timestamp']}] level={level} stage={dominant} "
        f"duration_sec={stages[dominant]:.1f} exit={exit_code} "
        f"trigger={row['trigger']} branch={row['branch']}"
    )


def _inject_quality_issues(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    raw = frame.copy()
    n_dupes = int(round(DUPLICATE_FRACTION * len(raw)))
    dup_index = rng.choice(raw.index.to_numpy(), size=n_dupes, replace=False)
    dupes = raw.loc[dup_index].copy()
    dupes["log_text"] = dupes["log_text"].map(lambda text: f"webhook retry delivery | {text}")
    raw = pd.concat([raw, dupes], ignore_index=True)

    nullable = MODEL_FEATURES + STAGE_COLUMNS
    for column in nullable:
        mask = rng.random(len(raw)) < MISSING_CELL_FRACTION
        raw.loc[mask, column] = np.nan

    raw["timestamp"] = [_format_timestamp(pd.Timestamp(ts), rng) for ts in raw["timestamp"]]
    return raw


def _diagnostic_f1(frame: pd.DataFrame) -> float:
    """Quick read on whether behavior, not a single column, carries the label."""
    model = RandomForestClassifier(
        n_estimators=80,
        max_depth=8,
        min_samples_leaf=5,
        random_state=SEED,
        n_jobs=1,
    )
    scores = cross_val_score(
        model,
        frame[BEHAVIOR_ONLY],
        frame["bottleneck"],
        cv=3,
        scoring="f1",
    )
    return float(scores.mean())


def generate(seed: int = SEED) -> pd.DataFrame:
    """Build the clean structural table, calibrate it, then return the dirty raw table."""
    rng = np.random.default_rng(seed)
    if sum(int(spec["n_runs"]) for spec in PROJECT_SPECS) != N_RUNS:
        raise RuntimeError("Project catalog does not sum to 4,800 runs.")

    blocks: list[pd.DataFrame] = []
    dev_cursor = 1
    run_cursor = 1
    for spec in PROJECT_SPECS:
        block, dev_cursor, run_cursor = _one_project(spec, rng, dev_cursor, run_cursor)
        blocks.append(block)
    features = pd.concat(blocks, ignore_index=True)
    if len(features) != N_RUNS:
        raise RuntimeError(f"Expected {N_RUNS} runs, built {len(features)}.")

    noise_scale = _calibrate(features)
    clean = _materialize(features, noise_scale)
    rate = float(clean["bottleneck"].mean())
    logger.info(
        "Calibrated noise_scale=%.4f (gain=%.4f at SNR %.2f). Bottleneck rate=%.4f",
        noise_scale,
        SIGNAL_TO_NOISE * noise_scale,
        SIGNAL_TO_NOISE,
        rate,
    )
    if abs(rate - TARGET_BOTTLENECK_RATE) > BOTTLENECK_RATE_TOLERANCE:
        raise RuntimeError(f"Class rate {rate:.4f} is outside the tolerance around 0.35.")

    behavior_f1 = _diagnostic_f1(clean)
    logger.info("Diagnostic 3-fold F1 using behavior/process features only: %.4f", behavior_f1)

    export_cols = [
        "project_id",
        "project_name",
        "team_type",
        "run_id",
        "developer_id",
        "platform",
        "trigger",
        "branch",
        "timestamp",
        "commit_frequency",
        "code_churn",
        "pr_review_latency_hours",
        "build_success_ratio",
        "test_execution_duration_sec",
        "deployment_delay_min",
        "historical_build_success_ratio",
        "build_duration_sec",
        *STAGE_COLUMNS,
        "total_execution_time_sec",
        "bottleneck",
    ]
    structural = clean[export_cols].copy()
    structural["code_churn"] = structural["code_churn"].round(2)
    structural["pr_review_latency_hours"] = structural["pr_review_latency_hours"].round(3)
    structural["deployment_delay_min"] = structural["deployment_delay_min"].round(3)
    structural["historical_build_success_ratio"] = structural["historical_build_success_ratio"].round(4)
    # Re-apply the label after rounding so the stored total and the rule agree.
    structural["bottleneck"] = assign_bottleneck_labels(structural)
    final_rate = float(structural["bottleneck"].mean())
    logger.info("Bottleneck rate after rounding: %.4f", final_rate)

    structural["log_text"] = structural.apply(lambda row: _log_line(row, rng), axis=1)
    raw = _inject_quality_issues(structural, rng)

    meta = {
        "n_unique_runs": int(structural["run_id"].nunique()),
        "n_raw_rows": int(len(raw)),
        "n_projects": int(structural["project_id"].nunique()),
        "bottleneck_rate": final_rate,
        "noise_scale": noise_scale,
        "signal_to_noise": SIGNAL_TO_NOISE,
        "diagnostic_behavior_f1": behavior_f1,
        "distributions": {
            "commit_frequency": "Poisson",
            "code_churn": "log-normal",
            "pr_review_latency_hours": "log-normal",
            "build_success_ratio": "Binomial(10, p) / 10",
            "deployment_delay_min": "gamma",
            "historical_build_success_ratio": "beta latent + gaussian noise",
            "stage_durations": "log-normal multipliers; deployment carries the behavior score",
        },
        "label_rule": "bottleneck = 1 if total_execution_time_sec > 1.5 * project median",
        "assumptions": [
            "Latent developer discipline is not stored; it only shapes the observed features.",
            "The label uses each project's full-sample median, so it is retrospective.",
            "Timing features are this run's compile time and unit+integration time.",
            "The behavior interaction is loaded mainly onto the deployment stage so timing features are not a sufficient statistic for the label.",
        ],
    }
    (config.DATA_RAW / "generation_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    raw.to_csv(RAW_RUNS, index=False)
    logger.info("Wrote %s (%d rows including webhook retries).", RAW_RUNS, len(raw))
    return raw


def main() -> None:
    generate(SEED)


if __name__ == "__main__":
    main()
