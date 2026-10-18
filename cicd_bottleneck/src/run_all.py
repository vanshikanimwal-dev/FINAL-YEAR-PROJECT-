"""Run the full study and print the results summary from the saved artifacts."""

from __future__ import annotations

import json

import pandas as pd

from config import (
    ABLATION_PATH,
    DETECTION_REPORT_PATH,
    FEATURE_IMPORTANCE_PATH,
    METRICS_PATH,
    MODEL_COMPARISON_PATH,
    SUMMARY_PATH,
    get_logger,
)
from src.detect import main as detect_main
from src.evaluate import main as evaluate_main
from src.explain import main as explain_main
from src.features import main as features_main
from src.generate_data import main as generate_main
from src.preprocess import main as preprocess_main
from src.train import main as train_main

logger = get_logger("run_all")


def _fmt_pm(mean: float, std: float) -> str:
    return f"{mean * 100:.1f}% ± {std * 100:.1f}"


def write_summary() -> str:
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    comparison = pd.read_csv(MODEL_COMPARISON_PATH)
    importance = pd.read_csv(FEATURE_IMPORTANCE_PATH)
    ablation = pd.read_csv(ABLATION_PATH)
    detection = json.loads(DETECTION_REPORT_PATH.read_text(encoding="utf-8"))
    meta_path = METRICS_PATH.parents[1] / "data" / "raw" / "generation_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    lines = [
        "# RESULTS SUMMARY",
        "",
        f"- Unique runs: {meta.get('n_unique_runs', 'n/a')}",
        f"- Projects: {meta.get('n_projects', 'n/a')}",
        f"- Bottleneck rate: {meta.get('bottleneck_rate', float('nan')):.1%}",
        f"- Calibrated noise scale: {meta.get('noise_scale', float('nan')):.4f} at SNR {meta.get('signal_to_noise', 'n/a')}",
        f"- Diagnostic behavior-only F1 (3-fold, pre-tuning): {meta.get('diagnostic_behavior_f1', float('nan')):.3f}",
        f"- Best saved model: {metrics['best_model_display']} (seed-42 test F1 {metrics['best_f1'] * 100:.1f}%)",
        "",
        "## Model comparison",
        "",
        "| Model | Seed-42 Acc | Prec | Rec | F1 | ROC-AUC | F1 mean ± std | Paper F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in comparison.iterrows():
        lines.append(
            "| {name} | {acc:.1f}% | {prec:.1f}% | {rec:.1f}% | {f1:.1f}% | {auc:.3f} | {f1pm} | {paper:.1f}% |".format(
                name=row["display_name"],
                acc=row["accuracy"] * 100,
                prec=row["precision"] * 100,
                rec=row["recall"] * 100,
                f1=row["f1"] * 100,
                auc=row["roc_auc"],
                f1pm=_fmt_pm(row["f1_mean"], row["f1_std"]),
                paper=row["paper_f1"],
            )
        )
    lines.extend(["", "## Feature importance (Random Forest MDI)", ""])
    for _, row in importance.sort_values("mdi_pct", ascending=False).iterrows():
        shap_bit = "n/a" if pd.isna(row["shap_pct"]) else f"{row['shap_pct']:.1f}%"
        lines.append(
            f"- {row['label']}: MDI {row['mdi_pct']:.1f}% | permutation {row['permutation_pct']:.1f}% | SHAP {shap_bit}"
        )
    shares = importance.groupby("group")["mdi_pct"].sum()
    lines.extend(
        [
            "",
            f"- Behavioral share: {shares.get('behavioral', 0):.1f}%",
            f"- Infrastructure timing share: {shares.get('timing', 0):.1f}%",
            f"- Deployment delay (reported separately): {shares.get('deployment_delay', 0):.1f}%",
            "",
            "## Leakage ablation (test F1)",
            "",
        ]
    )
    for _, row in ablation.iterrows():
        lines.append(f"- {row['setting']}: {row['f1'] * 100:.1f}% ({row['delta_f1_vs_all'] * 100:+.1f} pp vs all features)")
    far = detection.get("false_alarm_rate")
    far_text = "n/a" if far is None else f"{far * 100:.1f}%"
    lead = detection.get("mean_lead_time_hours")
    lead_text = "n/a" if lead is None else f"{lead:.2f} h"
    det = detection.get("detection_rate")
    det_text = "n/a" if det is None else f"{det * 100:.1f}%"
    lines.extend(
        [
            "",
            "## Detection layer (held-out runs)",
            "",
            f"- Alerts saved (all runs): {detection.get('n_alerts_all_runs', 0)}",
            f"- Alerts on the test set: {detection.get('n_alerts_test', 0)}",
            f"- False-alarm rate: {far_text} (target < 8%: {'MET' if detection.get('target_met') else 'NOT MET'})",
            f"- Detection rate: {det_text}",
            f"- Mean lead time: {lead_text}",
            "",
            detection.get("lead_time_definition", ""),
            "",
        ]
    )
    text = "\n".join(lines)
    SUMMARY_PATH.write_text(text, encoding="utf-8")
    return text


def main() -> None:
    generate_main()
    preprocess_main()
    features_main()
    train_main()
    evaluate_main()
    explain_main()
    detect_main()
    summary = write_summary()
    logger.info("\n%s", summary)
    print(summary)


if __name__ == "__main__":
    main()
