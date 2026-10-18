# RESULTS SUMMARY

- Unique runs: 4800
- Projects: 12
- Bottleneck rate: 35.0%
- Calibrated noise scale: 1.0404 at SNR 2.2
- Diagnostic behavior-only F1 (3-fold, pre-tuning): 0.684
- Best saved model: SVM (RBF) (seed-42 test F1 74.1%)

## Model comparison

| Model | Seed-42 Acc | Prec | Rec | F1 | ROC-AUC | F1 mean ± std | Paper F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 80.0% | 68.4% | 79.8% | 73.6% | 0.879 | 71.5% ± 2.1 | 78.4% |
| SVM (RBF) | 80.3% | 68.6% | 80.7% | 74.1% | 0.880 | 71.2% ± 2.3 | 83.1% |
| Gradient Boosting | 80.2% | 73.7% | 67.6% | 70.5% | 0.883 | 69.2% ± 2.5 | 91.7% |
| Random Forest | 80.4% | 69.5% | 78.6% | 73.7% | 0.881 | 71.4% ± 1.9 | 93.2% |

## Feature importance (Random Forest MDI)

- Code churn: MDI 25.0% | permutation 25.8% | SHAP 25.1%
- PR review latency: MDI 19.8% | permutation 26.3% | SHAP 21.7%
- Historical build success: MDI 18.3% | permutation 15.7% | SHAP 18.2%
- Build duration: MDI 13.9% | permutation 0.1% | SHAP 11.4%
- Recent build success: MDI 7.2% | permutation 9.9% | SHAP 8.7%
- Deployment delay: MDI 7.0% | permutation 12.1% | SHAP 7.0%
- Test duration: MDI 4.5% | permutation 5.6% | SHAP 2.2%
- Commit frequency: MDI 4.4% | permutation 4.4% | SHAP 5.6%

- Behavioral share: 74.6%
- Infrastructure timing share: 18.4%
- Deployment delay (reported separately): 7.0%

## Leakage ablation (test F1)

- all_features: 74.1% (+0.0 pp vs all features)
- no_timing: 73.3% (-0.9 pp vs all features)
- behavior_only: 70.5% (-3.6 pp vs all features)

## Detection layer (held-out runs)

- Alerts saved (all runs): 10
- Alerts on the test set: 0
- False-alarm rate: n/a (target < 8%: NOT MET)
- Detection rate: 0.0%
- Mean lead time: n/a

Mean hours, over true-positive test alerts, from the earliest preceding run whose bottleneck probability already exceeded tau to the alerted run. Zero means probability first crossed tau on the alerted run.
