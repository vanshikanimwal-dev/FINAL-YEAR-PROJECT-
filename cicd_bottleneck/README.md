# PipelineIQ

Simulated study of the paper *Machine Learning-Based Developer Behavior and Process Bottleneck Analysis in CI/CD Pipelines*, plus a jury dashboard.

CI/CD teams usually notice a bottleneck after the log is already red. This project asks a different question: can developer behavior — commit size, review latency, recent build success — flag a risky run before the pipeline finishes? Five layers answer it.

1. **Data collection.** 4,800 simulated runs across 12 projects (8 single-team, 4 multi-team) on Jenkins, GitHub Actions, and GitLab CI.
2. **Feature processing.** Log parsing, UTC timestamps, webhook deduplication, median imputation and min-max scaling fit on the training split only.
3. **Models.** Logistic regression, RBF SVM, gradient boosting, and random forest, tuned with 5-fold stratified F1 search.
4. **Detection.** A stage alert fires only when model probability exceeds 0.65 **and** the stage stays more than 2 standard deviations above its previous 20-run mean for at least 3 consecutive runs.
5. **Analytics.** The PipelineIQ Streamlit dashboard.

## Assumptions

- No public dataset links developer behavior to run-level bottleneck labels, so the runs are simulated. Distributions: Poisson commit counts, log-normal churn and review latency, binomial recent build success, gamma deployment delay, beta historical success.
- A run is a bottleneck when its total time exceeds **1.5× that project's own median**. The median is computed on the full project, so the label is retrospective.
- The behavior interaction (high churn is costly when reviews are slow and historical success is low) is loaded mainly onto the deployment stage. Compile and test times are only mild functions of behavior, so timing features are not a sufficient statistic for the label.
- `runs_clean.csv` stays in original units. `models/scaler.joblib` holds the training-split scaler.
- RFE is diagnostic. Classifiers train on all eight features so the ablation stays interpretable.
- The headline model is chosen by seed-42 test F1, so that single number is mildly optimistic. The five-seed mean ± std is the uncertainty check.
- Predictions cover every run, including the training rows. Metrics and the detection rates use the held-out 960 runs.
- sklearn 1.9 deprecates `SVC(probability=True)`. The saved SVM still uses it, because the study needs probabilities. The warning is filtered in the app.
- Relative times in the alert feed are measured from the latest run in view, not from today's wall clock. The sample ends in mid-2025.

## How to run

From this folder, with the virtualenv Python:

```bash
python -m src.generate_data
python -m src.preprocess
python -m src.features
python -m src.train
python -m src.evaluate
python -m src.explain
python -m src.detect
python -m pytest -q tests
python -m streamlit run app/dashboard.py
```

Or, if GNU make is installed: `make all`, then `make app`.

`make data` builds the dataset, `make train` fits and evaluates, `make detect` writes alerts, `make test` runs pytest.

## Results

4,800 unique runs, 12 projects, bottleneck rate **35.0%** after calibrating the residual scale (noise scale 1.04 at a fixed signal-to-noise ratio of 2.2). Raw file has 4,896 rows because 2% are webhook retries. Training split 3,840 / test 960.

Seed-42 held-out scores, and F1 across five seeds:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | F1 mean ± std | Paper F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 80.0% | 68.4% | 79.8% | 73.6% | 0.879 | 71.5% ± 2.1 | 78.4% |
| SVM (RBF) | 80.3% | 68.6% | 80.7% | 74.1% | 0.880 | 71.2% ± 2.3 | 83.1% |
| Gradient Boosting | 80.2% | 73.7% | 67.6% | 70.5% | 0.883 | 69.2% ± 2.5 | 91.7% |
| Random Forest | 80.4% | 69.5% | 78.6% | 73.7% | 0.881 | 71.4% ± 1.9 | 93.2% |

Best saved model: **SVM (RBF), test F1 74.1%**.

These sit below the paper, especially for the tree models (paper RF F1 93.2%). That is expected. The simulator was not adjusted to chase those percentages. The label is a threshold on a noisy, interactive function of several features, so no model gets a near-perfect F1. ROC-AUC is about 0.88 for every model: ranking is easier than a hard 0.5 cutoff. Trees do not beat the linear model here, because much of the signal is still close to linear once the interaction is only one part of the score.

Random Forest MDI (sums to 100%): code churn 25.0, PR review latency 19.8, historical build success 18.3, build duration 13.9, recent build success 7.2, deployment delay 7.0, test duration 4.5, commit frequency 4.4. Permutation importance on the SVM agrees that churn and review latency dominate; build duration's MDI share collapses under permutation (0.1%), which is the bias MDI is known for.

- Behavioral share of MDI: **74.6%**
- Infrastructure timing (build + test duration): **18.4%**
- Deployment delay, reported separately: **7.0%**

Leakage ablation, same SVM hyperparameters, test F1:

| Setting | F1 | vs all features |
| --- | ---: | ---: |
| All features | 74.1% | — |
| No timing features | 73.3% | −0.9 pp |
| Behavior only | 70.5% | −3.6 pp |

Behavior carries almost the same F1 as the full model. Timing features add a little and are not required for the claim.

Detection at τ = 0.65: **10 alerts** on the full timeline, **0 on the held-out runs**. False-alarm rate is therefore undefined and the **8% target is not met**. Detection rate on the test set is 0%. The persistence rule (3 consecutive z > 2) is strict relative to how rarely a stage stays that hot, so most labeled bottlenecks never become confirmed stage alerts. Figures: `reports/figures/tau_tradeoff.png`. At τ = 0.50 there is 1 test alert and a 0% false-alarm rate, with almost no detection. Loosening τ does not create a useful detector in this sample.

RFE (diagnostic, 5 of 7 core features kept): code churn, PR review latency, recent build success, deployment delay, historical build success. Test duration and commit frequency were ranked lower. No feature pair exceeded |r| = 0.80.

## Screenshots

Start the app, set the browser to 1920×1080, and capture each page from the left menu: Executive Overview, Developer Behavior, Stage Performance, Alert Center, What-If Simulator, Model Lab, Architecture. Turn on **Presentation mode** in the sidebar for a projector.

## Limitations

Simulated data. Generalization to a real organization is unproven. The label uses a full-sample median. Seven features omit review text, flaky tests, and queue time. Concept drift is not modelled. A production system would need real GitHub/GitLab logs, sequence models, NLP on commits and comments, and online updates.

## Three-minute jury script

**0:00 — Executive Overview.** "We forecast bottleneck risk from developer behavior instead of waiting for the log." Point at the F1 chip (SVM, 74.1% held-out), the bottleneck rate (35%), and hours at risk. Read one story card. Show the league table: colour is always paired with HIGH / MEDIUM / LOW.

**0:40 — Developer Behavior.** Switch the heatmap to PR review latency. Open the top developer. Read the coaching note, which is computed from their z-scores versus the filtered team. On the scatter, the dotted lines are the 75th percentiles: that corner is the interaction zone.

**1:20 — What-If Simulator.** Press **Big-bang commit**, then **Healthy team**. The gauge is the saved model, not a lookup table. Read the recommendation; the two percentages are the model's probabilities before and after the edit.

**2:10 — Model Lab.** "We did not match the paper's 93% random-forest F1, and we did not tune the data to try." Show the paper-versus-ours column, then the behavior 74.6% versus timing 18.4% split, then the ablation (behavior-only loses 3.6 points). End on the detection gauge: the 8% false-alarm target is **not** met, because the persistence rule confirmed no held-out alerts. That is a finding, not a hidden failure.

**2:50 — Architecture, only if asked.** Five layers, then the limitations card.
