# Methodology and limitations

## Scope

This is an offline comparison of subscription-propensity rankings under a contact-capacity constraint. Results measure concentration of historical positive labels among selected records; they do not measure the causal effect of contacting a customer.

The main question is whether prioritization is useful at a fixed budget. Probability quality is evaluated separately because a useful local ranking need not produce reliable absolute probabilities.

## Experimental sequence

1. Load `bank-additional-full.csv` without shuffling, separate the target, and exclude `duration`.
2. Partition by row position: 60% training, 15% selection, 10% calibration, and the remaining 15% test. Integer truncation is used for the first three partitions.
3. Fit logistic preprocessing and both base estimators on training. Use selection for CatBoost early stopping, with patience 40. That partition is therefore not independent of the fitted CatBoost model.
4. Record raw-model metrics on selection. Hyperparameters otherwise come from configuration; there is no automated search or deployment gate.
5. Fit sigmoid calibrators on the separate calibration partition with base estimators frozen.
6. Evaluate raw and calibrated scores on test. Export predictions, aggregate metrics, consecutive-batch results, and diagnostic plots.

Both models use class balancing. Logistic regression uses standardized numeric features and one-hot categorical features; CatBoost uses native categorical handling. See [config.yaml](../config/config.yaml) for parameters and [the feature audit](data_audit.md) for availability assumptions.

## Metrics

For `n` candidates, select `m = max(1, round(k * n))` highest scores. Ties preserve input order. If `TP` is the positive-label count among selected records and `P` is the total positive-label count:

| Metric | Definition | Question |
| --- | --- | --- |
| Recall@k | `TP / P` | What share of observed subscriptions is selected? |
| Precision@k | `TP / m` | What share of selected records has a positive label? |
| Lift@k | `Precision@k / (P / n)` | How concentrated are positive labels relative to random selection? |
| Average precision | `sklearn.metrics.average_precision_score` | How does precision behave across score thresholds? |
| ROC-AUC | `sklearn.metrics.roc_auc_score` | How well are positive-negative pairs ordered overall? |
| Brier score | Mean squared error between probability and binary outcome | How accurate are absolute probability estimates? |

The JSON field `pr_auc` stores average precision for compatibility with the original artifacts. It should be described as average precision rather than trapezoidal area under the precision-recall curve.

The primary operating point is 20%. Trying other capacities in the app is exploratory. Global and batch policies have different information and capacity assumptions; see [batch notes](consecutive_blocks_notes.md).

## Validation does not support a strong deployment claim

The historical audit records these raw-model validation metrics:

| Model | Recall@20% | Lift@20% | Average precision | ROC-AUC |
| --- | ---: | ---: | ---: | ---: |
| Logistic regression | 8.19% | 0.409 | 0.0968 | 0.3804 |
| CatBoost | 17.13% | 0.856 | 0.1290 | 0.4968 |

CatBoost is better than logistic regression on validation Recall@20%, but both rank below random selection at that operating point. Selecting the better of two models does not establish that either should replace a random or existing business policy.

The saved audit describes CatBoost as selected ex ante. The runnable pipeline records raw validation comparisons and CatBoost early stopping, but does not enforce an automatic champion-selection policy. The audit narrative alone is not a versioned pre-registration of the experiment.

On test, logistic regression has the better global top-20% result. Promoting it based on that observation uses test information. Any subsequent tuning or model choice would need a fresh holdout or a prospective evaluation.

## Distribution shift and calibration

Positive-label rates change from 4.81% in training to 12.85% in selection, 6.82% in calibration, and 38.46% in test. Macroeconomic variables also change over the sequence. These are observed distribution differences; their causes and contributions are not identified by this experiment.

The saved audit reports these test diagnostics:

| Model | Mean probability | Observed positive rate | Brier score |
| --- | ---: | ---: | ---: |
| Logistic, raw | 0.0006 | 0.3846 | 0.3841 |
| Logistic, calibrated | 0.1191 | 0.3846 | 0.3279 |
| CatBoost, raw | 0.5189 | 0.3846 | 0.2637 |
| CatBoost, calibrated | 0.0606 | 0.3846 | 0.3424 |

Calibration improves logistic Brier but worsens CatBoost Brier on test. CatBoost's saved ROC-AUC changes from 0.5272 raw to 0.4728 calibrated, consistent with a reversal of score order. Calibration can therefore change ranking direction, rather than merely rescale scores. Inspecting the fitted sigmoid and evaluating its behavior across periods is a required follow-up before operational use.

A constant probability equal to the **test** prevalence has Brier 0.2367. It uses information unavailable at scoring time and is only a retrospective diagnostic. A constant equal to the **training** prevalence has test Brier about 0.3500 and is an available-at-training reference. The random-selection row in the original JSON mixes ranking references with the retrospective constant-probability Brier; these should be interpreted separately.

The results do not establish that a linear model is generally more robust than trees, that economic variables caused the subscription-rate change, or that calibration solves temporal drift. No ablation or repeated temporal evaluation isolates those explanations.

## Economic scenarios

The simulator applies a margin per selected positive label and a cost per selected contact. Defaults are EUR 150 and EUR 5. These are hypothetical inputs, not values measured in the dataset.

```text
scenario net value = selected positive labels * assumed margin
                     - selected contacts * assumed contact cost
```

For compatibility with the original simulation, the economic helper rounds the random expected subscription count to a whole number. With global 20% allocation this produces 475 rather than approximately 475.4 expected labels. Consequently, the stored EUR 11,400 logistic difference is an illustrative rounded scenario, not realized incremental profit. It also assumes historical outcomes remain applicable under the simulated selection policy.

There is no randomized no-contact group, so propensity must not be presented as treatment uplift. Future realized profit would depend on attribution, changing behavior, actual margins, and operational execution.

## Reproducibility and artifact provenance

- `make train` rebuilds models, raw selection metrics, raw and calibrated test metrics, batch metrics, row-level test predictions, and plots using the configured seed.
- `run_summary.json` and `block_evaluation.json` are generated pipeline outputs.
- `test_row_predictions.csv` contains test labels and raw/calibrated scores. It also records global selections and within-batch ranks/selections at the configured capacity.
- `audited_metrics_summary.json` is a separate historical audit snapshot. It contains extra selection and score-distribution diagnostics and is not regenerated by the pipeline. Its prose should be read as audit notes, not executable policy.
- A dependency lock and fixed seeds improve repeatability, but do not guarantee bit-identical results across hardware and library implementations.

Raw data and generated model binaries remain local. A public clone can reproduce them through the documented download and training commands. The app needs only the benchmark prediction export.

## Next experiments

These are planned improvements, not completed capabilities:

1. Use repeated rolling-origin evaluation to check whether ranking improvements persist across periods. Keep a new final period untouched after selecting a policy.
2. Add baselines using only information available at each decision point, including constant-probability references and simple prior-campaign rules.
3. Examine why raw validation ranking is weak, including feature ablations, macroeconomic variables, class weights, and handling of `pdays = 999`.
4. Inspect sigmoid coefficients and score distributions; compare uncalibrated and calibrated models across multiple chronological windows.
5. Quantify uncertainty with resampling that respects temporal dependence rather than assuming independent rows.
6. Define candidate-pool timing and contact policy with an operational stakeholder before interpreting batch results as actionable.
7. Before a real deployment, establish scoring contracts, data checks, monitoring, retraining criteria, and prospective evaluation. None is implemented here.
