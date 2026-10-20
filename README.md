# Bank Campaign Prioritizer

An ML case study on ranking bank marketing contacts under a fixed contact budget, with chronological evaluation and explicit analysis of distribution shift.

**Question:** if only 20% of the available candidates can be contacted, which ranking captures the largest share of observed term deposit subscriptions?

The project compares logistic regression and CatBoost on the public UCI Bank Marketing dataset. It includes a training pipeline, probability calibration, ranking metrics, a scenario simulator, and tests. It is an offline portfolio project; no live campaign or production deployment is claimed.

## Results and interpretation

The saved test set contains 6,180 records and 2,377 observed subscriptions. Selecting the top 20% globally means selecting 1,236 records.

| Strategy | Recall@20% | Precision@20% | Lift@20% | Average precision | Brier score |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random selection, expected | 20.00% | 38.46% | 1.00 | 0.3846* | — |
| Logistic regression, raw | 23.18% | 44.58% | 1.159 | 0.3984 | 0.3841 |
| Logistic regression, calibrated | 23.18% | 44.58% | 1.159 | 0.3984 | 0.3279 |
| CatBoost, raw | 22.25% | 42.80% | 1.113 | 0.4058 | 0.2637 |
| CatBoost, calibrated | 21.88% | 42.07% | 1.094 | 0.4002 | 0.3424 |

*The test prevalence is the conventional no-skill precision-recall reference, not a measured average precision from a particular random ranking. Higher ranking metrics are better; lower Brier is better. The implementation uses `average_precision_score`, not trapezoidal PR-AUC.*

These values come from [the saved run summary](artifacts/metrics/run_summary.json). The calibrated logistic model captures 551 observed subscriptions, versus approximately 475.4 expected under random selection. This is a modest ranking improvement in one historical test period, without an uncertainty estimate.

Three qualifications are central to the case study:

- **Validation and test favor different models.** The saved audit reports validation Recall@20% of 17.13% for raw CatBoost and 8.19% for raw logistic regression. Both are below the random reference. Logistic regression's better test result is a retrospective comparison, not evidence that validation would have selected it. The pipeline records the comparison; it does not implement an automatic deployment decision.
- **Calibration does not resolve the shift.** Test prevalence is 38.46%, compared with 6.82% in the calibration partition. In the saved audit, mean calibrated predictions are about 11.91% for logistic regression and 6.06% for CatBoost. Logistic calibration improves Brier relative to its raw output; CatBoost calibration worsens it.
- **Global and batch ranking are different policies.** Selecting 20% of the full test pool gives different results from selecting 20% within each consecutive batch. Batch results cannot be substituted for the global result.

See [methodology and limitations](docs/methodology.md) for the full interpretation and next experiments.

## Data and evaluation setup

Source: [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing), specifically `bank-additional-full.csv`: 41,188 records, ordered by date from May 2008 to November 2010. The target is whether a term deposit was subscribed (`y = yes`). This is the 20-input dataset variant, not the older 45,211-row file.

The loader preserves row order and excludes `duration`, which is only known after the call. Other pre-contact availability assumptions are documented in the [feature audit](docs/data_audit.md).

| Partition | Records | Positive labels | Positive rate | Purpose |
| --- | ---: | ---: | ---: | --- |
| Train | 24,712 | 1,188 | 4.81% | Fit preprocessing and base models |
| Selection | 6,178 | 794 | 12.85% | CatBoost early stopping and model comparison |
| Calibration | 4,118 | 281 | 6.82% | Fit sigmoid calibration on frozen estimators |
| Test | 6,180 | 2,377 | 38.46% | Final historical comparison and diagnostics |

These are contiguous row partitions, not verified calendar-year partitions. The observed prevalence shift is substantial; this experiment does not establish its cause.

## Reproduce locally

Requirements: Python 3.13+, [uv](https://docs.astral.sh/uv/), and `make` for the shortcuts. Run commands from the repository root.

```bash
uv sync --locked
make test
```

The raw dataset is excluded from version control. Download and extract the exact variant with:

```bash
uv run python scripts/download_data.py
make train
make app
```

The downloader uses Python's standard library, downloads the UCI archive, and extracts only `bank-additional-full.csv` into `data/raw/bank-additional/`. Training uses the parameters in [config/config.yaml](config/config.yaml).

Equivalent commands without `make`:

```bash
uv run python -m pytest -v
uv run python -m src.pipeline
uv run python -m streamlit run app/main.py
```

The app reads saved test predictions and does not need model binaries or raw data. For each capacity setting, it recomputes observed top-k outcomes. Costs and margins are user-supplied scenario assumptions. The expected random subscription count in the economic helper is rounded to a whole number; financial outputs are illustrative, not realized revenue or estimates of causal uplift.

## Repository map

```text
app/main.py                      Interactive historical evaluation
config/config.yaml               Features, splits, models, economic assumptions
scripts/download_data.py         Download the exact public dataset variant
src/data/                        Loading and contiguous splitting
src/features/                    Scaling, one-hot encoding, categorical handling
src/models/                      Logistic regression, CatBoost, calibration
src/metrics/                     Top-k metrics and economic scenarios
src/evaluation/                  Consecutive-batch evaluation
src/pipeline.py                  Training, evaluation, plots, prediction export
tests/                           Leakage, split, metric, export, and app checks
artifacts/metrics/                Saved aggregate results and audit snapshot
artifacts/predictions/            Benchmark test labels and model scores
artifacts/plots/                  Saved diagnostic figures
docs/                            Methods, feature audit, and publication guidance
```

`make train` writes `run_summary.json`, `block_evaluation.json`, test predictions, plots, and local model binaries. `audited_metrics_summary.json` is a separate historical audit snapshot, not an output regenerated by that command. Its selection analysis and batch counts provide additional context; it should not be treated as an independently reproduced experiment.

Tests cover specific properties. They do not prove that every feature is available at decision time, that a model is suitable for production, or that the measured improvement will persist.

## Further reading

- [Methodology, calibration diagnostics, and next experiments](docs/methodology.md)
- [Feature availability audit](docs/data_audit.md)
- [Global ranking versus consecutive batches](docs/consecutive_blocks_notes.md)
- [What to include in a public portfolio repository](docs/publication.md)

## Dataset attribution and licensing

Moro, S., Rita, P., & Cortez, P. (2014). *Bank Marketing* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306.

UCI lists the dataset under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). This attribution also applies to benchmark labels and scores derived from that dataset. The dataset license is separate from the repository's source code; a code license has not yet been selected.
