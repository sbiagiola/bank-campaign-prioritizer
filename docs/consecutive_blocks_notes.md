# Global ranking and consecutive-batch ranking

## Two allocation policies

**Global policy:** assume all 6,180 test candidates are available together. Rank the full pool once and select its highest-scoring 20%: 1,236 records.

**Batch policy:** divide the same test sequence into five consecutive groups of 1,236 records. Rank candidates independently within each group and select approximately 20% from every group: 247 records per group, or 1,235 overall.

Both policies use the same previously fitted model scores. They allocate capacity differently. A global policy can concentrate contacts in a subset of periods; a batch policy reserves capacity for every batch.

Batch ranking assumes that the entire batch's candidate features are available before its allocation decision. This is a retrospective simulation, not an implemented real-time queue.

## Saved batch results at 20%

The following calibrated-model counts come from the [saved audit snapshot](../artifacts/metrics/audited_metrics_summary.json). Each batch selects 247 records.

| Batch | Available positive labels | Logistic captured | Logistic lift | CatBoost captured | CatBoost lift |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 178 | 56 | 1.574 | 41 | 1.153 |
| 2 | 444 | 118 | 1.330 | 99 | 1.116 |
| 3 | 477 | 152 | 1.595 | 99 | 1.039 |
| 4 | 649 | 131 | 1.010 | 138 | 1.064 |
| 5 | 629 | 187 | 1.488 | 148 | 1.177 |
| Total | 2,377 | 644 | 1.356 | 525 | 1.105 |

Aggregate lift is computed from total captured labels, total selected records, and overall prevalence. It is not the arithmetic mean of batch lifts. The random reference is the sum of each batch's expected captured positives, accounting for its actual allocation fraction.

For logistic regression, batch Recall@20% is 27.09%, compared with 23.18% globally. This is a policy difference, not a new model improvement. Batch 4 has lift close to 1; reporting only batches 1, 3, and 5 would obscure that variability.

## Rounding and ties

The top-k helper uses `max(1, int(round(k * n)))` for a nonempty candidate pool. Tied scores preserve original row order through stable sorting. Actual selected fractions can differ slightly from the requested percentage because contacts are indivisible.

## What a batch does not establish

The UCI file is chronologically ordered, but it has no year column or complete dates for individual records. Equal-sized row groups are not equal-duration intervals or calendar months. They may span different lengths of time.

Five batches from one held-out sequence show variation within that sequence. They do not establish long-term stability, uncertainty bounds, or performance in a future economic regime. Changing batch boundaries after inspecting test performance would be exploratory policy tuning and requires a fresh evaluation period.
