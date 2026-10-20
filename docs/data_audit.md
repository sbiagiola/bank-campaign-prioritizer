# Feature availability audit

## Decision being modeled

Rank candidates before making a scheduled contact. A feature must be available at that moment. Removing one known leakage variable is necessary, but it does not establish that the remaining feature set is operationally valid in every campaign workflow.

Source definitions: [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing), `bank-additional-full.csv`.

## Availability assumptions

| Feature group | Columns | Assumption and remaining checks |
| --- | --- | --- |
| Customer profile | `age`, `job`, `marital`, `education` | Available from a CRM before contact; historical values must reflect what was known then. |
| Financial status | `default`, `housing`, `loan` | Available before contact if the CRM snapshot is current. Unknown values are not equivalent to a negative answer. |
| Planned contact | `contact`, `month`, `day_of_week` | Usable when channel and contact date have already been scheduled. For ranking before scheduling, use a separate feature set. |
| Current campaign | `campaign` | Includes the current contact. Treat as prior attempts plus the planned attempt only if that count can be reconstructed before the call. |
| Previous campaigns | `pdays`, `previous`, `poutcome` | Use only completed campaign history. `pdays = 999` is a sentinel meaning no previous contact, not an elapsed duration of 999 days. |
| Economic context | `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed` | Assume the values were available at scoring time. Publication lags and later revisions are not verified by the released CSV. |
| Call outcome information | `duration` | Excluded: call duration is unavailable until after the contact. |

The target `y` is removed from the feature matrix and converted to a binary label. The loader excludes `duration` by default and the training pipeline checks its absence. Tests exercise this exclusion and basic target separation.

## Preprocessing

- Logistic regression standardizes numeric columns and one-hot encodes categoricals using preprocessing fitted on training data only. Unseen categories are ignored by the encoder.
- CatBoost receives numeric columns and string-valued categorical columns, with missing categorical values replaced by `unknown`.
- Both model configurations use class balancing. Scores from a weighted objective should not automatically be interpreted as population probabilities.
- The current baseline treats the `pdays` sentinel as numeric. An explicit previous-contact indicator and separate recency feature are useful follow-up experiments.

## Limits of this audit

The dataset does not include a reliable customer identifier or complete per-row timestamps. A row split therefore does not verify customer independence, and exact calendar cutoffs cannot be recovered from `month` alone. Repeated customer history can cross partitions without being detectable.

There is no uncontacted control group. Observed subscriptions support a propensity-ranking exercise, not an estimate of how much a call changes a customer's subscription probability.

The presence of demographic attributes also warrants a separate assessment of segment performance and suitability before operational use. This repository does not contain that assessment.
