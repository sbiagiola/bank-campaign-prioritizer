"""Checks for auditable allocations and deterministic handling of score ties."""

import numpy as np
import pandas as pd

from src.evaluation.prediction_export import build_prediction_export
from src.metrics.top_k import get_top_k_indices


def test_tied_scores_keep_arrival_order():
    scores = np.array([0.1, 0.8, 0.8, 0.8, 0.2])
    np.testing.assert_array_equal(get_top_k_indices(scores, k=0.4), [1, 2])


def test_export_distinguishes_global_and_batch_allocation():
    labels = pd.Series([1, 0, 1, 0, 1, 0], index=np.arange(100, 106))
    scores = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.0])
    frame = build_prediction_export(
        labels,
        {"prob_lr_cal": scores, "prob_cb_cal": scores},
        num_blocks=2,
        k=1 / 3,
    )

    assert frame["row_id"].tolist() == list(range(100, 106))
    assert frame["block_id"].tolist() == [1, 1, 1, 2, 2, 2]
    assert frame.loc[frame["selected_lr_global"], "row_id"].tolist() == [100, 101]
    assert frame.loc[frame["selected_lr_batch"], "row_id"].tolist() == [100, 103]
    assert frame["rank_lr_batch"].tolist() == [1, 2, 3, 1, 2, 3]


def test_export_assigns_remainder_to_last_batch():
    labels = pd.Series([0, 1, 0, 1, 0, 1, 0])
    scores = np.arange(7, dtype=float) / 7
    frame = build_prediction_export(
        labels, {"prob_lr_cal": scores, "prob_cb_cal": scores}, num_blocks=3
    )
    assert frame.groupby("block_id").size().tolist() == [2, 2, 3]
