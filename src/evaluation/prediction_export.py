"""Export held-out labels and scores for reproducible offline exploration."""

import numpy as np
import pandas as pd

from src.metrics.top_k import get_top_k_indices


def build_prediction_export(
    y_test: pd.Series,
    scores: dict[str, np.ndarray],
    num_blocks: int = 5,
    k: float = 0.20,
) -> pd.DataFrame:
    """Keep original row IDs and record global and within-batch allocations."""
    n = len(y_test)
    if not 1 <= num_blocks <= n:
        raise ValueError("num_blocks must be between 1 and the test row count")

    frame = pd.DataFrame({"row_id": y_test.index, "y_true": y_test.to_numpy()})
    for name, values in scores.items():
        frame[name] = np.asarray(values)

    block_size = n // num_blocks
    frame["block_id"] = np.minimum(np.arange(n) // block_size, num_blocks - 1) + 1

    for model in ("lr", "cb"):
        column = f"prob_{model}_cal"
        frame[f"selected_{model}_global"] = False
        selected = get_top_k_indices(frame[column].to_numpy(), k)
        frame.loc[selected, f"selected_{model}_global"] = True
        frame[f"rank_{model}_batch"] = 0
        frame[f"selected_{model}_batch"] = False

        for _, batch in frame.groupby("block_id", sort=False):
            order = np.argsort(-batch[column].to_numpy(), kind="stable")
            ordered_indices = batch.index.to_numpy()[order]
            frame.loc[ordered_indices, f"rank_{model}_batch"] = np.arange(1, len(batch) + 1)
            count = len(get_top_k_indices(batch[column].to_numpy(), k))
            frame.loc[ordered_indices[:count], f"selected_{model}_batch"] = True

    return frame
