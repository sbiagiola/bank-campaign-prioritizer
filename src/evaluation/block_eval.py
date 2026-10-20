"""Evaluation of contact prioritization across consecutive chronological blocks in the test partition."""

from typing import Dict, List, Any
import numpy as np
import pandas as pd
from src.metrics.top_k import recall_at_k, precision_at_k, lift_at_k


def evaluate_consecutive_blocks(
    y_test: pd.Series,
    predictions_dict: Dict[str, np.ndarray],
    num_blocks: int = 5,
    k: float = 0.20,
) -> List[Dict[str, Any]]:
    """
    Evaluates prioritization performance across consecutive blocks within the test set.
    NOTE: These blocks represent consecutive chronological contact batches in call center operations,
    and DO NOT represent calendar months.

    Args:
        y_test: Ground truth labels of the test set.
        predictions_dict: Dictionary mapping model names to predicted probabilities on test set.
        num_blocks: Number of consecutive chunks to partition the test set into.
        k: Contact capacity ratio (default 0.20).

    Returns:
        List of dictionaries with block-level metrics.
    """
    y_arr = np.asarray(y_test)
    n = len(y_arr)
    block_size = n // num_blocks
    results = []

    for b in range(num_blocks):
        start_idx = b * block_size
        end_idx = (b + 1) * block_size if b < num_blocks - 1 else n

        y_block = y_arr[start_idx:end_idx]
        n_block = len(y_block)
        positives_block = int(np.sum(y_block == 1))
        base_rate_block = float(positives_block / n_block) if n_block > 0 else 0.0

        block_data: Dict[str, Any] = {
            "block_id": b + 1,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "n_contacts_available": n_block,
            "positive_subscriptions": positives_block,
            "block_base_rate": round(base_rate_block, 4),
            "models": {},
        }

        # Random baseline performance on this block
        block_data["models"]["random"] = {
            "recall_at_k": round(k, 4),
            "precision_at_k": round(base_rate_block, 4),
            "lift_at_k": 1.0,
        }

        # Models performance
        for model_name, y_prob in predictions_dict.items():
            y_prob_block = y_prob[start_idx:end_idx]
            rec = recall_at_k(y_block, y_prob_block, k=k)
            prec = precision_at_k(y_block, y_prob_block, k=k)
            lift = lift_at_k(y_block, y_prob_block, k=k)

            block_data["models"][model_name] = {
                "recall_at_k": round(rec, 4),
                "precision_at_k": round(prec, 4),
                "lift_at_k": round(lift, 4),
            }

        results.append(block_data)

    return results
