"""Economic simulation module translating ML ranking into banking P&L."""

from typing import Dict, Any
import numpy as np
from src.metrics.top_k import get_top_k_indices


def simulate_campaign_economics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    k: float = 0.20,
    margin_per_subscription: float = 150.0,
    contact_cost: float = 5.0,
) -> Dict[str, Any]:
    """
    Simulates commercial and financial campaign outcomes under capacity constraint k.
    Separates business assumptions from observed empirical metrics.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    n_total = len(y_true)
    total_positives = int(np.sum(y_true == 1))

    # Top-K selection
    top_indices = get_top_k_indices(y_prob, k)
    n_contacted = len(top_indices)
    captured_positives = int(np.sum(y_true[top_indices] == 1))

    # Observed performance
    recall = float(captured_positives / total_positives) if total_positives > 0 else 0.0
    precision = float(captured_positives / n_contacted) if n_contacted > 0 else 0.0

    # Model Economics
    cost_model = n_contacted * contact_cost
    gross_model = captured_positives * margin_per_subscription
    net_profit_model = gross_model - cost_model
    roi_model = (net_profit_model / cost_model * 100.0) if cost_model > 0 else 0.0

    # Random Baseline Economics (contacting k fraction at random)
    expected_rand_positives = int(round(k * total_positives))
    cost_rand = cost_model
    gross_rand = expected_rand_positives * margin_per_subscription
    net_profit_rand = gross_rand - cost_rand
    roi_rand = (net_profit_rand / cost_rand * 100.0) if cost_rand > 0 else 0.0

    # Contact All (100%) Baseline Economics
    cost_all = n_total * contact_cost
    gross_all = total_positives * margin_per_subscription
    net_profit_all = gross_all - cost_all
    roi_all = (net_profit_all / cost_all * 100.0) if cost_all > 0 else 0.0

    # Value added by model
    incremental_profit_vs_random = net_profit_model - net_profit_rand
    incremental_subscriptions_vs_random = captured_positives - expected_rand_positives

    return {
        "assumptions": {
            "contact_budget_fraction_k": k,
            "margin_per_subscription": margin_per_subscription,
            "contact_cost": contact_cost,
        },
        "observed_campaign_facts": {
            "total_candidates": n_total,
            "total_available_subscriptions": total_positives,
            "overall_base_rate": round(total_positives / n_total, 4) if n_total > 0 else 0.0,
            "n_contacted": n_contacted,
            "observed_subscriptions_captured": captured_positives,
            "recall_capture_rate": round(recall, 4),
            "precision_conversion_rate": round(precision, 4),
        },
        "financial_results": {
            "model": {
                "total_cost": round(cost_model, 2),
                "gross_margin": round(gross_model, 2),
                "net_profit": round(net_profit_model, 2),
                "roi_percentage": round(roi_model, 2),
            },
            "random_selection": {
                "total_cost": round(cost_rand, 2),
                "expected_subscriptions": expected_rand_positives,
                "gross_margin": round(gross_rand, 2),
                "net_profit": round(net_profit_rand, 2),
                "roi_percentage": round(roi_rand, 2),
            },
            "contact_all_100pct": {
                "total_cost": round(cost_all, 2),
                "total_subscriptions": total_positives,
                "gross_margin": round(gross_all, 2),
                "net_profit": round(net_profit_all, 2),
                "roi_percentage": round(roi_all, 2),
            },
            "incremental_value_added_vs_random": {
                "incremental_net_profit": round(incremental_profit_vs_random, 2),
                "incremental_subscriptions": incremental_subscriptions_vs_random,
            },
        },
    }
