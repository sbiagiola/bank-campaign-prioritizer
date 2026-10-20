"""Unit tests for ranking, Top-K, and economic metrics."""

import numpy as np
import pytest
from src.metrics.top_k import recall_at_k, precision_at_k, lift_at_k, compute_all_metrics
from src.metrics.economic import simulate_campaign_economics


def test_top_k_metrics_exact_calculation():
    # 10 candidates, 4 positives: indices 0, 1, 2, 3
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    # Model assigns high prob to indices 0, 1 and then wrong ones
    y_prob = np.array([0.9, 0.8, 0.2, 0.1, 0.7, 0.6, 0.05, 0.04, 0.03, 0.02])

    # With k=0.20 (top 2 candidates): indices 0 and 1 (predicted scores 0.9 and 0.8)
    # Both are positive (y=1)
    rec_20 = recall_at_k(y_true, y_prob, k=0.20)
    prec_20 = precision_at_k(y_true, y_prob, k=0.20)
    lift_20 = lift_at_k(y_true, y_prob, k=0.20)

    # 2 captured out of 4 total positives -> recall = 2/4 = 0.50
    assert rec_20 == 0.50

    # 2 captured out of 2 contacted -> precision = 2/2 = 1.00
    assert prec_20 == 1.00

    # Base rate = 4/10 = 0.40 -> lift = 1.00 / 0.40 = 2.50
    assert lift_20 == 2.50


def test_economic_simulation_math():
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    y_prob = np.array([0.9, 0.8, 0.2, 0.1, 0.7, 0.6, 0.05, 0.04, 0.03, 0.02])

    econ = simulate_campaign_economics(
        y_true, y_prob, k=0.20, margin_per_subscription=100.0, contact_cost=10.0
    )

    # Contacted: 2 candidates -> Cost = 2 * 10 = 20
    # Captured: 2 conversions -> Gross = 2 * 100 = 200
    # Net profit = 200 - 20 = 180
    assert econ["financial_results"]["model"]["total_cost"] == 20.0
    assert econ["financial_results"]["model"]["gross_margin"] == 200.0
    assert econ["financial_results"]["model"]["net_profit"] == 180.0
    assert econ["financial_results"]["model"]["roi_percentage"] == (180.0 / 20.0) * 100.0
