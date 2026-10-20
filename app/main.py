"""Explore historical ranking results and hypothetical economic scenarios."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.block_eval import evaluate_consecutive_blocks
from src.metrics.economic import simulate_campaign_economics
from src.metrics.top_k import compute_cumulative_gain_curve, compute_decile_lifts


st.set_page_config(page_title="Bank Campaign Prioritizer", page_icon="🏦", layout="wide")
st.title("Bank Campaign Prioritizer")
st.markdown(
    "Explore how historical subscription rankings change under a limited contact budget. "
    "Results use the held-out UCI Bank Marketing records. Economic values are scenarios "
    "based on your assumptions."
)
st.caption(
    "Offline portfolio study · Chronological row splits · Post-call duration excluded · "
    "No live campaign or causal uplift estimate"
)


@st.cache_data(ttl=60, max_entries=2)
def load_predictions():
    path = ROOT / "artifacts/predictions/test_row_predictions.csv"
    return pd.read_csv(path) if path.exists() else None


st.sidebar.header("Contact policy")
budget_pct = st.sidebar.slider(
    "Contact capacity (% of candidates)", 5, 50, 20, 5, key="capacity"
)
score_columns = {
    "Logistic regression (calibrated)": "prob_lr_cal",
    "Logistic regression (raw)": "prob_lr_raw",
    "CatBoost (calibrated)": "prob_cb_cal",
    "CatBoost (raw)": "prob_cb_raw",
}
model_choice = st.sidebar.selectbox("Model to inspect", list(score_columns), key="model")
st.sidebar.caption("Model comparisons on this test period are retrospective.")
st.sidebar.header("Economic assumptions")
margin = st.sidebar.number_input(
    "Assumed margin per subscription (EUR)", 10.0, 1000.0, 150.0, 10.0, key="margin"
)
cost = st.sidebar.number_input(
    "Assumed cost per contact (EUR)", 0.5, 50.0, 5.0, 0.5, key="cost"
)

predictions = load_predictions()
if predictions is None:
    st.error("Test predictions are missing. Run `make train` from the repository root.")
    st.stop()

k = budget_pct / 100
labels = predictions["y_true"].to_numpy()
scores = predictions[score_columns[model_choice]].to_numpy()
base_rate = float(labels.mean())
economics = simulate_campaign_economics(labels, scores, k, margin, cost)
facts = economics["observed_campaign_facts"]
financial = economics["financial_results"]
recall = facts["recall_capture_rate"]
precision = facts["precision_conversion_rate"]
lift = precision / base_rate if base_rate else 0.0

st.subheader(f"Global ranking at {budget_pct}% contact capacity")
st.caption(
    f"Select {facts['n_contacted']:,} of {len(labels):,} candidates from the full test pool. "
    "Each capacity setting is computed from saved scores and observed labels."
)
col1, col2, col3, col4 = st.columns(4)
col1.metric(
    f"Recall@{budget_pct}%", f"{recall:.1%}",
    delta=f"{(recall - k) * 100:+.1f} pp vs random reference",
)
col2.metric(
    f"Precision@{budget_pct}%", f"{precision:.1%}",
    delta=f"{(precision - base_rate) * 100:+.1f} pp vs test prevalence",
)
col3.metric(f"Lift@{budget_pct}%", f"{lift:.2f}x")
col4.metric(
    "Scenario value above random (EUR)",
    f"€{financial['incremental_value_added_vs_random']['incremental_net_profit']:,.0f}",
)
st.caption(
    f"The selected records contain {facts['observed_subscriptions_captured']:,} historical "
    "positive labels. The economic helper rounds random expected subscriptions to a whole "
    "number. These values are hypothetical; they are not realized profit."
)

left, right = st.columns(2)
with left:
    st.subheader("Cumulative gain")
    fractions, gains = compute_cumulative_gain_curve(labels, scores)
    gain_data = pd.DataFrame({
        "Contact capacity (%)": fractions * 100,
        "Model": gains * 100,
        "Random reference": fractions * 100,
    })
    st.line_chart(gain_data, x="Contact capacity (%)", y=["Model", "Random reference"])
    st.caption("Vertical axis: share of observed subscriptions captured (%).")
with right:
    st.subheader("Cumulative lift")
    deciles = compute_decile_lifts(labels, scores)
    lift_data = pd.DataFrame({
        "Contact capacity (%)": np.asarray(deciles["deciles"]) * 100,
        "Model": deciles["lifts"],
        "Random reference": 1.0,
    })
    st.line_chart(lift_data, x="Contact capacity (%)", y=["Model", "Random reference"])
    st.caption("Lift compares selected-record precision with overall test prevalence.")

st.subheader("Separate policy: allocate capacity within each batch")
st.caption(
    "Five consecutive groups of rows, not calendar months. Each group receives its own "
    "contact allocation at the selected capacity. These results use a different allocation "
    "policy from the global metrics above."
)
blocks = evaluate_consecutive_blocks(
    pd.Series(labels), {"selected_model": scores}, num_blocks=5, k=k
)
rows = []
for block in blocks:
    metrics = block["models"]["selected_model"]
    rows.append({
        "Batch": block["block_id"],
        "Candidates": block["n_contacts_available"],
        "Observed positive labels": block["positive_subscriptions"],
        "Positive rate": block["block_base_rate"],
        f"Recall@{budget_pct}%": metrics["recall_at_k"],
        f"Precision@{budget_pct}%": metrics["precision_at_k"],
        f"Lift@{budget_pct}%": metrics["lift_at_k"],
    })
st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

st.subheader("Interpretation")
st.info(
    "Validation favored raw CatBoost over raw logistic regression, although both were below "
    "the random reference on validation Recall@20%. Logistic regression had the better "
    "top-20% test ranking. That retrospective result does not establish a deployment choice "
    "or general superiority of linear models."
)
st.markdown(
    "**Calibration remains a limitation.** The historical positive rate changes substantially "
    "across partitions. Calibrated scores can still underestimate test prevalence, and "
    "CatBoost calibration worsened its test Brier score. See the repository methodology "
    "for probability diagnostics, feature-availability assumptions, and planned experiments."
)
