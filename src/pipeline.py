"""End-to-end training, calibration, evaluation, and artifact generation pipeline."""

import os
import json
import yaml
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

from src.data.loader import load_raw_dataset
from src.data.split import sequential_temporal_split, get_split_summary
from src.metrics.top_k import (
    compute_all_metrics,
    compute_cumulative_gain_curve,
    compute_decile_lifts,
)
from src.metrics.economic import simulate_campaign_economics
from src.evaluation.block_eval import evaluate_consecutive_blocks
from src.evaluation.prediction_export import build_prediction_export
from src.models.logistic import LogisticRegressionModel
from src.models.catboost_model import CatBoostModel


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Loads configuration YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def plot_cumulative_gains(
    y_test: np.ndarray,
    probs_dict: dict,
    k_target: float = 0.20,
    save_path: str = "artifacts/plots/cumulative_gain.png",
):
    """Plots and saves Cumulative Gains Curves for all models vs Random."""
    plt.figure(figsize=(8, 6), dpi=150)

    # Random baseline
    percentages = np.linspace(0, 1.0, 100)
    plt.plot(percentages * 100, percentages * 100, "k--", label="Random selection (expected)")

    colors = {"Logistic Regression (Calibrated)": "#1f77b4", "CatBoost (Calibrated)": "#2ca02c"}

    for name, probs in probs_dict.items():
        pcts, gains = compute_cumulative_gain_curve(y_test, probs)
        color = colors.get(name, None)
        plt.plot(pcts * 100, gains * 100, label=name, lw=2.2, color=color)

    # Vertical line at budget k
    plt.axvline(x=k_target * 100, color="red", linestyle=":", lw=1.8, label=f"Contact capacity ({int(k_target*100)}%)")

    plt.title("Cumulative gain: observed subscriptions captured", fontsize=13, pad=12)
    plt.xlabel("Candidates selected by score (%)", fontsize=11)
    plt.ylabel("Observed subscriptions captured (%)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def plot_calibration_curves(
    y_test: np.ndarray,
    models_dict: dict,
    save_path: str = "artifacts/plots/calibration_curve.png",
):
    """Plots reliability diagrams (predicted prob vs observed fraction of positives)."""
    plt.figure(figsize=(8, 6), dpi=150)
    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")

    for name, probs in models_dict.items():
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_test, probs, n_bins=10, strategy="uniform"
        )
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", label=name, lw=1.8)

    plt.title("Probability calibration (reliability diagram)", fontsize=13, pad=12)
    plt.xlabel("Mean predicted probability", fontsize=11)
    plt.ylabel("Observed positive fraction", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper left", frameon=True)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def plot_decile_lift(
    decile_dict: dict,
    save_path: str = "artifacts/plots/lift_curve.png",
):
    """Plots Lift across deciles."""
    plt.figure(figsize=(8, 5), dpi=150)
    deciles = [int(d * 100) for d in decile_dict[list(decile_dict.keys())[0]]["deciles"]]

    x = np.arange(len(deciles))
    width = 0.35

    model_names = list(decile_dict.keys())
    plt.bar(x - width/2, decile_dict[model_names[0]]["lifts"], width=width, label=model_names[0], color="#1f77b4")
    if len(model_names) > 1:
        plt.bar(x + width/2, decile_dict[model_names[1]]["lifts"], width=width, label=model_names[1], color="#2ca02c")

    plt.axhline(y=1.0, color="black", linestyle="--", label="Random reference (lift = 1.0)")
    plt.xticks(x, [f"Decile {i+1} ({d}%)" for i, d in enumerate(deciles)], rotation=25)
    plt.title("Cumulative lift by contact capacity", fontsize=13, pad=12)
    plt.ylabel("Lift (Precision / Base Rate)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.4, axis="y")
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def run_pipeline(config_path: str = "config/config.yaml"):
    """Executes full pipeline and writes artifacts."""
    print("=" * 60)
    print("STARTING PIPELINE: BANK CAMPAIGN PRIORITIZER")
    print("=" * 60)

    cfg = load_config(config_path)

    # 1. Load data and verify leakage exclusion
    print("\n[1/6] Loading data and excluding post-contact features...")
    exclude_cols = cfg["features"]["exclude"]
    X, y = load_raw_dataset(
        filepath=cfg["data"]["raw_path"],
        delimiter=cfg["data"]["delimiter"],
        target_col=cfg["data"]["target_col"],
        positive_val=cfg["data"]["target_positive_val"],
        exclude_cols=exclude_cols,
    )
    print(f"-> Rows loaded: {len(X):,}. Features in X: {list(X.columns)}")
    assert "duration" not in X.columns, "CRITICAL ERROR: 'duration' leakage detected in feature matrix!"

    # 2. Sequential temporal split
    print("\n[2/6] Splitting consecutive rows using configured partition shares...")
    splits = sequential_temporal_split(
        X,
        y,
        train_pct=cfg["split"]["train_pct"],
        selection_pct=cfg["split"]["selection_pct"],
        calibration_pct=cfg["split"]["calibration_pct"],
        test_pct=cfg["split"]["test_pct"],
    )
    split_summary = get_split_summary(splits)
    for part, stats in split_summary.items():
        print(f"   * {part.capitalize()}: {stats['n_samples']} rows ({stats['pct_of_total']}%), Positive rate: {stats['positive_rate']*100:.2f}%")

    X_train, y_train = splits["train"]
    X_sel, y_sel = splits["selection"]
    X_cal, y_cal = splits["calibration"]
    X_test, y_test = splits["test"]

    cat_cols = cfg["features"]["categorical"]
    num_cols = cfg["features"]["numerical"]

    # 3. Model training
    print("\n[3/6] Training logistic regression and CatBoost...")
    lr = LogisticRegressionModel(cat_cols, num_cols, cfg["models"]["logistic_regression"])
    lr.fit(X_train, y_train)

    cb = CatBoostModel(cat_cols, num_cols, cfg["models"]["catboost"])
    cb.fit(X_train, y_train, X_sel, y_sel)

    k_target = cfg["economics"]["target_budget_pct"]
    selection_metrics = {
        "Logistic Regression (Raw)": compute_all_metrics(
            np.asarray(y_sel), lr.predict_proba(X_sel), k=k_target
        ),
        "CatBoost (Raw)": compute_all_metrics(
            np.asarray(y_sel), cb.predict_proba(X_sel), k=k_target
        ),
    }

    # 4. Calibration
    print("\n[4/6] Fitting sigmoid calibrators on the calibration partition...")
    lr.calibrate(X_cal, y_cal, method="sigmoid")
    cb.calibrate(X_cal, y_cal, method="sigmoid")

    # 5. Predictions on Test
    k_target = cfg["economics"]["target_budget_pct"]
    y_test_arr = np.asarray(y_test)

    preds = {
        "Logistic Regression (Raw)": lr.predict_proba(X_test, calibrated=False),
        "Logistic Regression (Calibrated)": lr.predict_proba(X_test, calibrated=True),
        "CatBoost (Raw)": cb.predict_proba(X_test, calibrated=False),
        "CatBoost (Calibrated)": cb.predict_proba(X_test, calibrated=True),
    }

    # Evaluate all models
    print("\n[5/6] Evaluating held-out test metrics...")
    model_metrics = {}
    for name, p in preds.items():
        metrics = compute_all_metrics(y_test_arr, p, k=k_target)
        model_metrics[name] = metrics
        print(f"-> {name}: Recall@20% = {metrics['recall_at_20pct']:.4f} | Precision@20% = {metrics['precision_at_20pct']:.4f} | Lift@20% = {metrics['lift_at_20pct']:.2f} | Brier = {metrics['brier_score']:.4f} | Average precision = {metrics['pr_auc']:.4f}")

    # Random baseline metrics
    base_rate_test = float(np.mean(y_test_arr == 1))
    random_metrics = {
        "recall_at_20pct": round(k_target, 4),
        "precision_at_20pct": round(base_rate_test, 4),
        "lift_at_20pct": 1.0,
        "pr_auc": round(base_rate_test, 4),
        "roc_auc": 0.5000,
        "brier_score": round(float(np.mean((base_rate_test - y_test_arr)**2)), 4),
        "base_rate": round(base_rate_test, 4),
    }
    model_metrics["Random Selection (Baseline)"] = random_metrics

    # Economic simulation
    print("\n[6/6] Computing economic scenarios and consecutive-batch metrics...")
    margin = cfg["economics"]["net_margin_per_subscription"]
    cost = cfg["economics"]["contact_cost"]

    econ_lr = simulate_campaign_economics(
        y_test_arr, preds["Logistic Regression (Calibrated)"], k=k_target, margin_per_subscription=margin, contact_cost=cost
    )
    econ_cb = simulate_campaign_economics(
        y_test_arr, preds["CatBoost (Calibrated)"], k=k_target, margin_per_subscription=margin, contact_cost=cost
    )

    # Block evaluation
    num_blocks = cfg["evaluation"]["num_consecutive_blocks"]
    block_eval = evaluate_consecutive_blocks(
        y_test,
        {
            "logistic_calibrated": preds["Logistic Regression (Calibrated)"],
            "catboost_calibrated": preds["CatBoost (Calibrated)"],
        },
        num_blocks=num_blocks,
        k=k_target,
    )

    # Decile lifts
    decile_lifts = {
        "Logistic Regression": compute_decile_lifts(y_test_arr, preds["Logistic Regression (Calibrated)"]),
        "CatBoost": compute_decile_lifts(y_test_arr, preds["CatBoost (Calibrated)"]),
    }

    # Generate plots
    plot_cumulative_gains(
        y_test_arr,
        {
            "Logistic Regression (Calibrated)": preds["Logistic Regression (Calibrated)"],
            "CatBoost (Calibrated)": preds["CatBoost (Calibrated)"],
        },
        k_target=k_target,
    )
    plot_calibration_curves(
        y_test_arr,
        {
            "Logistic (Raw)": preds["Logistic Regression (Raw)"],
            "Logistic (Calibrated)": preds["Logistic Regression (Calibrated)"],
            "CatBoost (Raw)": preds["CatBoost (Raw)"],
            "CatBoost (Calibrated)": preds["CatBoost (Calibrated)"],
        },
    )
    plot_decile_lift(decile_lifts)

    # Save artifacts
    os.makedirs("artifacts/metrics", exist_ok=True)
    os.makedirs("artifacts/models", exist_ok=True)
    os.makedirs("artifacts/predictions", exist_ok=True)

    prediction_export = build_prediction_export(
        y_test,
        {
            "prob_lr_raw": preds["Logistic Regression (Raw)"],
            "prob_lr_cal": preds["Logistic Regression (Calibrated)"],
            "prob_cb_raw": preds["CatBoost (Raw)"],
            "prob_cb_cal": preds["CatBoost (Calibrated)"],
        },
        num_blocks=num_blocks,
        k=k_target,
    )
    prediction_export.to_csv("artifacts/predictions/test_row_predictions.csv", index=False)

    summary_output = {
        "split_summary": split_summary,
        "selection_metrics_raw": selection_metrics,
        "test_probability_means": {name: float(p.mean()) for name, p in preds.items()},
        "metrics_summary": model_metrics,
        "economic_simulation": {
            "logistic_regression": econ_lr,
            "catboost": econ_cb,
        },
    }

    with open("artifacts/metrics/run_summary.json", "w") as f:
        json.dump(summary_output, f, indent=2)

    with open("artifacts/metrics/block_evaluation.json", "w") as f:
        json.dump(block_eval, f, indent=2)

    joblib.dump(lr, "artifacts/models/logistic_regression_pipeline.joblib")
    joblib.dump(cb, "artifacts/models/catboost_model.joblib")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED")
    print("Reports: artifacts/metrics/; predictions: artifacts/predictions/; plots: artifacts/plots/")
    print("=" * 60)

    return summary_output


if __name__ == "__main__":
    run_pipeline()
