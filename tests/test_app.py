"""Exercise changes in contact capacity using known historical labels and scores."""

from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest


def test_capacity_changes_use_observed_scores(monkeypatch):
    # At 20%, both selected records are positive; at 50%, four of five are.
    benchmark = pd.DataFrame({
        "y_true": [1, 1, 0, 1, 1, 0, 0, 0, 0, 0],
        "prob_lr_cal": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
    })
    for column in ("prob_lr_raw", "prob_cb_raw", "prob_cb_cal"):
        benchmark[column] = benchmark["prob_lr_cal"]
    monkeypatch.setattr(pd, "read_csv", lambda *args, **kwargs: benchmark.copy())
    # The test supplies predictions even in a checkout without saved artifacts.
    original_exists = Path.exists
    monkeypatch.setattr(
        Path, "exists",
        lambda path: True if path.name == "test_row_predictions.csv" else original_exists(path),
    )

    app_path = Path(__file__).resolve().parents[1] / "app/main.py"
    app = AppTest.from_file(app_path, default_timeout=15).run()
    assert not app.exception
    assert app.metric[0].value == "50.0%"
    assert app.metric[1].value == "100.0%"

    app.slider(key="capacity").set_value(50).run()
    assert not app.exception
    assert app.metric[0].value == "100.0%"
    assert app.metric[1].value == "80.0%"
    assert app.dataframe[0].value.columns.tolist()[-3:] == [
        "Recall@50%", "Precision@50%", "Lift@50%"
    ]

    app.selectbox(key="model").select("CatBoost (raw)").run()
    assert not app.exception
    assert app.metric[1].value == "80.0%"
