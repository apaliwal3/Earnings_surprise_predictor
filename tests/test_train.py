"""Tests for Stage 5 training module."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.train import split_xy, train_model


def _toy_features(n_rows: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    df = pd.DataFrame(
        {
            "ticker": ["AAPL"] * n_rows,
            "quarter": [f"2020Q{(i % 4) + 1}" for i in range(n_rows)],
            "feature_1": rng.normal(size=n_rows),
            "feature_2": rng.normal(size=n_rows),
            "feature_3": rng.normal(size=n_rows),
            "label": rng.integers(0, 2, size=n_rows),
        }
    )
    return df


def test_split_xy_drops_identifier_columns():
    df = _toy_features()
    x, y = split_xy(df)

    assert "ticker" not in x.columns
    assert "quarter" not in x.columns
    assert "label" not in x.columns
    assert y.name == "label"
    assert len(x) == len(y)


def test_train_model_returns_metrics_and_model():
    df = _toy_features()
    params = {
        "model": {
            "num_leaves": 31,
            "learning_rate": 0.05,
            "n_estimators": 50,
            "max_depth": -1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
        "train": {"test_size": 0.2, "random_state": 42},
    }

    model, metrics = train_model(df, params)

    assert hasattr(model, "predict_proba")
    for metric in ["pr_auc", "roc_auc", "precision", "recall", "n_train", "n_test", "n_features"]:
        assert metric in metrics
