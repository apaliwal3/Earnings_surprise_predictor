"""Tests for Stage 6 evaluation module."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.models.evaluate import evaluate_model, run_evaluation


def _toy_features(n_rows: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    return pd.DataFrame(
        {
            "ticker": ["AAPL"] * n_rows,
            "quarter": [f"2020Q{(i % 4) + 1}" for i in range(n_rows)],
            "feature_1": rng.normal(size=n_rows),
            "feature_2": rng.normal(size=n_rows),
            "feature_3": rng.normal(size=n_rows),
            "label": rng.integers(0, 2, size=n_rows),
        }
    )


def _params() -> dict:
    return {
        "model": {
            "num_leaves": 31,
            "learning_rate": 0.05,
            "n_estimators": 50,
            "max_depth": -1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
        "train": {"test_size": 0.2, "random_state": 42},
        "eval": {"recall_gate": 0.55},
    }


def test_evaluate_model_returns_expected_metrics_and_predictions():
    df = _toy_features()
    metrics, predictions = evaluate_model(df, _params())

    for key in [
        "pr_auc",
        "roc_auc",
        "precision",
        "recall",
        "f1",
        "accuracy",
        "tn",
        "fp",
        "fn",
        "tp",
        "recall_gate",
        "passes_recall_gate",
        "n_test",
    ]:
        assert key in metrics

    assert set(predictions.columns) == {"y_true", "y_pred", "y_proba"}
    assert len(predictions) == metrics["n_test"]


def test_run_evaluation_writes_artifacts(tmp_path):
    features_path = tmp_path / "features.parquet"
    _toy_features().to_parquet(features_path, index=False)

    metrics = run_evaluation(features_path=str(features_path), output_dir=str(tmp_path))

    metrics_path = tmp_path / "eval_metrics.json"
    preds_path = tmp_path / "eval_predictions.parquet"

    assert metrics_path.exists()
    assert preds_path.exists()

    with open(metrics_path, "r", encoding="utf-8") as f:
        loaded_metrics = json.load(f)

    assert loaded_metrics["n_test"] == metrics["n_test"]
    preds = pd.read_parquet(preds_path)
    assert len(preds) == metrics["n_test"]
