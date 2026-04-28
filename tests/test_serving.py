"""Tests for Stage 7 serving API and threshold behavior."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from src.serving.app import app
from src.serving.forecast_inputs import resolve_feature_input
from src.serving.service import ModelBundle, get_model_bundle, predict_from_ticker_quarter, predict_with_threshold


@dataclass
class _DummyModel:
    proba: float = 0.73

    def predict_proba(self, x):
        return np.array([[1.0 - self.proba, self.proba]])


def test_predict_with_threshold_uses_override_and_aligns_missing_features():
    bundle = ModelBundle(
        model=_DummyModel(),
        feature_names=["feature_1", "feature_2", "feature_3"],
        run_id="run-123",
    )

    result = predict_with_threshold(
        feature_values={"feature_1": 1.5},
        min_confidence=0.8,
        bundle=bundle,
        default_threshold=0.5,
    )

    assert result.probability == 0.73
    assert result.prediction == 0
    assert result.threshold_used == 0.8
    assert result.run_id == "run-123"
    assert result.feature_count == 3


def test_predict_endpoint_uses_default_threshold_and_override():
    bundle = ModelBundle(
        model=_DummyModel(proba=0.73),
        feature_names=["feature_1", "feature_2"],
        run_id="run-123",
    )

    app.dependency_overrides.clear()
    app.dependency_overrides[get_model_bundle] = lambda: bundle

    client = TestClient(app)

    response = client.post("/predict", json={"features": {"feature_1": 1.0, "feature_2": 2.0}})
    assert response.status_code == 200
    assert response.json()["prediction"] == 1
    assert response.json()["threshold_used"] == 0.5

    overridden = client.post(
        "/predict",
        json={"features": {"feature_1": 1.0, "feature_2": 2.0}, "min_confidence": 0.9},
    )
    assert overridden.status_code == 200
    assert overridden.json()["prediction"] == 0
    assert overridden.json()["threshold_used"] == 0.9


def test_resolve_feature_input_supports_exact_and_next_modes():
    features_df = pd.DataFrame(
        [
            {"ticker": "AAPL", "quarter": "2024Q2", "feature_1": 1.0, "feature_2": 2.0},
            {"ticker": "AAPL", "quarter": "2024Q3", "feature_1": 3.0, "feature_2": 4.0},
            {"ticker": "AAPL", "quarter": "2025Q1", "feature_1": 5.0, "feature_2": 6.0},
        ]
    )

    exact = resolve_feature_input(features_df, ticker="AAPL", quarter="2024Q3", forecast_mode="exact")
    assert exact.target_quarter == "2024Q3"
    assert exact.as_of_quarter == "2024Q3"
    assert exact.features["feature_1"] == 3.0

    nxt = resolve_feature_input(features_df, ticker="AAPL", quarter="2025Q2", forecast_mode="next")
    assert nxt.target_quarter == "2025Q2"
    assert nxt.as_of_quarter == "2025Q1"
    assert nxt.features["feature_1"] == 5.0


def test_predict_endpoint_accepts_ticker_and_quarter_mode():
    bundle = ModelBundle(
        model=_DummyModel(proba=0.73),
        feature_names=["feature_1", "feature_2"],
        run_id="run-123",
    )

    features_df = pd.DataFrame(
        [
            {"ticker": "AAPL", "quarter": "2024Q3", "feature_1": 1.0, "feature_2": 2.0},
            {"ticker": "AAPL", "quarter": "2025Q1", "feature_1": 3.0, "feature_2": 4.0},
        ]
    )

    app.dependency_overrides.clear()
    app.dependency_overrides[get_model_bundle] = lambda: bundle
    from src.serving import service

    service.get_features_dataframe = lambda: features_df

    client = TestClient(app)

    response = client.post(
        "/predict",
        json={"ticker": "AAPL", "quarter": "2025Q2", "forecast_mode": "next"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ticker"] == "AAPL"
    assert payload["target_quarter"] == "2025Q2"
    assert payload["as_of_quarter"] == "2025Q1"
    assert payload["prediction"] == 1
