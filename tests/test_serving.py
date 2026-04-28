"""Tests for Stage 7 serving API and threshold behavior."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from fastapi.testclient import TestClient

from src.serving.app import app
from src.serving.service import ModelBundle, get_model_bundle, predict_with_threshold


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
