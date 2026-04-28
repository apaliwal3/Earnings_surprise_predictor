"""Model loading and prediction helpers for Stage 7 serving."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Mapping

import mlflow
import mlflow.lightgbm
import pandas as pd
import yaml


@dataclass(frozen=True)
class ServeConfig:
    """Configuration for serving-time model loading and thresholding."""

    experiment_name: str
    decision_threshold: float
    model_uri: str | None = None


@dataclass(frozen=True)
class ModelBundle:
    """Loaded model plus feature metadata."""

    model: Any
    feature_names: list[str]
    run_id: str | None = None


@dataclass(frozen=True)
class PredictionResult:
    """Single prediction response payload."""

    probability: float
    prediction: int
    threshold_used: float
    run_id: str | None
    feature_count: int


def load_params(params_path: str = "params.yaml") -> dict:
    """Load the shared parameter file."""
    with open(params_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_serve_config(params_path: str = "params.yaml") -> ServeConfig:
    """Extract serving config from params.yaml."""
    params = load_params(params_path)
    serve_cfg = params.get("serve", {})
    return ServeConfig(
        experiment_name=str(serve_cfg.get("experiment_name", "earnings-surprise-predictor")),
        decision_threshold=float(serve_cfg.get("decision_threshold", 0.5)),
        model_uri=(str(serve_cfg["model_uri"]) if serve_cfg.get("model_uri") else None),
    )


def _latest_run_model_uri(experiment_name: str) -> tuple[str, str | None]:
    """Resolve the most recent MLflow model URI for the configured experiment."""
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise FileNotFoundError(f"No MLflow experiment named '{experiment_name}' was found.")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attribute.start_time DESC"],
        max_results=1,
    )
    if runs.empty:
        raise FileNotFoundError(f"No MLflow runs found for experiment '{experiment_name}'.")

    run_id = str(runs.iloc[0]["run_id"])
    return f"runs:/{run_id}/model", run_id


def _feature_names_from_model(model: Any) -> list[str]:
    """Extract the training feature order from a loaded LightGBM model."""
    if hasattr(model, "booster_") and getattr(model, "booster_") is not None:
        try:
            return list(model.booster_.feature_name())
        except Exception:  # noqa: BLE001
            pass

    if hasattr(model, "feature_name_"):
        feature_names = getattr(model, "feature_name_")
        if feature_names is not None:
            return list(feature_names)

    raise RuntimeError("Unable to determine feature names from the loaded model.")


def load_model_bundle(config: ServeConfig | None = None) -> ModelBundle:
    """Load the latest trained model from MLflow, or a configured model URI."""
    config = config or load_serve_config()
    model_uri = config.model_uri
    run_id: str | None = None

    if not model_uri:
        model_uri, run_id = _latest_run_model_uri(config.experiment_name)

    model = mlflow.lightgbm.load_model(model_uri)
    feature_names = _feature_names_from_model(model)
    return ModelBundle(model=model, feature_names=feature_names, run_id=run_id)


@lru_cache(maxsize=1)
def get_model_bundle() -> ModelBundle:
    """Cache the model bundle so the API loads the model once per process."""
    return load_model_bundle()


def _align_features(feature_values: Mapping[str, float], feature_names: list[str]) -> pd.DataFrame:
    """Align caller-provided features to the model's expected column order."""
    frame = pd.DataFrame([feature_values], dtype=float)
    aligned = frame.reindex(columns=feature_names, fill_value=0.0)
    return aligned.astype(float)


def predict_with_threshold(
    feature_values: Mapping[str, float],
    min_confidence: float | None = None,
    bundle: ModelBundle | None = None,
    default_threshold: float = 0.5,
) -> PredictionResult:
    """Predict probability and apply a caller-controlled threshold."""
    bundle = bundle or get_model_bundle()
    threshold = default_threshold if min_confidence is None else float(min_confidence)

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("min_confidence must be between 0 and 1.")

    aligned_features = _align_features(feature_values, bundle.feature_names)
    probability = float(bundle.model.predict_proba(aligned_features)[:, 1][0])
    prediction = int(probability >= threshold)

    return PredictionResult(
        probability=probability,
        prediction=prediction,
        threshold_used=threshold,
        run_id=bundle.run_id,
        feature_count=len(bundle.feature_names),
    )
