"""FastAPI app for serving predictions."""

from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.serving.service import PredictionResult, get_model_bundle, load_serve_config, predict_with_threshold

app = FastAPI(title="Earnings Surprise Predictor", version="7.0.0")
serve_config = load_serve_config()


class PredictRequest(BaseModel):
    """Request payload for /predict."""

    features: dict[str, float] = Field(..., description="Feature names mapped to numeric values.")
    min_confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional caller-controlled decision threshold.",
    )


class PredictResponse(BaseModel):
    """Response payload for /predict."""

    probability: float
    prediction: int
    threshold_used: float
    run_id: str | None = None
    feature_count: int


@app.get("/health")
def health() -> dict[str, str]:
    """Simple health check for readiness probes."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(
    request: PredictRequest,
    bundle=Depends(get_model_bundle),
) -> PredictResponse:
    """Score a single feature payload and apply the configured threshold."""
    try:
        result: PredictionResult = predict_with_threshold(
            feature_values=request.features,
            min_confidence=request.min_confidence,
            bundle=bundle,
            default_threshold=serve_config.decision_threshold,
        )
        return PredictResponse(**result.__dict__)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
