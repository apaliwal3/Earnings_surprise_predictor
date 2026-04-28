"""FastAPI app for serving predictions."""

from __future__ import annotations

from typing import Literal

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.serving.service import (
    ForecastPredictionResult,
    PredictionResult,
    get_model_bundle,
    load_serve_config,
    predict_from_ticker_quarter,
    predict_with_threshold,
)

app = FastAPI(title="Earnings Surprise Predictor", version="7.0.0")
serve_config = load_serve_config()


class PredictRequest(BaseModel):
    """Request payload for /predict."""

    features: dict[str, float] | None = Field(
        default=None,
        description="Feature names mapped to numeric values.",
    )
    ticker: str | None = Field(
        default=None,
        description="Ticker symbol for quarter-aware predictions.",
    )
    quarter: str | None = Field(
        default=None,
        description="Target quarter such as 2025Q2 or Q2 2025.",
    )
    forecast_mode: Literal["exact", "next"] = Field(
        default="next",
        description="exact scores the requested quarter; next uses the latest available pre-target quarter.",
    )
    min_confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional caller-controlled decision threshold.",
    )


class PredictResponse(BaseModel):
    """Response payload for /predict."""

    ticker: str | None = None
    target_quarter: str | None = None
    as_of_quarter: str | None = None
    forecast_mode: str | None = None
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
        if request.features is not None:
            result: PredictionResult = predict_with_threshold(
                feature_values=request.features,
                min_confidence=request.min_confidence,
                bundle=bundle,
                default_threshold=serve_config.decision_threshold,
            )
            return PredictResponse(**result.__dict__)

        if not request.ticker:
            raise HTTPException(
                status_code=400,
                detail="Provide either 'features' or 'ticker' for quarter-aware prediction.",
            )

        result: ForecastPredictionResult = predict_from_ticker_quarter(
            ticker=request.ticker,
            quarter=request.quarter,
            forecast_mode=request.forecast_mode,
            min_confidence=request.min_confidence,
            bundle=bundle,
            default_threshold=serve_config.decision_threshold,
        )
        return PredictResponse(**result.__dict__)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
