# Serving API Documentation

Complete reference for the Earnings Surprise Predictor FastAPI serving layer.

## Table of Contents

- [Quick Start](#quick-start)
- [API Overview](#api-overview)
- [Endpoints](#endpoints)
  - [Health Check](#health-check)
  - [Predict](#predict)
- [Request/Response Schemas](#requestresponse-schemas)
- [Prediction Modes](#prediction-modes)
- [Threshold Control](#threshold-control)
- [Examples](#examples)
- [Error Handling](#error-handling)
- [Performance](#performance)
- [Deployment](#deployment)

## Quick Start

### Start the Server

```bash
python -m src.serving.app
```

Server runs on `http://localhost:8000` by default.

### Test Health

```bash
curl http://localhost:8000/health
```

### Make a Prediction

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "AAPL",
    "quarter": "2025Q2",
    "forecast_mode": "next"
  }'
```

## API Overview

The serving layer provides:
- **Model loading** from MLflow with caching.
- **Two prediction modes**: raw features or ticker+quarter.
- **Probabilistic outputs**: caller controls decision threshold.
- **Per-request feature alignment**: missing features auto-filled with 0.0.

## Endpoints

### Health Check

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "ok"
}
```

**Use case:** Kubernetes/container readiness probes.

---

### Predict

**Endpoint:** `POST /predict`

**Description:** Score a single sample with two input modes.

**Request body:** `PredictRequest` (see schemas below)

**Response:** `PredictResponse` (see schemas below)

**Status codes:**
- `200`: Success.
- `400`: Missing required fields or malformed input.
- `500`: Model loading or scoring error.

---

## Request/Response Schemas

### PredictRequest

Input schema for predictions:

```json
{
  "features": {
    "pe_ratio": 25.5,
    "dividend_yield": 0.015,
    "embedding_0": -0.123,
    ...
  },
  "ticker": "AAPL",
  "quarter": "2025Q2",
  "forecast_mode": "next",
  "min_confidence": 0.6
}
```

**Fields:**

| Field | Type | Required? | Description |
|-------|------|-----------|-------------|
| `features` | dict | No (if ticker provided) | Feature names → numeric values. Missing features auto-filled with 0.0. |
| `ticker` | string | No (if features provided) | Ticker symbol (e.g., "AAPL"). |
| `quarter` | string | No | Target quarter (e.g., "2025Q2", "Q2 2025"). Only used with ticker mode. |
| `forecast_mode` | "exact" \| "next" | No (default: "next") | How to resolve as-of data for ticker/quarter predictions. |
| `min_confidence` | float [0, 1] | No | Override default decision threshold from params.yaml. |

**Either `features` OR `ticker` must be provided.**

### PredictResponse

Output schema:

```json
{
  "ticker": "AAPL",
  "target_quarter": "2025Q2",
  "as_of_quarter": "2024Q4",
  "forecast_mode": "next",
  "probability": 0.78,
  "prediction": 1,
  "threshold_used": 0.5,
  "run_id": "abc123xyz789",
  "feature_count": 796
}
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `ticker` | string \| null | Ticker symbol (null if raw features mode). |
| `target_quarter` | string \| null | Requested target quarter. |
| `as_of_quarter` | string \| null | Actual quarter data used (may differ from target in "next" mode). |
| `forecast_mode` | string \| null | "exact" or "next" (null if raw features mode). |
| `probability` | float | Model output: probability of beating EPS (0–1). |
| `prediction` | int | Final decision: 1 = beat, 0 = miss (based on threshold). |
| `threshold_used` | float | Decision threshold applied. |
| `run_id` | string \| null | MLflow run ID for model traceability. |
| `feature_count` | int | Number of features passed to model. |

---

## Prediction Modes

### Mode 1: Raw Features

**When:** You have pre-computed features and want a quick score.

**Request:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "pe_ratio": 22.0,
      "dividend_yield": 0.02,
      "momentum_12m": 0.15,
      "embedding_0": -0.45,
      "embedding_1": 0.23,
      ...
      "_all_ missing features default to 0_": null
    },
    "min_confidence": 0.6
  }'
```

**Response:**
```json
{
  "ticker": null,
  "target_quarter": null,
  "as_of_quarter": null,
  "forecast_mode": null,
  "probability": 0.72,
  "prediction": 1,
  "threshold_used": 0.6,
  "run_id": "mlflow_run_id",
  "feature_count": 796
}
```

**Notes:**
- All 796 model features must be present (or missing ones default to 0.0).
- Feature order does not matter (dict-based lookup).
- Threshold can be overridden per request.

### Mode 2: Ticker + Quarter (Forecast)

**When:** You want a forecast for a company's earnings in a specific quarter.

**Request:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "MSFT",
    "quarter": "2025Q3",
    "forecast_mode": "next",
    "min_confidence": 0.65
  }'
```

**Response:**
```json
{
  "ticker": "MSFT",
  "target_quarter": "2025Q3",
  "as_of_quarter": "2025Q2",
  "forecast_mode": "next",
  "probability": 0.81,
  "prediction": 1,
  "threshold_used": 0.65,
  "run_id": "mlflow_run_id",
  "feature_count": 796
}
```

**Notes:**
- System automatically fetches features from `data/processed/features.parquet`.
- `forecast_mode` determines as-of logic (see below).

#### Forecast Modes

**`forecast_mode: "exact"` (default)**
- Scores the **exact** target quarter (e.g., 2025Q3) if available in data.
- If target quarter not found: **raises 400 error**.
- Use when you want historical backtests or known-quarter analysis.

**`forecast_mode: "next"` (recommended for forward-looking predictions)**
- Finds the **latest quarter before** the target quarter in data.
- Example: target=2025Q3, latest available=2025Q2 → uses 2025Q2 features.
- Useful for real-time forecasts when full-quarter data is not yet available.
- Returns `as_of_quarter` in response showing which quarter was actually used.

---

## Threshold Control

The model outputs a **probability** (0–1). The final prediction (0 or 1) depends on a **threshold**.

### Default Threshold
```yaml
# params.yaml
serve:
  decision_threshold: 0.5
```

### Per-Request Override
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "AAPL",
    "quarter": "2025Q2",
    "min_confidence": 0.7  # Use 0.7 instead of 0.5
  }'
```

### Business Use Cases

- **Conservative (high precision)**: `min_confidence: 0.7` → only predict "beat" if model is ≥70% confident.
- **Sensitive (high recall)**: `min_confidence: 0.3` → catch more beats, accept false positives.
- **Balanced**: `min_confidence: 0.5` (default).

---

## Examples

### Example 1: Predict if Apple Will Beat Q2 2025

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "AAPL",
    "quarter": "2025Q2",
    "forecast_mode": "next"
  }' | python -m json.tool
```

**Response:**
```json
{
  "ticker": "AAPL",
  "target_quarter": "2025Q2",
  "as_of_quarter": "2024Q4",
  "forecast_mode": "next",
  "probability": 0.78,
  "prediction": 1,
  "threshold_used": 0.5,
  "run_id": "abc123",
  "feature_count": 796
}
```

**Interpretation:** 78% probability of beat. Using 2024Q4 features as latest pre-2025Q2 data. Prediction: **beat** (1).

---

### Example 2: Batch Predictions (Python)

```python
import requests
import json

API_URL = "http://localhost:8000/predict"

tickers_quarters = [
    ("AAPL", "2025Q2"),
    ("MSFT", "2025Q2"),
    ("GOOGL", "2025Q3"),
]

results = []
for ticker, quarter in tickers_quarters:
    payload = {
        "ticker": ticker,
        "quarter": quarter,
        "forecast_mode": "next",
        "min_confidence": 0.6,
    }
    resp = requests.post(API_URL, json=payload)
    results.append(resp.json())

# Filter for predicted beats
beats = [r for r in results if r["prediction"] == 1]
print(f"Predicted beats: {len(beats)}/{len(results)}")
for r in beats:
    print(f"  {r['ticker']} {r['target_quarter']}: {r['probability']:.2%}")
```

---

### Example 3: Raw Feature Prediction

```bash
# Imagine features pre-computed in your pipeline
FEATURES='{
  "pe_ratio": 28.5,
  "dividend_yield": 0.015,
  "revenue_growth": 0.12,
  "earnings_volatility": 0.18,
  "ebitda_margin": 0.35,
  "embedding_0": -0.234,
  "embedding_1": 0.567,
  ...
  "embedding_767": -0.089
}'

curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d "{\"features\": $FEATURES, \"min_confidence\": 0.55}"
```

---

## Error Handling

### Missing Required Fields

**Request:**
```json
{
  "min_confidence": 0.6
}
```

**Response (400):**
```json
{
  "detail": "Provide either 'features' or 'ticker' for quarter-aware prediction."
}
```

---

### Quarter Not Found (exact mode)

**Request:**
```json
{
  "ticker": "AAPL",
  "quarter": "2030Q1",
  "forecast_mode": "exact"
}
```

**Response (400):**
```json
{
  "detail": "Quarter 2030Q1 not found in features data."
}
```

**Solution:** Use `forecast_mode: "next"` instead.

---

### Model Loading Error

**Response (500):**
```json
{
  "detail": "Failed to load model artifact from MLflow."
}
```

**Troubleshooting:**
- Check MLflow artifact server is running: `mlflow server --backend-store-uri sqlite:///mlflow.db`
- Verify model artifact path in MLflow runs.
- Ensure `dvc pull` has downloaded all artifacts.

---

## Performance

### Latency

- **Raw features mode**: ~50–100ms (model inference only).
- **Ticker+quarter mode**: ~100–200ms (includes feature lookup).

### Throughput

- Single-threaded: ~10–20 predictions/sec.
- Multi-worker deployment (Uvicorn with workers): scale linearly.

### Resource Requirements

- **Memory**: ~2–3 GB (model + data cache).
- **CPU**: 1–2 cores sufficient; GPU optional (model uses CPU by default).

---

## Deployment

### Docker

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
ENV PYTHONUNBUFFERED=1

EXPOSE 8000
CMD ["python", "-m", "src.serving.app"]
```

**Build and run:**
```bash
docker build -t earnings-predictor .
docker run -p 8000:8000 earnings-predictor
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: earnings-predictor
spec:
  replicas: 3
  selector:
    matchLabels:
      app: earnings-predictor
  template:
    metadata:
      labels:
        app: earnings-predictor
    spec:
      containers:
      - name: app
        image: earnings-predictor:latest
        ports:
        - containerPort: 8000
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

### Cloud Platforms

- **AWS ECS/Fargate**: Use Docker image; container port 8000.
- **Google Cloud Run**: Same Docker image; set PORT=8000 env var.
- **Azure Container Instances**: Deploy Docker image and expose port 8000.

---

## Configuration

Model behavior controlled by `params.yaml`:

```yaml
serve:
  experiment_name: "earnings-surprise-predictor"  # MLflow experiment
  decision_threshold: 0.5                         # Default threshold
```

Override at runtime:
- Per-request: `min_confidence` in payload.
- Environment: (not yet supported; add if needed).

---

## Troubleshooting

### Server won't start
```bash
# Check if port 8000 is in use
lsof -i :8000
# Kill existing process or use different port
python -m src.serving.app --port 8001
```

### Model predictions are inconsistent
- Ensure `dvc pull` has latest artifacts.
- Check MLflow experiment name matches `params.yaml`.

### Features not aligned
- Verify feature names match model training exactly.
- Missing features auto-fill with 0.0 (check if this is intended).

### Ticker not found
- Confirm ticker exists in `data/processed/features.parquet`.
- Check ticker is normalized (uppercase, no spaces).

---

## API Reference

### Health

```
GET /health
```

**Response:**
```json
{"status": "ok"}
```

---

### Predict

```
POST /predict
Content-Type: application/json
```

**Request:**
```json
{
  "features": {...},
  "ticker": "AAPL",
  "quarter": "2025Q2",
  "forecast_mode": "next",
  "min_confidence": 0.6
}
```

**Response:**
```json
{
  "ticker": "AAPL",
  "target_quarter": "2025Q2",
  "as_of_quarter": "2024Q4",
  "forecast_mode": "next",
  "probability": 0.78,
  "prediction": 1,
  "threshold_used": 0.5,
  "run_id": "abc123",
  "feature_count": 796
}
```

---

## Support

For issues, feature requests, or questions:
1. Check README.md for project overview.
2. Review error responses and troubleshooting section.
3. Open a GitHub issue with request payload and response.

---

**Last updated:** April 2026
