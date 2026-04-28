# Earnings Surprise Predictor

A machine learning pipeline to predict whether a company will beat or miss earnings forecasts using 10-Q filings and market data. The system combines NLP embeddings (FinBERT) with tabular financial features in a LightGBM classifier, includes a FastAPI serving layer with per-request threshold control, and features an automated monthly retraining pipeline via GitHub Actions.

## Table of Contents

- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Data Pipeline](#data-pipeline)
- [Model Architecture](#model-architecture)
- [Serving API](#serving-api)
- [CLI Usage](#cli-usage)
- [CI/CD Pipeline](#cicd-pipeline)
- [Configuration](#configuration)
- [Development](#development)

## Quick Start

### Setup

1. **Clone and install dependencies:**
   ```bash
   git clone https://github.com/yourusername/earnings_surprise_predictor.git
   cd earnings_surprise_predictor
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Configure DVC (Data Version Control):**
   ```bash
   dvc remote add dagshub https://dagshub.com/apaliwal3/Earnings_surprise.dvc
   dvc remote default dagshub
   dvc pull  # Download tracked data/model artifacts
   ```

3. **Set up SEC Edgar access:**
   ```bash
   export SEC_EMAIL="your-email@example.com"
   ```

### Train and Evaluate

```bash
# Run the full pipeline (download data, fuse features, train, evaluate)
python -m src.models.train
python -m src.models.evaluate
```

### Serve Predictions

```bash
# Start the FastAPI server
python -m src.serving.app

# In another terminal, make a prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "AAPL",
    "quarter": "2025Q2",
    "forecast_mode": "next",
    "min_confidence": 0.6
  }'
```

See [SERVING.md](SERVING.md) for detailed API documentation.

## Project Structure

```
earnings_surprise_predictor/
├── .github/
│   ├── workflows/
│   │   └── retrain.yml           # GitHub Actions monthly retrain workflow
│   └── ci/
│       └── run_pipeline.sh       # CI runner script (download, train, eval, push)
├── data/
│   ├── raw/
│   │   ├── filings/              # 10-Q documents (DVC-tracked)
│   │   └── prices/               # Historical stock prices
│   └── processed/
│       ├── features.parquet      # Fused tabular + embedding features (DVC)
│       ├── eval_metrics.json     # Evaluation metrics + recall gate check
│       └── eval_predictions.parquet
├── src/
│   ├── data/
│   │   ├── download_filings.py   # SEC Edgar 10-Q downloader
│   │   ├── download_prices.py    # Yahoo Finance price downloader
│   │   └── validate.py           # Data validation
│   ├── features/
│   │   ├── tabular.py            # Financial ratio engineering
│   │   ├── text_embeddings.py    # FinBERT embeddings from filings
│   │   └── fuse.py               # Merge tabular + embeddings
│   ├── models/
│   │   ├── train.py              # LightGBM training & MLflow logging
│   │   └── evaluate.py           # Model eval with recall gate
│   └── serving/
│       ├── app.py                # FastAPI endpoints
│       ├── service.py            # Prediction logic & model loading
│       └── forecast_inputs.py    # Quarter resolution for forecasts
├── scripts/
│   ├── ticker_predict.py         # CLI to predict for a ticker/quarter
│   └── example_predict.py        # Example E2E prediction script
├── tests/
│   └── test_serving.py           # Unit tests for API
├── params.yaml                   # Single source of truth for all config
├── dvc.yaml                      # DVC pipeline definition
├── dvc.lock                      # DVC lock file (data artifacts)
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── SERVING.md                    # API documentation
```

## Data Pipeline

### Stage 1–3: Data Acquisition & Validation

1. **Download filings** (`src/data/download_filings.py`):
   - Fetches 10-Q filings from SEC Edgar for S&P 500 companies.
   - Extracts and flattens full-submission text files.
   - Supports date-range overrides via `--start-date` and `--end-date` CLI args or `CI_START_DATE`/`CI_END_DATE` env vars.

2. **Download prices** (`src/data/download_prices.py`):
   - Pulls historical stock prices and earnings event dates from yfinance.
   - Same CLI/env date override support.

3. **Validate** (`src/data/validate.py`):
   - Ensures filing and price data consistency.

### Stage 4–5: Feature Engineering

4. **Tabular features** (`src/features/tabular.py`):
   - Computes financial ratios (e.g., P/E, debt-to-equity, earnings volatility).
   - Each row: (ticker, quarter, ratios, label).

5. **Text embeddings** (`src/features/text_embeddings.py`):
   - Uses FinBERT to embed 10-Q document sections.
   - Pools embeddings into a single 768-dim vector per filing.
   - Batch processing with GPU/CPU fallback.

6. **Fuse** (`src/features/fuse.py`):
   - Aligns filings to quarters using filing dates.
   - Merges tabular features and embeddings.
   - Output: `data/processed/features.parquet` (15k+ rows × 796 features).

### Stage 6: Training & Evaluation

7. **Train** (`src/models/train.py`):
   - LightGBM binary classifier on fused features.
   - Logs model, metrics, and params to MLflow.
   - Outputs: model artifact, training metrics.

8. **Evaluate** (`src/models/evaluate.py`):
   - Test-set evaluation with confusion matrix, ROC-AUC, precision, recall, F1.
   - **Recall gate check**: fails if recall < threshold (default 0.55).
   - Outputs: `eval_metrics.json`, `eval_predictions.parquet`.

## Model Architecture

**Input:**
- Tabular features: financial ratios (e.g., P/E, dividend yield, volatility).
- Text embeddings: 768-dim FinBERT vectors from 10-Q disclosures.
- Total: 796 features.

**Model:**
- LightGBM classifier with 300 estimators.
- Binary output: 1 (beat EPS estimate), 0 (miss).

**Output:**
- Probability of beating earnings (0–1).
- Prediction: threshold-dependent (default 0.5, customizable at serving time).

**Quality Gate:**
- Recall ≥ 0.55 (catch at least 55% of beats).
- Pipeline fails CI if gate is not met.

## Serving API

Start the server with:
```bash
python -m src.serving.app
```

### Endpoints

#### `/health` (GET)
Simple readiness check.
```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

#### `/predict` (POST)
Score a single prediction with two input modes.

**Mode 1: Raw Features**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "pe_ratio": 25.5,
      "dividend_yield": 0.015,
      "embedding_0": -0.123,
      "embedding_1": 0.456,
      ...
    },
    "min_confidence": 0.6
  }'
```

**Mode 2: Ticker + Quarter (Forecast)**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "AAPL",
    "quarter": "2025Q2",
    "forecast_mode": "next",
    "min_confidence": 0.7
  }'
```

See [SERVING.md](SERVING.md) for full request/response schemas and examples.

## CLI Usage

### Predict for a Ticker/Quarter

```bash
python scripts/ticker_predict.py AAPL "2025Q2" --mode "next" --threshold 0.6
```

Options:
- `--mode exact|next`: `exact` scores the requested quarter; `next` uses the latest available pre-target quarter.
- `--threshold`: Override decision threshold (default 0.5 from params.yaml).

Output: probability, prediction (0/1), as-of-quarter used.

### Run Example End-to-End

```bash
python scripts/example_predict.py
```

Demonstrates loading model, making predictions on sample data, and interpreting results.

## CI/CD Pipeline

### Automatic Monthly Retrain

GitHub Actions runs on the **1st of each month at 06:00 UTC** (configurable in [.github/workflows/retrain.yml](.github/workflows/retrain.yml)).

**Steps:**
1. Download latest 10-Q filings and prices (end_date = today).
2. Fuse features.
3. Train model.
4. Evaluate and check recall gate.
5. If gate passes:
   - Commit `dvc.lock` and push to git branch.
   - Push DVC artifacts to DagsHub remote.
6. If gate fails: **pipeline stops**, no commits/pushes.

### Manual Trigger

Go to **Actions → Monthly Retrain → Run workflow** and optionally override `end_date` (YYYY-MM-DD format).

### Setup Required

**GitHub Secrets** (Settings → Secrets and variables → Actions):
- `SEC_EMAIL`: Your email for SEC Edgar (required).
- `DVC_REMOTE_URL`: (optional) DVC remote URL if not default.
- `DVC_REMOTE_USER`, `DVC_REMOTE_PASSWORD`: (optional) For HTTP remote auth.

### Local Dry-Run

Test the pipeline locally:
```bash
export SEC_EMAIL="your-email@example.com"
export CI_END_DATE="2026-04-28"
bash .github/ci/run_pipeline.sh
```

## Configuration

All configuration lives in **`params.yaml`** (single source of truth):

```yaml
data:
  start_date: "2015-01-01"
  end_date: "2024-12-31"           # Updated by monthly CI runs
  label_threshold: 0.02             # EPS surprise threshold for positive label

text:
  batch_size: 32                    # FinBERT embedding batch size
  max_length: 512                   # Max tokens per filing section

model:
  num_leaves: 31                    # LightGBM hyperparameter
  learning_rate: 0.05
  n_estimators: 300
  max_depth: -1
  subsample: 0.8
  colsample_bytree: 0.8

train:
  test_size: 0.2
  random_state: 42

eval:
  recall_gate: 0.55                 # Quality gate: fail if recall < this

serve:
  experiment_name: "earnings-surprise-predictor"
  decision_threshold: 0.5           # Default threshold for predictions
```

**Override at runtime:**
- Download scripts accept `--start-date` and `--end-date` CLI args.
- Serving API accepts `min_confidence` per request.
- CI sets `CI_END_DATE` env var for monthly runs.

## Development

### Run Tests

```bash
pytest tests/ -v
```

### Add a New Feature

1. Implement in `src/features/` (e.g., `new_feature.py`).
2. Call from `src/features/fuse.py`.
3. Update `params.yaml` if new hyperparameters.
4. Commit and push; CI will auto-retrain.

### Modify Model Hyperparameters

Edit `params.yaml` → commit → CI automatically retrains.

### Update Data Ranges

Edit `data.end_date` in `params.yaml` or pass `CI_END_DATE` to CI; monthly runs will pick up the new date.

## Troubleshooting

### DVC `dvc push` fails in CI
- Check GitHub Secrets: `DVC_REMOTE_URL` and auth credentials.
- Ensure local `.dvc/config` has a default remote configured.

### SEC Edgar download errors
- Verify `SEC_EMAIL` environment variable is set.
- Check SEC Edgar server status (sometimes rate-limited).

### Model predictions are all zeros/ones
- Check feature alignment in `src/serving/forecast_inputs.py`.
- Verify model artifact exists in MLflow (run `dvc pull`).

### Recall gate failures blocking CI
- Check `data/processed/eval_metrics.json` for actual recall value.
- Adjust `eval.recall_gate` in `params.yaml` if change is intentional.

## License

MIT License — see LICENSE file for details.

## Contributing

1. Create a feature branch from `main`.
2. Make changes and test locally.
3. Open a pull request describing your changes.
4. CI runs automated tests; approval required before merge.

---

**Questions or issues?** See [SERVING.md](SERVING.md) for API details or open a GitHub issue.
