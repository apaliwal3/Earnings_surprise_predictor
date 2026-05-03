#!/usr/bin/env bash
set -euo pipefail

# CI runner for monthly retrain pipeline.
# Usage: sets CI_START_DATE/CI_END_DATE env or accepts via params passed to download scripts.

echo "Starting retrain pipeline"

PYTHON_BIN="${PYTHON_BIN:-python}"

# If CI_END_DATE is empty, default to today's date
if [ -z "${CI_END_DATE-}" ]; then
  CI_END_DATE=$(date -u +%F)
fi

export CI_END_DATE

echo "Using CI_END_DATE=$CI_END_DATE"

if ! command -v dvc >/dev/null 2>&1; then
  echo "dvc CLI not found; install it before running this script"
  exit 1
fi

# 1) Download latest filings (assumes src/data/download_filings.py has CLI or module behavior)
if [ -f src/data/download_filings.py ]; then
  echo "Downloading filings..."
  "$PYTHON_BIN" -m src.data.download_filings --end-date "$CI_END_DATE" || true
fi

# 2) Download prices with the CI_END_DATE override
if [ -f src/data/download_prices.py ]; then
  echo "Downloading prices..."
  "$PYTHON_BIN" -m src.data.download_prices --end-date "$CI_END_DATE"
fi

# 3) Run feature fusion
if "$PYTHON_BIN" -c "import importlib.util,sys
spec = importlib.util.find_spec('src.features.fuse')
if spec is None: sys.exit(2)
"; then
  echo "Running feature fusion..."
  "$PYTHON_BIN" -m src.features.fuse
else
  echo "Feature fusion module not found; skipping"
fi

# 4) Train model
if "$PYTHON_BIN" -c "import importlib.util,sys
spec = importlib.util.find_spec('src.models.train')
if spec is None: sys.exit(2)
"; then
  echo "Training model..."
  "$PYTHON_BIN" -m src.models.train
else
  echo "Training module not found; skipping"
fi

# 5) Evaluate model
if "$PYTHON_BIN" -c "import importlib.util,sys
spec = importlib.util.find_spec('src.models.evaluate')
if spec is None: sys.exit(2)
"; then
  echo "Evaluating model..."
  "$PYTHON_BIN" -m src.models.evaluate
else
  echo "Evaluate module not found; skipping"
fi

# 5a) Check recall gate from eval metrics
if [ -f data/processed/eval_metrics.json ]; then
  echo "Checking recall gate..."
  passes_gate=$("$PYTHON_BIN" -c "import json; m=json.load(open('data/processed/eval_metrics.json')); print(int(m['passes_recall_gate']))")
  recall=$("$PYTHON_BIN" -c "import json; m=json.load(open('data/processed/eval_metrics.json')); print(f\"{m['recall']:.4f}\")")
  gate_threshold=$("$PYTHON_BIN" -c "import json; m=json.load(open('data/processed/eval_metrics.json')); print(f\"{m['recall_gate']:.4f}\")")
  
  echo "Recall: $recall, Gate threshold: $gate_threshold"
  
  if [ "$passes_gate" = "0" ]; then
    echo "ERROR: Recall $recall is below gate threshold $gate_threshold"
    exit 1
  fi
  echo "✓ Recall gate passed"
else
  echo "WARNING: eval_metrics.json not found; skipping recall gate check"
fi

# 6) Configure DVC remote if provided and push artifacts
if [ -n "${DVC_REMOTE_URL-}" ]; then
  if ! dvc remote list | awk '{print $1}' | grep -qx "ci"; then
    dvc remote add -f ci "$DVC_REMOTE_URL"
  else
    dvc remote modify ci url "$DVC_REMOTE_URL"
  fi
  dvc remote default ci
fi

if [ -n "${DVC_REMOTE_USER-}" ] && [ -n "${DVC_REMOTE_PASSWORD-}" ]; then
  # Supports HTTP remotes (for example DagsHub) in CI without storing credentials in repo.
  dvc remote modify --local ci user "$DVC_REMOTE_USER" || true
  dvc remote modify --local ci password "$DVC_REMOTE_PASSWORD" || true
fi

if dvc remote list | grep -q .; then
  echo "Committing DVC outputs..."
  dvc add data/processed/features.parquet
  git add data/processed/features.parquet.dvc dvc.lock || true

  if ! git diff --cached --quiet; then
    git commit -m "CI: update processed features for $CI_END_DATE"
  else
    echo "No DVC metadata changes to commit"
  fi

  if [ "${CI_GIT_PUSH-0}" = "1" ]; then
    git push origin HEAD
  fi

  echo "Pushing DVC data to remote..."
  dvc push
else
  echo "No DVC remote configured. Set DVC_REMOTE_URL (or preconfigure a remote) to enable dvc push."
fi

echo "Pipeline completed"
