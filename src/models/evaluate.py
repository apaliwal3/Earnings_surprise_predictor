"""Stage 6: Evaluate model quality and enforce recall gate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import lightgbm as lgb
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.models.train import split_xy


def load_params(params_path: str = "params.yaml") -> dict:
    """Load model/train/eval params from params.yaml."""
    with open(params_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_features(features_path: str = "data/processed/features.parquet") -> pd.DataFrame:
    """Load fused feature matrix for evaluation."""
    path = Path(features_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing features file: {features_path}")
    return pd.read_parquet(path)


def _split_data(df: pd.DataFrame, params: dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Build deterministic train/test split matching Stage 5 settings."""
    x, y = split_xy(df)

    train_cfg = params.get("train", {})
    test_size = float(train_cfg.get("test_size", 0.2))
    random_state = int(train_cfg.get("random_state", 42))

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if y.nunique() > 1 else None,
    )
    return x_train, x_test, y_train, y_test


def _build_model(params: dict) -> lgb.LGBMClassifier:
    """Build LightGBM model from params.yaml model section."""
    model_cfg = params.get("model", {})
    train_cfg = params.get("train", {})

    return lgb.LGBMClassifier(
        objective="binary",
        num_leaves=int(model_cfg.get("num_leaves", 31)),
        learning_rate=float(model_cfg.get("learning_rate", 0.05)),
        n_estimators=int(model_cfg.get("n_estimators", 300)),
        max_depth=int(model_cfg.get("max_depth", -1)),
        subsample=float(model_cfg.get("subsample", 0.8)),
        colsample_bytree=float(model_cfg.get("colsample_bytree", 0.8)),
        random_state=int(train_cfg.get("random_state", 42)),
        n_jobs=-1,
    )


def evaluate_model(df: pd.DataFrame, params: dict) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Train/test evaluate and return metrics + row-level predictions."""
    x_train, x_test, y_train, y_test = _split_data(df, params)
    model = _build_model(params)
    model.fit(x_train, y_train)

    y_proba = model.predict_proba(x_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    recall_gate = float(params.get("eval", {}).get("recall_gate", 0.55))
    recall_val = float(recall_score(y_test, y_pred, zero_division=0))

    metrics = {
        "pr_auc": float(average_precision_score(y_test, y_proba)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)) if y_test.nunique() > 1 else 0.5,
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": recall_val,
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "recall_gate": recall_gate,
        "passes_recall_gate": bool(recall_val >= recall_gate),
        "n_test": int(len(y_test)),
    }

    predictions = pd.DataFrame(
        {
            "y_true": y_test.to_numpy(),
            "y_pred": y_pred,
            "y_proba": y_proba,
        }
    ).reset_index(drop=True)

    return metrics, predictions


def run_evaluation(
    features_path: str = "data/processed/features.parquet",
    output_dir: str = "data/processed",
) -> Dict[str, float]:
    """End-to-end Stage 6 evaluation and artifact write."""
    params = load_params()
    df = load_features(features_path)
    metrics, predictions = evaluate_model(df, params)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    metrics_path = out / "eval_metrics.json"
    preds_path = out / "eval_predictions.parquet"

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    predictions.to_parquet(preds_path, index=False)

    return metrics


if __name__ == "__main__":
    result = run_evaluation()
    print("Evaluation complete")
    for k, v in result.items():
        print(f"{k}: {v}")
