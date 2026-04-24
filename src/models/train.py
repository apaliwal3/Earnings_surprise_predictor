"""Stage 5: Train LightGBM model and log run metadata to MLflow."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split


def load_params(params_path: str = "params.yaml") -> dict:
    """Load training and model params from params.yaml."""
    with open(params_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_features(features_path: str = "data/processed/features.parquet") -> pd.DataFrame:
    """Load fused feature matrix."""
    path = Path(features_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing features file: {features_path}")
    return pd.read_parquet(path)


def split_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Split fused dataframe into model matrix X and target y."""
    if "label" not in df.columns:
        raise ValueError("Expected 'label' column in features dataframe")

    drop_cols = ["label", "ticker", "quarter"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    if not feature_cols:
        raise ValueError("No feature columns found after dropping identifiers/label")

    x = df[feature_cols].copy()
    y = df["label"].astype(int)
    return x, y


def train_model(df: pd.DataFrame, params: dict):
    """Train LightGBM model and return artifacts/metrics."""
    x, y = split_xy(df)

    train_cfg = params.get("train", {})
    model_cfg = params.get("model", {})

    test_size = float(train_cfg.get("test_size", 0.2))
    random_state = int(train_cfg.get("random_state", 42))

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if y.nunique() > 1 else None,
    )

    model = lgb.LGBMClassifier(
        objective="binary",
        num_leaves=int(model_cfg.get("num_leaves", 31)),
        learning_rate=float(model_cfg.get("learning_rate", 0.05)),
        n_estimators=int(model_cfg.get("n_estimators", 300)),
        max_depth=int(model_cfg.get("max_depth", -1)),
        subsample=float(model_cfg.get("subsample", 0.8)),
        colsample_bytree=float(model_cfg.get("colsample_bytree", 0.8)),
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    y_proba = model.predict_proba(x_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = {
        "pr_auc": float(average_precision_score(y_test, y_proba)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)) if y_test.nunique() > 1 else 0.5,
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "n_features": int(x.shape[1]),
    }

    return model, metrics


def train_and_log(
    features_path: str = "data/processed/features.parquet",
    experiment_name: str = "earnings-surprise-predictor",
) -> dict:
    """Run training and log params/metrics/model to MLflow."""
    params = load_params()
    df = load_features(features_path)

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="lightgbm-train"):
        mlflow.log_params({f"model.{k}": v for k, v in params.get("model", {}).items()})
        mlflow.log_params({f"train.{k}": v for k, v in params.get("train", {}).items()})

        model, metrics = train_model(df, params)
        mlflow.log_metrics(metrics)

        mlflow.lightgbm.log_model(model, artifact_path="model")
        mlflow.log_param("features_path", features_path)

        run_id = mlflow.active_run().info.run_id

    return {"run_id": run_id, **metrics}


if __name__ == "__main__":
    result = train_and_log()
    print("Training complete")
    for k, v in result.items():
        print(f"{k}: {v}")
