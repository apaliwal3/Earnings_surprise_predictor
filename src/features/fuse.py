"""
Merge tabular features + text embeddings, compute labels, write to parquet.

Inputs:
  - tabular features from src/features/tabular.py: [ticker, quarter, 25 ratios]
  - text embeddings from src/features/text_embeddings.py: [ticker, quarter, 768 embeddings]
  
Output:
  - data/processed/features.parquet: [ticker, quarter, 25 ratios, 768 embeddings, label]
"""
import os
import pandas as pd
import numpy as np
import yaml
from pathlib import Path

from src.features.tabular import generate_tabular_features, load_prices_data
from src.features.text_embeddings import generate_text_embeddings


def load_params() -> dict:
    """Load params.yaml to get surprise threshold and other config."""
    params_path = Path("params.yaml")
    with params_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_label_threshold(params: dict) -> float:
    """Read label threshold from params.yaml as the single source of truth."""
    # Backward-compatible key support, but always from params.yaml.
    if "label" in params and "surprise_threshold" in params["label"]:
        return float(params["label"]["surprise_threshold"])
    if "data" in params and "label_threshold" in params["data"]:
        return float(params["data"]["label_threshold"])

    raise KeyError(
        "Missing label threshold in params.yaml. "
        "Expected either 'label.surprise_threshold' or 'data.label_threshold'."
    )


def compute_label(row: pd.Series, threshold: float) -> int:
    """
    Compute binary label for earnings surprise.
    
    Label = 1 if (actual_EPS - consensus_EPS) / abs(consensus_EPS) > threshold, else 0
    
    Args:
        row: A row from merged features DataFrame with columns actual_EPS, consensus_EPS
    
    Returns:
        int: 1 or 0
    """
    actual_eps = row.get("actual_EPS")
    consensus_eps = row.get("consensus_EPS")

    if pd.isna(actual_eps) or pd.isna(consensus_eps) or consensus_eps == 0:
        return 0

    surprise = (actual_eps - consensus_eps) / abs(consensus_eps)
    return int(surprise > threshold)


def merge_features_and_labels(tabular_df: pd.DataFrame, embeddings_df: pd.DataFrame, 
                              prices_df: dict, threshold: float) -> pd.DataFrame:
    """
    Merge tabular + embeddings on (ticker, quarter), add labels from price data.
    
    Args:
        tabular_df: From tabular.py: [ticker, quarter, feature_1, ..., feature_25]
        embeddings_df: From text_embeddings.py: [ticker, quarter, embedding_0, ..., embedding_767]
        prices_df: Dict {ticker: pd.DataFrame with EPS data}
        threshold: Surprise threshold from params.yaml
    
    Returns:
        pd.DataFrame: [ticker, quarter, feature_1, ..., feature_25, embedding_0, ..., embedding_767, label]
    """
    merged = tabular_df.merge(embeddings_df, on=["ticker", "quarter"], how="inner")

    label_rows = []
    for ticker, ticker_prices in prices_df.items():
        if ticker_prices.empty:
            continue

        df = ticker_prices.copy()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.set_index("date")

        if not isinstance(df.index, pd.DatetimeIndex):
            continue

        required_cols = {"reported_eps", "eps_estimate"}
        if not required_cols.issubset(df.columns):
            continue

        quarter_series = df.index.to_period("Q")
        grouped = df.groupby(quarter_series)
        for period, grp in grouped:
            actual_series = grp["reported_eps"].dropna()
            consensus_series = grp["eps_estimate"].dropna()
            if actual_series.empty or consensus_series.empty:
                continue

            label_rows.append(
                {
                    "ticker": str(ticker).upper(),
                    "quarter": f"{period.year}Q{period.quarter}",
                    "actual_EPS": float(actual_series.iloc[-1]),
                    "consensus_EPS": float(consensus_series.iloc[-1]),
                }
            )

    labels_df = pd.DataFrame(label_rows)
    if labels_df.empty:
        merged["label"] = 0
        return merged

    # Deduplicate in case of repeated snapshots per quarter.
    labels_df = labels_df.sort_values(["ticker", "quarter"]).drop_duplicates(
        subset=["ticker", "quarter"], keep="last"
    )

    merged = merged.merge(labels_df, on=["ticker", "quarter"], how="left")
    merged["label"] = merged.apply(lambda row: compute_label(row, threshold=threshold), axis=1)

    merged = merged.drop(columns=["actual_EPS", "consensus_EPS"], errors="ignore")

    numeric_cols = merged.select_dtypes(include=[np.number]).columns
    merged[numeric_cols] = merged[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return merged


def fuse_features(output_dir: str = "data/processed", device: str = "cpu") -> None:
    """
    End-to-end: generate tabular, generate embeddings, merge, compute labels, write parquet.
    
    Args:
        output_dir: Where to write features.parquet
        device: "cpu" or "mps"
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    params = load_params()
    threshold = get_label_threshold(params)

    tabular_df = generate_tabular_features()
    embeddings_df = generate_text_embeddings(device=device)
    prices_df = load_prices_data()

    fused_df = merge_features_and_labels(
        tabular_df=tabular_df,
        embeddings_df=embeddings_df,
        prices_df=prices_df,
        threshold=threshold,
    )

    output_path = out_dir / "features.parquet"
    fused_df.to_parquet(output_path, index=False)

    print(f"Wrote fused dataset: {output_path}")
    print(f"Shape: {fused_df.shape}")
    print(fused_df.head())


if __name__ == "__main__":
    fuse_features(device="cpu")  # or "mps" for Apple Silicon
