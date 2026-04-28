#!/usr/bin/env python3
"""Simple ticker + quarter prediction CLI."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.serving.service import load_model_bundle, predict_with_threshold


FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features.parquet"


def normalize_quarter(value: str) -> str:
    """Convert quarter text like 'Q2 2025' or '2025-Q2' into '2025Q2'."""
    cleaned = value.strip().upper().replace(" ", "")

    match = re.fullmatch(r"(\d{4})Q([1-4])", cleaned)
    if match:
        return f"{match.group(1)}Q{match.group(2)}"

    match = re.fullmatch(r"Q([1-4])(\d{4})", cleaned)
    if match:
        return f"{match.group(2)}Q{match.group(1)}"

    match = re.fullmatch(r"(\d{4})-?Q([1-4])", cleaned)
    if match:
        return f"{match.group(1)}Q{match.group(2)}"

    match = re.fullmatch(r"Q([1-4])-?(\d{4})", cleaned)
    if match:
        return f"{match.group(2)}Q{match.group(1)}"

    raise ValueError(
        f"Could not parse quarter '{value}'. Use formats like 'Q2 2025', '2025Q2', or '2025-Q2'."
    )


def load_feature_row(ticker: str, quarter: str) -> dict:
    """Load a single precomputed feature row for the requested ticker and quarter."""
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Missing {FEATURES_PATH}. Run the feature pipeline first to generate data/processed/features.parquet."
        )

    features_df = pd.read_parquet(FEATURES_PATH)
    ticker_value = ticker.strip().upper()
    quarter_value = normalize_quarter(quarter)

    match = features_df[
        features_df["ticker"].astype(str).str.upper().eq(ticker_value)
        & features_df["quarter"].astype(str).eq(quarter_value)
    ]

    if match.empty:
        available = features_df.loc[
            features_df["ticker"].astype(str).str.upper().eq(ticker_value), "quarter"
        ].dropna().astype(str).unique()
        hint = f" Available quarters for {ticker_value}: {', '.join(sorted(available))}." if len(available) else ""
        raise LookupError(
            f"No feature row found for {ticker_value} {quarter_value}.{hint} "
            "Run the feature pipeline again after adding the newer price and filing data, or pick one of the listed quarters."
        )

    row = match.iloc[0].to_dict()
    row.pop("ticker", None)
    row.pop("quarter", None)
    row.pop("label", None)
    return {key: float(value) for key, value in row.items() if pd.notna(value)}


def predict_ticker_quarter(ticker: str, quarter: str, min_confidence: float = 0.5) -> dict:
    """Predict earnings surprise from just a ticker and a quarter."""
    print(f"Loading precomputed features for {ticker.upper()} {quarter}...")
    features = load_feature_row(ticker, quarter)

    print("Loading trained model...")
    bundle = load_model_bundle()

    print("Scoring...")
    result = predict_with_threshold(
        feature_values=features,
        min_confidence=min_confidence,
        bundle=bundle,
        default_threshold=min_confidence,
    )

    return {
        "ticker": ticker.upper(),
        "quarter": normalize_quarter(quarter),
        **result.__dict__,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict earnings beat/miss from ticker and quarter.")
    parser.add_argument("ticker", help="Ticker symbol, for example AAPL")
    parser.add_argument("quarter", help="Quarter, for example 'Q2 2025' or '2025Q2'")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Decision threshold to apply to the model probability (default: 0.5)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    try:
        result = predict_ticker_quarter(args.ticker, args.quarter, min_confidence=args.threshold)
    except LookupError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc

    print(f"\nResult for {result['ticker']} ({result['quarter']}):")
    print(f"  Probability: {result['probability']:.1%}")
    print(f"  Prediction: {'BEAT' if result['prediction'] == 1 else 'MISS'}")
    print(f"  Threshold:   {result['threshold_used']:.1%}")
    print(f"  Run ID:      {result['run_id'] or 'unknown'}")


if __name__ == "__main__":
    main()