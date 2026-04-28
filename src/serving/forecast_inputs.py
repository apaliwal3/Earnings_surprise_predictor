"""Helpers for resolving ticker + quarter requests into model feature payloads."""

from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path
from typing import Literal

import pandas as pd

ForecastMode = Literal["exact", "next"]

FEATURES_PATH = Path("data/processed/features.parquet")


@dataclass(frozen=True)
class ResolvedForecastInput:
    """Resolved feature payload plus quarter metadata."""

    ticker: str
    target_quarter: str
    as_of_quarter: str
    forecast_mode: ForecastMode
    features: dict[str, float]


def normalize_quarter(value: str) -> str:
    """Normalize quarter text such as 'Q2 2025' or '2025-Q2' into '2025Q2'."""
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


def quarter_sort_key(value: str) -> int:
    """Convert a YYYYQn quarter string into a sortable integer."""
    normalized = normalize_quarter(value)
    year = int(normalized[:4])
    quarter = int(normalized[-1])
    return year * 4 + quarter


def next_quarter(value: str) -> str:
    """Return the quarter after the provided YYYYQn value."""
    normalized = normalize_quarter(value)
    year = int(normalized[:4])
    quarter = int(normalized[-1])
    if quarter == 4:
        return f"{year + 1}Q1"
    return f"{year}Q{quarter + 1}"


def load_features_dataframe(features_path: str | Path = FEATURES_PATH) -> pd.DataFrame:
    """Load the fused feature table used for serving."""
    path = Path(features_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run the feature pipeline first.")
    return pd.read_parquet(path)


def _feature_payload(row: pd.Series) -> dict[str, float]:
    """Convert a resolved feature row into the numeric payload expected by the model."""
    payload = row.to_dict()
    payload.pop("ticker", None)
    payload.pop("quarter", None)
    payload.pop("label", None)
    payload.pop("quarter_key", None)
    return {key: float(value) for key, value in payload.items() if pd.notna(value)}


def resolve_feature_input(
    features_df: pd.DataFrame,
    ticker: str,
    quarter: str | None = None,
    forecast_mode: ForecastMode = "next",
) -> ResolvedForecastInput:
    """Resolve ticker + quarter into a feature payload.

    exact: score the exact historical quarter if it exists.
    next: use the latest available quarter before the requested target quarter.
          If no target quarter is provided, use the latest available quarter and
          label the output as the next quarter after that row.
    """
    ticker_value = ticker.strip().upper()
    ticker_rows = features_df[features_df["ticker"].astype(str).str.upper().eq(ticker_value)].copy()
    if ticker_rows.empty:
        raise LookupError(f"No feature rows found for ticker {ticker_value}.")

    ticker_rows = ticker_rows[ticker_rows["quarter"].notna()].copy()
    ticker_rows["quarter_key"] = ticker_rows["quarter"].astype(str).map(quarter_sort_key)
    ticker_rows = ticker_rows.dropna(subset=["quarter_key"])
    if ticker_rows.empty:
        raise LookupError(f"No valid quarter rows found for ticker {ticker_value}.")

    if forecast_mode == "exact":
        if quarter is None:
            raise ValueError("quarter is required when forecast_mode='exact'.")

        target_quarter = normalize_quarter(quarter)
        exact_match = ticker_rows[ticker_rows["quarter"].astype(str).eq(target_quarter)]
        if exact_match.empty:
            available = sorted(ticker_rows["quarter"].astype(str).unique(), key=quarter_sort_key)
            raise LookupError(
                f"No exact feature row found for {ticker_value} {target_quarter}. "
                f"Available quarters: {', '.join(available)}."
            )

        selected = exact_match.iloc[-1]
        return ResolvedForecastInput(
            ticker=ticker_value,
            target_quarter=target_quarter,
            as_of_quarter=target_quarter,
            forecast_mode=forecast_mode,
            features=_feature_payload(selected),
        )

    if quarter is None:
        selected = ticker_rows.sort_values("quarter_key").iloc[-1]
        as_of_quarter = str(selected["quarter"])
        target_quarter = next_quarter(as_of_quarter)
    else:
        target_quarter = normalize_quarter(quarter)
        target_key = quarter_sort_key(target_quarter)
        eligible = ticker_rows[ticker_rows["quarter_key"] < target_key].sort_values("quarter_key")
        if eligible.empty:
            available = sorted(ticker_rows["quarter"].astype(str).unique(), key=quarter_sort_key)
            raise LookupError(
                f"No pre-target feature row found for {ticker_value} {target_quarter}. "
                f"Available quarters: {', '.join(available)}."
            )
        selected = eligible.iloc[-1]
        as_of_quarter = str(selected["quarter"])

    return ResolvedForecastInput(
        ticker=ticker_value,
        target_quarter=target_quarter,
        as_of_quarter=as_of_quarter,
        forecast_mode=forecast_mode,
        features=_feature_payload(selected),
    )