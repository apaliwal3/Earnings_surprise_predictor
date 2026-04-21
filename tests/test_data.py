"""Stage 3 tests for raw ingestion outputs.

These tests should validate data contracts, not model quality.
"""

from __future__ import annotations

from pathlib import Path
import re

import pandas as pd


RAW_PRICES_DIR = Path("data/raw/prices")
RAW_FILINGS_DIR = Path("data/raw/filings")
EXPECTED_PRICE_COLUMNS = [
    "date",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
    "dividends",
    "stock_splits",
    "reported_eps",
    "eps_estimate",
    "surprise_percent",
]
FILING_NAME_PATTERN = re.compile(r"^[A-Z\-]+_\d{4}Q[1-4]_[A-Za-z0-9\-]+\.txt$")


def test_prices_outputs_exist() -> None:
    """Ensure at least one per-ticker price CSV exists."""
    assert RAW_PRICES_DIR.exists(), f"Missing prices directory: {RAW_PRICES_DIR}"
    csvs = sorted(RAW_PRICES_DIR.glob("*.csv"))
    assert csvs, f"No ticker CSV files found in {RAW_PRICES_DIR}"


def test_price_csv_has_expected_schema() -> None:
    """Load one sample CSV and assert expected schema + basic shape."""
    csvs = sorted(RAW_PRICES_DIR.glob("*.csv"))
    assert csvs, "No CSV files available to validate"

    sample = csvs[0]
    df = pd.read_csv(sample)

    assert list(df.columns) == EXPECTED_PRICE_COLUMNS
    assert len(df) > 0, f"Sample CSV is empty: {sample}"
    parsed_dates = pd.to_datetime(df["date"], errors="coerce")
    assert parsed_dates.notna().all(), f"Unparseable date values in {sample}"

    ticker_values = set(df["ticker"].astype(str).str.upper().str.strip())
    assert ticker_values == {sample.stem.upper()}


def test_filings_outputs_exist() -> None:
    """Ensure at least one flattened filing text file exists."""
    assert RAW_FILINGS_DIR.exists(), f"Missing filings directory: {RAW_FILINGS_DIR}"
    txts = sorted(RAW_FILINGS_DIR.glob("*.txt"))
    assert txts, f"No flattened filing text files found in {RAW_FILINGS_DIR}"


def test_filings_payload_not_tiny() -> None:
    """Assert sample filing payload has deterministic name and useful length."""
    txts = sorted(RAW_FILINGS_DIR.glob("*.txt"))
    assert txts, "No filing text files available to validate"

    sample = txts[0]
    assert FILING_NAME_PATTERN.match(sample.name), (
        f"Unexpected filing filename format: {sample.name}"
    )

    content = sample.read_text(encoding="utf-8", errors="ignore")
    assert len(content) >= 1000, f"Filing payload appears too short: {sample}"
