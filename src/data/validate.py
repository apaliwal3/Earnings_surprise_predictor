"""Validate raw downloaded data before feature engineering.
"""

from __future__ import annotations

import re
from pathlib import Path

import great_expectations as gx
import great_expectations.expectations as gxe
import pandas as pd
import yaml


def load_params(params_path: str = "params.yaml") -> dict:
    """Load project params from the single source of truth."""
    with open(params_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def prices_dir() -> Path:
    """Return location of per-ticker raw price CSVs."""
    return Path("data/raw/prices")


def filings_dir() -> Path:
    """Return location of flattened filing text files."""
    return Path("data/raw/filings")


def expected_price_columns() -> list[str]:
    """Canonical schema expected from download_prices.py outputs."""
    return [
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


def _ge_success(result: object) -> bool:
    """Handle Great Expectations result objects across versions."""
    if hasattr(result, "success"):
        return bool(result.success)
    if isinstance(result, dict):
        return bool(result.get("success", False))
    return False


def _build_batch(df: pd.DataFrame):
    """Build a transient GE batch for dataframe validation."""
    context = gx.get_context(mode="ephemeral")
    datasource = context.data_sources.add_pandas("prices_datasource")
    asset = datasource.add_dataframe_asset("prices_asset")
    batch_definition = asset.add_batch_definition_whole_dataframe("prices_batch")
    return batch_definition.get_batch(batch_parameters={"dataframe": df})


def _run_expectation(batch, expectation, failures: list[str], message: str) -> None:
    """Run one GE expectation and accumulate a readable failure message."""
    result = batch.validate(expectation)
    if not _ge_success(result):
        failures.append(message)


def validate_prices_schema(
    price_csv: Path,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> None:
    """Validate one per-ticker price CSV contract with Great Expectations."""
    if not price_csv.exists():
        raise FileNotFoundError(f"Price CSV missing: {price_csv}")

    df = pd.read_csv(price_csv)
    if df.empty:
        raise ValueError(f"Price CSV is empty: {price_csv}")

    batch = _build_batch(df)
    failures: list[str] = []

    _run_expectation(
        batch,
        gxe.ExpectTableColumnsToMatchOrderedList(column_list=expected_price_columns()),
        failures,
        "columns do not match expected schema/order",
    )
    _run_expectation(
        batch,
        gxe.ExpectTableRowCountToBeBetween(min_value=1),
        failures,
        "table has no rows",
    )

    for col in ["date", "ticker", "open", "high", "low", "close", "volume"]:
        _run_expectation(
            batch,
            gxe.ExpectColumnValuesToNotBeNull(column=col),
            failures,
            f"column has nulls: {col}",
        )

    for col in ["open", "high", "low", "close", "adj_close", "volume"]:
        _run_expectation(
            batch,
            gxe.ExpectColumnValuesToBeBetween(column=col, min_value=0),
            failures,
            f"column has negative values: {col}",
        )

    _run_expectation(
        batch,
        gxe.ExpectColumnValuesToMatchRegex(column="ticker", regex=r"^[A-Z\-]+$"),
        failures,
        "ticker column contains invalid symbols",
    )
    _run_expectation(
        batch,
        gxe.ExpectColumnUniqueValueCountToBeBetween(
            column="ticker", min_value=1, max_value=1
        ),
        failures,
        "ticker CSV contains more than one ticker value",
    )

    ticker_values = set(df["ticker"].astype(str).str.upper().str.strip())
    expected_ticker = price_csv.stem.upper()
    if ticker_values != {expected_ticker}:
        failures.append(
            f"ticker values {ticker_values} do not match filename stem {expected_ticker}"
        )

    parsed_dates = pd.to_datetime(df["date"], errors="coerce")
    if parsed_dates.isna().any():
        failures.append("date column contains unparsable values")
    else:
        min_date = parsed_dates.min().normalize()
        max_date = parsed_dates.max().normalize()
        if min_date < start_date or max_date > end_date:
            failures.append(
                "date range is outside configured params window: "
                f"[{min_date.date()} to {max_date.date()}] vs "
                f"[{start_date.date()} to {end_date.date()}]"
            )

    if failures:
        raise ValueError(f"Validation failed for {price_csv}: " + "; ".join(failures))


def validate_filings_payloads(filings_root: Path) -> None:
    """Validate flattened filing text corpus integrity."""
    if not filings_root.exists():
        raise FileNotFoundError(f"Filings directory missing: {filings_root}")

    flat_files = sorted(filings_root.glob("*.txt"))
    if not flat_files:
        raise ValueError(f"No flattened filing text files found in {filings_root}")

    pattern = re.compile(r"^[A-Z\-]+_\d{4}Q[1-4]_[A-Za-z0-9\-]+\.txt$")
    invalid_names = [f.name for f in flat_files if not pattern.match(f.name)]
    if invalid_names:
        raise ValueError(
            "Invalid filing filename format for: " + ", ".join(invalid_names[:10])
        )

    tiny_files = [f.name for f in flat_files if f.stat().st_size < 1000]
    if tiny_files:
        raise ValueError(
            "Filing payload appears too small for: " + ", ".join(tiny_files[:10])
        )


def main() -> None:
    params = load_params()
    start_date = pd.Timestamp(params["data"]["start_date"]).normalize()
    end_date = pd.Timestamp(params["data"]["end_date"]).normalize()

    pdir = prices_dir()
    fdir = filings_dir()
    if not pdir.exists():
        raise FileNotFoundError(f"Prices directory missing: {pdir}")

    price_csvs = sorted(pdir.glob("*.csv"))
    if not price_csvs:
        raise ValueError(f"No price CSV files found in {pdir}")

    for csv_path in price_csvs:
        validate_prices_schema(csv_path, start_date=start_date, end_date=end_date)

    validate_filings_payloads(fdir)
    print("Data validation passed for prices and filings.")


if __name__ == "__main__":
    main()
