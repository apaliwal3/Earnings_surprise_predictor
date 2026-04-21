"""Download market and EPS-related data for the target equity universe.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml
import yfinance as yf

try:
    from .download_filings import load_sp500_tickers as load_universe
except ImportError:
    # Allows running as `python src/data/download_prices.py`.
    from download_filings import load_sp500_tickers as load_universe


LOGGER = logging.getLogger(__name__)


def load_params(params_path: str = "params.yaml") -> dict:
    """Load project params from the single source of truth."""
    with open(params_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def output_dir() -> Path:
    """Return raw prices output directory."""
    return Path("data/raw/prices")


def load_sp500_tickers() -> Iterable[str]:
    """Return the same S&P 500 universe used by filings download.
    """
    return load_universe()


def fetch_price_and_eps(
    ticker: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Pull ticker price history and merge earnings event fields.

    Output has one row per trading day, with earnings values populated on
    earnings event dates when available.
    """
    stock = yf.Ticker(ticker)

    history = stock.history(
        start=start_date,
        end=end_date,
        auto_adjust=False,
        actions=True,
    )
    if history.empty:
        return pd.DataFrame(
            columns=[
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
        )

    history = history.reset_index().rename(
        columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
            "Dividends": "dividends",
            "Stock Splits": "stock_splits",
        }
    )
    history["date"] = pd.to_datetime(history["date"]).dt.tz_localize(None).dt.date
    history["ticker"] = ticker

    try:
        earnings = stock.get_earnings_dates(limit=80)
    except Exception:  # noqa: BLE001
        earnings = pd.DataFrame()

    earnings_df = pd.DataFrame(
        columns=["date", "reported_eps", "eps_estimate", "surprise_percent"]
    )
    if earnings is not None and not earnings.empty:
        earnings = earnings.reset_index().rename(columns={"Earnings Date": "date"})
        if "date" in earnings.columns:
            earnings["date"] = (
                pd.to_datetime(earnings["date"]).dt.tz_localize(None).dt.date
            )

        col_map = {
            "Reported EPS": "reported_eps",
            "EPS Estimate": "eps_estimate",
            "Surprise(%)": "surprise_percent",
        }
        existing = [c for c in col_map if c in earnings.columns]
        if "date" in earnings.columns:
            earnings_df = earnings[["date", *existing]].rename(columns=col_map)

    merged = history.merge(earnings_df, on="date", how="left")

    ordered_cols = [
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
    for col in ordered_cols:
        if col not in merged.columns:
            merged[col] = pd.NA

    return merged[ordered_cols].sort_values("date").reset_index(drop=True)


def write_ticker_csv(ticker: str, payload: pd.DataFrame, destination: Path) -> None:
    """Persist one CSV per ticker in data/raw/prices/."""
    file_path = destination / f"{ticker}.csv"
    payload.to_csv(file_path, index=False)


def remove_if_exists(path: Path) -> None:
    """Delete stale output files when ticker payload is empty."""
    if path.exists():
        path.unlink()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    params = load_params()
    start_date = params["data"]["start_date"]
    end_date = params["data"]["end_date"]

    out = output_dir()
    out.mkdir(parents=True, exist_ok=True)

    tickers = list(load_sp500_tickers())
    failures = 0
    skipped_empty = 0

    for idx, ticker in enumerate(tickers, start=1):
        try:
            LOGGER.info("[%d/%d] downloading prices for %s", idx, len(tickers), ticker)
            payload = fetch_price_and_eps(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
            )
            if payload.empty:
                skipped_empty += 1
                stale_path = out / f"{ticker}.csv"
                remove_if_exists(stale_path)
                LOGGER.warning(
                    "%s: no price history in requested window, skipping output file",
                    ticker,
                )
                continue
            write_ticker_csv(ticker=ticker, payload=payload, destination=out)
        except Exception as exc:  # noqa: BLE001
            failures += 1
            LOGGER.exception("Failed downloading %s: %s", ticker, exc)

    LOGGER.info(
        "Done. total_tickers=%d failures=%d skipped_empty=%d",
        len(tickers),
        failures,
        skipped_empty,
    )


if __name__ == "__main__":
    main()
