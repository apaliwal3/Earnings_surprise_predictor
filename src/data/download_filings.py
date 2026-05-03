"""Download 10-Q filings for the target equity universe.
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime
import argparse
from io import StringIO
from pathlib import Path
from typing import Iterable, Optional

import certifi
import pandas as pd
import requests
import yaml
from sec_edgar_downloader import Downloader


LOGGER = logging.getLogger(__name__)

WIKI_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

# Small deterministic fallback for offline/dev scenarios.
FALLBACK_TICKERS = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META",
]


def load_params(params_path: str = "params.yaml") -> dict:
    """Load project params from the single source of truth."""
    with open(params_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def output_dir() -> Path:
    """Return raw filings output directory."""
    return Path("data/raw/filings")


def normalize_ticker(ticker: str) -> str:
    """Normalize tickers to a stable format shared across ingestion scripts."""
    return ticker.strip().upper().replace(".", "-")


def load_sp500_tickers() -> Iterable[str]:
    """Return normalized S&P 500 ticker symbols.

    Uses a deterministic fallback set if the online source is unavailable.
    """
    wiki_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        response = requests.get(
            wiki_url,
            timeout=30,
            verify=certifi.where(),
            headers=WIKI_HEADERS,
        )
        response.raise_for_status()
        table = pd.read_html(StringIO(response.text), flavor="lxml")[0]
        symbols = table["Symbol"].astype(str).tolist()
        normalized = sorted({normalize_ticker(symbol) for symbol in symbols})
        LOGGER.info("Loaded %d S&P 500 tickers from Wikipedia", len(normalized))
        return normalized
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning(
            "Falling back to small static ticker set because S&P fetch failed: %s",
            exc,
        )
        return FALLBACK_TICKERS


def filing_quarter(filed_at: datetime) -> str:
    """Convert a filing date to quarter notation, e.g. 2024Q3."""
    quarter = ((filed_at.month - 1) // 3) + 1
    return f"{filed_at.year}Q{quarter}"


def extract_filed_date(raw_text: str) -> Optional[datetime]:
    """Extract filed date from SEC full-submission text."""
    match = re.search(r"FILED\s+AS\s+OF\s+DATE:\s*(\d{8})", raw_text)
    if not match:
        return None
    return datetime.strptime(match.group(1), "%Y%m%d")


def materialize_flat_text_files(ticker: str, destination: Path) -> int:
    """Create one flattened text payload per filing for downstream NLP."""
    filings_root = destination / "sec-edgar-filings" / ticker / "10-Q"
    if not filings_root.exists():
        return 0

    written = 0
    for accession_dir in sorted(p for p in filings_root.iterdir() if p.is_dir()):
        submission_path = accession_dir / "full-submission.txt"
        if not submission_path.exists():
            continue

        raw_text = submission_path.read_text(encoding="utf-8", errors="ignore")
        filed_at = extract_filed_date(raw_text)
        quarter_tag = filing_quarter(filed_at) if filed_at else "unknownQ"

        flat_name = f"{ticker}_{quarter_tag}_{accession_dir.name}.txt"
        flat_path = destination / flat_name
        flat_path.write_text(raw_text, encoding="utf-8")
        written += 1

    return written


def download_10q_for_ticker(
    downloader: Downloader,
    ticker: str,
    start_date: str,
    end_date: str,
    destination: Path,
) -> None:
    """Download 10-Q filings for one ticker and flatten text payloads."""
    normalized = normalize_ticker(ticker)
    downloader.get("10-Q", normalized, after=start_date, before=end_date)
    materialized = materialize_flat_text_files(normalized, destination)
    LOGGER.info("%s: materialized %d filing text files", normalized, materialized)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    parser = argparse.ArgumentParser(
        prog="download_filings",
        description="Download SEC 10-Q filings for tickers.",
    )
    parser.add_argument("--params", default="params.yaml", help="Path to params.yaml")
    parser.add_argument("--start-date", dest="start_date", help="Override start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", dest="end_date", help="Override end date (YYYY-MM-DD)")
    args = parser.parse_args()

    params = load_params(args.params)

    # Resolve start/end date precedence: CLI args > CI env vars > params.yaml
    start_date = args.start_date or os.getenv("CI_START_DATE") or params["data"]["start_date"]
    end_date = args.end_date or os.getenv("CI_END_DATE") or params["data"]["end_date"]

    # Default end_date to today if not provided
    if end_date in (None, ""):
        end_date = datetime.utcnow().date().isoformat()

    out = output_dir()
    out.mkdir(parents=True, exist_ok=True)

    company_name = os.getenv("SEC_COMPANY_NAME", "earnings-surprise-predictor")
    email_address = os.getenv("SEC_EMAIL")
    if not email_address:
        raise ValueError(
            "Set SEC_EMAIL in your environment before downloading SEC filings."
        )

    downloader = Downloader(company_name, email_address, str(out))

    tickers = list(load_sp500_tickers())
    failures = 0
    for idx, ticker in enumerate(tickers, start=1):
        try:
            LOGGER.info("[%d/%d] downloading 10-Qs for %s", idx, len(tickers), ticker)
            download_10q_for_ticker(
                downloader=downloader,
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                destination=out,
            )
        except Exception as exc:  # noqa: BLE001
            failures += 1
            LOGGER.exception("Failed downloading %s: %s", ticker, exc)

    LOGGER.info("Done. total_tickers=%d failures=%d", len(tickers), failures)


if __name__ == "__main__":
    main()
