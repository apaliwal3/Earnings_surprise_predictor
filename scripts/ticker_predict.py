#!/usr/bin/env python3
"""Simple ticker + quarter prediction CLI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.serving.service import predict_from_ticker_quarter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict earnings beat/miss from ticker and quarter.")
    parser.add_argument("ticker", help="Ticker symbol, for example AAPL")
    parser.add_argument(
        "quarter",
        nargs="?",
        help="Quarter, for example 'Q2 2025' or '2025Q2'. Optional when using --mode next.",
    )
    parser.add_argument(
        "--mode",
        choices=["next", "exact"],
        default="next",
        help="next uses the latest available pre-target quarter; exact scores the requested quarter itself.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold to apply to the model probability (default: 0.5)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    try:
        result = predict_from_ticker_quarter(
            ticker=args.ticker,
            quarter=args.quarter,
            forecast_mode=args.mode,
            min_confidence=args.threshold,
        )
    except LookupError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc

    print(f"\nResult for {result.ticker} targeting {result.target_quarter}:")
    print(f"  Mode:         {result.forecast_mode}")
    print(f"  As-of quarter: {result.as_of_quarter}")
    print(f"  Probability:   {result.probability:.1%}")
    print(f"  Prediction:    {'BEAT' if result.prediction == 1 else 'MISS'}")
    print(f"  Threshold:     {result.threshold_used:.1%}")
    print(f"  Run ID:        {result.run_id or 'unknown'}")


if __name__ == "__main__":
    main()