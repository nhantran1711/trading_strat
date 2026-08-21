"""
Fetch and cache 1-minute OHLCV bars for the research universe from Alpaca.

Usage:
    python scripts/fetch_data.py [--start 2022-08-01] [--end 2025-08-01] [--force]

Requires ALPACA_API_KEY / ALPACA_SECRET_KEY (see .env.example). Without them
this will raise data.loader.MissingCredentialsError with setup instructions.
"""
from __future__ import annotations

import argparse
import logging
import sys

sys.path.insert(0, ".")

from config import DEFAULT_END_DATE, DEFAULT_START_DATE, UNIVERSE
from data.loader import AlpacaBarLoader, MissingCredentialsError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=DEFAULT_START_DATE)
    parser.add_argument("--end", default=DEFAULT_END_DATE)
    parser.add_argument("--symbols", nargs="+", default=UNIVERSE)
    parser.add_argument("--force", action="store_true", help="Ignore cache, refetch everything.")
    args = parser.parse_args()

    loader = AlpacaBarLoader()
    try:
        for symbol in args.symbols:
            logger.info("Fetching %s [%s, %s] ...", symbol, args.start, args.end)
            df = loader.get_bars(symbol, args.start, args.end, force_refresh=args.force)
            logger.info("  -> %d bars cached for %s", len(df), symbol)
    except MissingCredentialsError as e:
        logger.error(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
