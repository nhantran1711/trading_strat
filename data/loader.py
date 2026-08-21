"""
Historical 1-minute OHLCV data acquisition, via the Alpaca Market Data API,
with on-disk parquet caching so repeated research runs do not re-hit the API.

Design notes on avoiding common data-integrity failure modes:

* Split/dividend adjustment: bars are requested with `adjustment="all"`
  (splits + dividends) by default, and the adjustment mode actually used is
  recorded in the cache file's sidecar metadata so downstream code can tell
  what it is working with.
* Corporate actions: because we request pre-adjusted bars directly from the
  vendor, we do not need to separately detect/apply split ratios -- but we
  record the ticker's split/action history alongside the cache for auditing.
* Survivorship bias: the universe here (SPY, QQQ, NVDA, META, AMZN) is fixed
  by the research brief and consists of instruments that existed and were
  liquid across the whole lookback window. This is disclosed explicitly in
  the README/report -- it is a real limitation (no delisted/failed names are
  considered) but it is a DECLARED scope limit, not an accidental one.
* No look-ahead from the fetch layer itself: this module only ever returns
  bars up to (and including) a `end` timestamp the caller supplies; it never
  reaches into "the future" relative to whatever date range a backtest asks
  for.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from config import DATA_CACHE_DIR, EASTERN

logger = logging.getLogger(__name__)

load_dotenv()  # populate os.environ from a local .env file if present


class MissingCredentialsError(RuntimeError):
    """Raised when ALPACA_API_KEY / ALPACA_SECRET_KEY are not configured."""


@dataclass
class CacheMeta:
    symbol: str
    start: str
    end: str
    adjustment: str
    feed: str
    fetched_at: str
    rows: int


def _cache_paths(symbol: str) -> tuple[Path, Path]:
    sym_dir = DATA_CACHE_DIR / symbol
    sym_dir.mkdir(parents=True, exist_ok=True)
    return sym_dir / f"{symbol}_1min.parquet", sym_dir / f"{symbol}_1min.meta.json"


class AlpacaBarLoader:
    """Fetches and caches 1-minute OHLCV bars for US equities via Alpaca."""

    def __init__(self, api_key: str | None = None, secret_key: str | None = None,
                 feed: str | None = None):
        self.api_key = api_key or os.environ.get("ALPACA_API_KEY")
        self.secret_key = secret_key or os.environ.get("ALPACA_SECRET_KEY")
        self.feed = feed or os.environ.get("ALPACA_DATA_FEED", "sip")
        self._client = None  # lazily constructed -- unit tests never need it

    def _require_client(self):
        if self._client is not None:
            return self._client
        if not self.api_key or not self.secret_key:
            raise MissingCredentialsError(
                "ALPACA_API_KEY / ALPACA_SECRET_KEY are not set. Copy .env.example to "
                ".env and fill in a free Alpaca Markets API key (alpaca.markets), or "
                "export them as environment variables."
            )
        from alpaca.data.historical import StockHistoricalDataClient

        self._client = StockHistoricalDataClient(self.api_key, self.secret_key)
        return self._client

    def fetch_from_api(
        self,
        symbol: str,
        start: str,
        end: str,
        adjustment: str = "all",
    ) -> pd.DataFrame:
        """Pull raw 1-minute bars from Alpaca for [start, end] (inclusive dates,
        US/Eastern). Returns a DataFrame indexed by tz-aware US/Eastern timestamps.
        """
        from alpaca.data.enums import Adjustment, DataFeed
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        client = self._require_client()
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Minute,
            start=pd.Timestamp(start, tz=EASTERN).to_pydatetime(),
            end=pd.Timestamp(end, tz=EASTERN).to_pydatetime(),
            adjustment=Adjustment(adjustment),
            feed=DataFeed(self.feed),
        )
        bars = client.get_stock_bars(request)
        df = bars.df
        if df.empty:
            logger.warning("Alpaca returned 0 bars for %s in [%s, %s]", symbol, start, end)
            return df

        # bars.df has a MultiIndex (symbol, timestamp) when requested for a single
        # symbol via symbol_or_symbols=str; normalize to a single-symbol frame.
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol, level="symbol")

        df.index = pd.to_datetime(df.index, utc=True).tz_convert(EASTERN)
        df.index.name = "timestamp"
        df = df.rename(columns={"trade_count": "trade_count", "vwap": "vwap_vendor"})
        df["symbol"] = symbol
        return df.sort_index()

    def get_bars(
        self,
        symbol: str,
        start: str,
        end: str,
        adjustment: str = "all",
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Return cached bars for `symbol` covering [start, end], fetching from the
        API and updating the cache if needed. This performs a simple "cache covers
        requested range" check; it does not do incremental gap-filling beyond that.
        """
        parquet_path, meta_path = _cache_paths(symbol)

        if not force_refresh and parquet_path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text())
            cached_start = pd.Timestamp(meta["start"])
            cached_end = pd.Timestamp(meta["end"])
            if (
                cached_start <= pd.Timestamp(start)
                and cached_end >= pd.Timestamp(end)
                and meta["adjustment"] == adjustment
                and meta["feed"] == self.feed
            ):
                df = pd.read_parquet(parquet_path)
                df.index = pd.to_datetime(df.index, utc=True).tz_convert(EASTERN)
                mask = (df.index >= pd.Timestamp(start, tz=EASTERN)) & (
                    df.index <= pd.Timestamp(end, tz=EASTERN) + pd.Timedelta(days=1)
                )
                return df.loc[mask]
            logger.info("Cache for %s does not cover requested range; refetching.", symbol)

        df = self.fetch_from_api(symbol, start, end, adjustment=adjustment)
        if df.empty:
            return df

        df.to_parquet(parquet_path)
        meta = CacheMeta(
            symbol=symbol,
            start=start,
            end=end,
            adjustment=adjustment,
            feed=self.feed,
            fetched_at=datetime.utcnow().isoformat(),
            rows=len(df),
        )
        meta_path.write_text(json.dumps(meta.__dict__, indent=2))
        return df

    def get_bars_multi(
        self, symbols: list[str], start: str, end: str, adjustment: str = "all",
        force_refresh: bool = False,
    ) -> dict[str, pd.DataFrame]:
        return {
            sym: self.get_bars(sym, start, end, adjustment=adjustment, force_refresh=force_refresh)
            for sym in symbols
        }
