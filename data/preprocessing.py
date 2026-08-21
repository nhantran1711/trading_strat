"""
Cleaning, session-alignment, and derived-feature computation for 1-minute bars.

Everything in this module is written to be CAUSAL: any per-minute quantity
(VWAP, relative volume, etc.) is computed using only that minute's bar and
earlier bars/days. Nothing here ever uses information from later in the same
day or from future trading days. This is the module most responsible for
preventing look-ahead bias, so functions are kept small and single-purpose to
make that easy to audit (and to unit test -- see tests/test_preprocessing.py).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

from config import EASTERN, MARKET_CLOSE, MARKET_OPEN

logger = logging.getLogger(__name__)

NYSE = mcal.get_calendar("XNYS")


def get_trading_days(start: str, end: str) -> pd.DatetimeIndex:
    """Valid NYSE regular-session trading days in [start, end], from the official
    exchange calendar -- used to detect holidays and to distinguish a real market
    holiday (expected, not a gap) from a genuine data gap (missing day we do
    have data for on either side, unexpected)."""
    schedule = NYSE.schedule(start_date=start, end_date=end)
    return pd.DatetimeIndex(schedule.index.date)


def to_eastern(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(EASTERN)
    return df


def filter_regular_session(
    df: pd.DataFrame, start: str = MARKET_OPEN, end: str = MARKET_CLOSE
) -> pd.DataFrame:
    """Drop premarket/after-hours bars. Regular session is [start, end) -- the
    16:00 bar itself (if present) represents the last minute of the session and
    is excluded here since our strategy never holds into it via the 60-min /
    11:00-cutoff rules, but callers doing end-of-day analysis can pass end='16:01'.
    """
    df = to_eastern(df)
    return df.between_time(start, end, inclusive="left")


def missing_minutes_report(df: pd.DataFrame, trading_days: pd.DatetimeIndex) -> pd.DataFrame:
    """For each expected trading day, count how many of the 390 regular-session
    minutes are actually present. Used to flag days with excessive gaps (halts,
    vendor outages) BEFORE they silently distort opening-range or VWAP
    calculations. Does not modify data -- purely diagnostic.
    """
    df = filter_regular_session(df)
    counts = df.groupby(df.index.date).size()
    counts.index = pd.DatetimeIndex(counts.index)
    report = pd.DataFrame({"day": trading_days})
    report["bars_present"] = report["day"].map(counts).fillna(0).astype(int)
    report["expected_bars"] = 390
    report["pct_present"] = report["bars_present"] / report["expected_bars"]
    return report.set_index("day")


def session_minute_grid(day: pd.Timestamp, start: str = MARKET_OPEN, end: str = MARKET_CLOSE) -> pd.DatetimeIndex:
    day_str = pd.Timestamp(day).strftime("%Y-%m-%d")
    start_ts = pd.Timestamp(f"{day_str} {start}", tz=EASTERN)
    end_ts = pd.Timestamp(f"{day_str} {end}", tz=EASTERN)
    return pd.date_range(start_ts, end_ts, freq="1min", inclusive="left")


def fill_missing_minutes(day_df: pd.DataFrame) -> pd.DataFrame:
    """Reindex a single day's bars onto the full 09:30-16:00 minute grid. Missing
    minutes (no trades printed that minute -- rare but real for the megacap/ETF
    universe here) are filled as a FLAT bar at the previous close with volume=0,
    and flagged `is_synthetic=True`. This is a deliberate, disclosed assumption
    (a "no trade" minute, not an invented price move) rather than silently
    interpolating or forward-filling without a trace. Only ever uses PRIOR bars
    within the same day -- never a later bar -- so it introduces no look-ahead.
    """
    if day_df.empty:
        return day_df
    day = day_df.index[0].normalize()
    grid = session_minute_grid(day)
    out = day_df.reindex(grid)
    out["is_synthetic"] = out["close"].isna()
    out["close"] = out["close"].ffill()
    out["open"] = out["open"].fillna(out["close"])
    out["high"] = out["high"].fillna(out["close"])
    out["low"] = out["low"].fillna(out["close"])
    out["volume"] = out["volume"].fillna(0)
    if "symbol" in out.columns:
        out["symbol"] = out["symbol"].ffill().bfill()
    out.index.name = "timestamp"
    return out


def clean_symbol_bars(
    df: pd.DataFrame, trading_days: pd.DatetimeIndex, max_missing_frac: float = 0.05
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full per-symbol cleaning pipeline: restrict to regular session, reindex
    each day onto the full minute grid (flagging synthetic fill-in bars), and
    drop any day whose ORIGINAL (pre-fill) bar coverage was too sparse to trust
    (default: more than 5% of the session missing -- e.g. a vendor outage or a
    trading halt lasting most of the day). Returns (clean_bars, dropped_days_report).
    """
    df = filter_regular_session(df)
    report = missing_minutes_report(df, trading_days)
    bad_days = report[report["pct_present"] < (1 - max_missing_frac)]

    cleaned_days = []
    for day in trading_days:
        if day in bad_days.index:
            continue
        day_df = df[df.index.date == day.date()]
        if day_df.empty:
            continue
        cleaned_days.append(fill_missing_minutes(day_df))

    if not cleaned_days:
        return df.iloc[0:0], bad_days
    clean = pd.concat(cleaned_days).sort_index()
    return clean, bad_days


def compute_session_vwap(df: pd.DataFrame) -> pd.Series:
    """Causal, session-resetting VWAP: cumulative sum(typical_price * volume) /
    cumulative sum(volume), reset at the start of each trading day. Because it
    is a running cumulative computed bar-by-bar, the VWAP value at minute t uses
    only bars up to and including t -- never a later bar. This matches what a
    live trading system would actually see intraday.
    """
    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = typical_price * df["volume"]
    day_key = df.index.date
    cum_pv = pv.groupby(day_key).cumsum()
    cum_vol = df["volume"].groupby(day_key).cumsum().replace(0, np.nan)
    vwap = cum_pv / cum_vol
    vwap.name = "vwap"
    return vwap


def compute_or_window_volume_baseline(
    df: pd.DataFrame, or_minutes: int, lookback_days: int
) -> pd.Series:
    """For each trading day, the rolling MEAN opening-range-window volume over the
    trailing `lookback_days` PRIOR days (the day itself and all future days are
    excluded via `shift(1)` before the rolling window is applied). This is the
    "appropriate historical baseline" the strategy compares same-day OR volume
    against to judge whether it is "unusually high" -- and because it is built
    strictly from the past, a day's baseline is identical whether computed
    live on that morning or in hindsight years later, which is exactly the
    walk-forward-safety property required.
    """
    df = filter_regular_session(df)
    or_bars = df.between_time(
        _time_str(0), _time_str(or_minutes), inclusive="left"
    )
    daily_or_volume = or_bars.groupby(or_bars.index.date)["volume"].sum()
    daily_or_volume.index = pd.DatetimeIndex(daily_or_volume.index)
    baseline = daily_or_volume.shift(1).rolling(window=lookback_days, min_periods=lookback_days).mean()
    baseline.name = "or_volume_baseline"
    return baseline


def _time_str(minutes_after_open: int) -> str:
    open_h, open_m = (int(x) for x in _MARKET_OPEN_PARTS)
    total = open_h * 60 + open_m + minutes_after_open
    return f"{total // 60:02d}:{total % 60:02d}"


_MARKET_OPEN_PARTS = MARKET_OPEN.split(":")


def detect_price_discontinuities(df: pd.DataFrame, threshold: float = 0.20) -> pd.DataFrame:
    """Flag overnight close-to-open gaps larger than `threshold` (default 20%).
    Legitimate on rare binary-event days, but also the classic symptom of an
    unadjusted stock split slipping through -- surfaced here for manual audit,
    never auto-corrected silently.
    """
    daily_close = df["close"].groupby(df.index.date).last()
    daily_open = df["open"].groupby(df.index.date).first()
    daily_close.index = pd.DatetimeIndex(daily_close.index)
    daily_open.index = pd.DatetimeIndex(daily_open.index)
    prev_close = daily_close.shift(1)
    gap = (daily_open - prev_close) / prev_close
    flagged = gap[gap.abs() > threshold]
    return pd.DataFrame({"prev_close": prev_close.loc[flagged.index],
                          "next_open": daily_open.loc[flagged.index],
                          "gap_pct": flagged})
