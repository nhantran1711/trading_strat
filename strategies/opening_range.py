"""
Opening-range breakout signal generation.

This module is deliberately pure and stateless: given one day's 1-minute bars
for one symbol (already cleaned/aligned by data/preprocessing.py) plus a
pre-computed historical volume baseline (which must itself have been built
from PRIOR days only -- see data.preprocessing.compute_or_window_volume_baseline),
it computes the opening range and, if applicable, a single breakout signal.

No look-ahead by construction: `generate_signal` scans bars strictly in time
order and returns as soon as it finds the first qualifying breakout close --
it never inspects a bar later than the one it signals on, and the volume
baseline it compares against is passed in already computed from the past.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from config import StrategyParams


@dataclass
class OpeningRange:
    day: pd.Timestamp
    symbol: str
    or_high: float
    or_low: float
    or_mid: float
    or_width: float
    or_volume: float
    or_end_time: pd.Timestamp


@dataclass
class Signal:
    symbol: str
    day: pd.Timestamp
    direction: str  # "long" or "short"
    signal_time: pd.Timestamp     # timestamp of the bar whose CLOSE triggered the signal
    signal_price: float           # that bar's close (theoretical trigger price, NOT the fill price)
    or_high: float
    or_low: float
    or_mid: float
    or_width: float
    or_volume: float
    relative_volume: float | None  # OR-window volume / historical baseline, if baseline available


def compute_opening_range(day_bars: pd.DataFrame, symbol: str, or_minutes: int) -> OpeningRange | None:
    """`day_bars` must be a single day's regular-session bars for one symbol,
    sorted by time, starting at the session open. Returns None if there are
    not enough bars to form a complete opening range (e.g. a truncated day).
    """
    if day_bars.empty:
        return None
    day_start = day_bars.index[0]
    or_end_time = day_start + pd.Timedelta(minutes=or_minutes)
    or_bars = day_bars[day_bars.index < or_end_time]
    if len(or_bars) < or_minutes:
        return None  # incomplete opening range (e.g. early close / data gap) -> skip the day

    or_high = float(or_bars["high"].max())
    or_low = float(or_bars["low"].min())
    return OpeningRange(
        day=day_start.normalize(),
        symbol=symbol,
        or_high=or_high,
        or_low=or_low,
        or_mid=(or_high + or_low) / 2.0,
        or_width=or_high - or_low,
        or_volume=float(or_bars["volume"].sum()),
        or_end_time=or_end_time,
    )


def generate_signal(
    day_bars: pd.DataFrame,
    opening_range: OpeningRange,
    params: StrategyParams,
    volume_baseline: float | None,
) -> Signal | None:
    """Scan bars after the opening range, in time order, for the FIRST 1-minute
    close outside [or_low, or_high]. Stops scanning at `params.cutoff_time` (no
    new entries after that). Applies `params.relative_volume_threshold` as a
    pass/fail filter if set (None => Phase-1 baseline behavior: no volume
    filter at all, direction-only).

    Returns at most one Signal per day per symbol -- the baseline model takes
    only the first breakout, matching "enter on the first 1-minute close
    outside the opening range."
    """
    day_str = opening_range.day.strftime("%Y-%m-%d")
    cutoff_ts = pd.Timestamp(f"{day_str} {params.cutoff_time}", tz=day_bars.index.tz)

    post_or = day_bars[
        (day_bars.index >= opening_range.or_end_time) & (day_bars.index < cutoff_ts)
    ]

    relative_volume = (
        opening_range.or_volume / volume_baseline
        if volume_baseline and volume_baseline > 0
        else None
    )

    if params.relative_volume_threshold is not None:
        if relative_volume is None or relative_volume < params.relative_volume_threshold:
            return None  # fails the volume filter -> no trade this day, regardless of price action

    for ts, bar in post_or.iterrows():
        close = bar["close"]
        if close > opening_range.or_high:
            direction = "long"
        elif close < opening_range.or_low:
            direction = "short"
        else:
            continue

        return Signal(
            symbol=opening_range.symbol,
            day=opening_range.day,
            direction=direction,
            signal_time=ts,
            signal_price=float(close),
            or_high=opening_range.or_high,
            or_low=opening_range.or_low,
            or_mid=opening_range.or_mid,
            or_width=opening_range.or_width,
            or_volume=opening_range.or_volume,
            relative_volume=relative_volume,
        )
    return None
