"""Shared synthetic-data fixtures for tests. None of this is real market data --
it exists purely to exercise the pipeline's logic deterministically."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config import EASTERN


def make_day_bars(
    date: str,
    open_price: float = 100.0,
    or_high: float = 101.0,
    or_low: float = 99.0,
    breakout_direction: str | None = "long",
    breakout_minute_offset: int = 20,  # minutes after 09:30 that the breakout close happens
    post_breakout_drift: float = 0.05,  # $ per minute drift after breakout, in breakout direction
    or_volume_per_min: float = 10_000,
    symbol: str = "TEST",
) -> pd.DataFrame:
    """Build one synthetic regular-session day (09:30-16:00, 390 one-minute bars)
    with a controlled opening range and a controlled breakout, for deterministic
    unit testing of opening-range/execution/preprocessing logic.
    """
    idx = pd.date_range(f"{date} 09:30", f"{date} 16:00", freq="1min", tz=EASTERN, inclusive="left")
    n = len(idx)
    rng = np.random.default_rng(0)

    close = np.full(n, open_price)
    high = np.full(n, open_price + 0.1)
    low = np.full(n, open_price - 0.1)
    vol = np.full(n, 500.0)

    # First 15 minutes: opening range, oscillating within [or_low, or_high].
    or_minutes = 15
    close[:or_minutes] = np.linspace(open_price, (or_high + or_low) / 2, or_minutes)
    high[:or_minutes] = or_high
    low[:or_minutes] = or_low
    vol[:or_minutes] = or_volume_per_min

    if breakout_direction is not None:
        bi = breakout_minute_offset
        if breakout_direction == "long":
            close[bi] = or_high + 0.5
            high[bi] = or_high + 0.6
            low[bi] = or_high - 0.1
            for j in range(bi + 1, n):
                close[j] = close[bi] + post_breakout_drift * (j - bi)
                high[j] = close[j] + 0.1
                low[j] = close[j] - 0.1
        else:
            close[bi] = or_low - 0.5
            high[bi] = or_low + 0.1
            low[bi] = or_low - 0.6
            for j in range(bi + 1, n):
                close[j] = close[bi] - post_breakout_drift * (j - bi)
                high[j] = close[j] + 0.1
                low[j] = close[j] - 0.1

    open_ = np.roll(close, 1)
    open_[0] = open_price

    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol, "symbol": symbol},
        index=idx,
    )
    df.index.name = "timestamp"
    return df


@pytest.fixture
def long_breakout_day():
    return make_day_bars("2024-01-10", breakout_direction="long")


@pytest.fixture
def short_breakout_day():
    return make_day_bars("2024-01-10", breakout_direction="short")


@pytest.fixture
def no_breakout_day():
    return make_day_bars("2024-01-10", breakout_direction=None)
