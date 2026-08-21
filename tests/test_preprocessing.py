import numpy as np
import pandas as pd

from data.preprocessing import (
    compute_or_window_volume_baseline,
    compute_session_vwap,
    fill_missing_minutes,
)
from tests.conftest import make_day_bars


def test_vwap_is_causal_not_using_future_bars():
    """The VWAP at minute t must be identical whether computed from the full
    day or from only the first t+1 bars -- proving it never looks ahead."""
    df = make_day_bars("2024-01-10", breakout_direction="long")
    vwap_full = compute_session_vwap(df)

    truncated = df.iloc[:50]
    vwap_truncated = compute_session_vwap(truncated)

    pd.testing.assert_series_equal(
        vwap_full.iloc[:50].reset_index(drop=True),
        vwap_truncated.reset_index(drop=True),
        check_names=False,
    )


def test_vwap_resets_each_session():
    day1 = make_day_bars("2024-01-10", breakout_direction=None)
    day2 = make_day_bars("2024-01-11", breakout_direction=None, open_price=200.0, or_high=201.0, or_low=199.0)
    both = pd.concat([day1, day2])
    vwap = compute_session_vwap(both)
    # first bar of day 2 should equal day 2's own typical price, not be dragged
    # toward day 1's price level by a running cumulative that spans days.
    day2_first_typical = (day2.iloc[0]["high"] + day2.iloc[0]["low"] + day2.iloc[0]["close"]) / 3
    assert abs(vwap.loc[day2.index[0]] - day2_first_typical) < 1e-6


def test_fill_missing_minutes_flags_synthetic_bars():
    df = make_day_bars("2024-01-10", breakout_direction=None)
    # drop 5 minutes in the middle to simulate a data gap
    gapped = df.drop(df.index[100:105])
    filled = fill_missing_minutes(gapped)
    assert len(filled) == 390
    assert filled["is_synthetic"].sum() == 5
    assert filled.loc[df.index[100], "volume"] == 0
    # synthetic bars should be flat at the previous close, not interpolated/invented
    assert filled.loc[df.index[100], "close"] == filled.loc[df.index[99], "close"]


def test_or_volume_baseline_excludes_current_and_future_days():
    """The baseline for day N must be built strictly from days < N. Concretely:
    changing day N's own OR volume, or any FUTURE day's OR volume, must not
    change day N's baseline value.
    """
    days = pd.date_range("2024-01-02", periods=30, freq="B")  # business days as a stand-in calendar
    frames = []
    for i, d in enumerate(days):
        vol_per_min = 1000 + i * 10  # steadily increasing OR volume day over day
        frames.append(make_day_bars(d.strftime("%Y-%m-%d"), breakout_direction=None, or_volume_per_min=vol_per_min))
    bars = pd.concat(frames)

    baseline_original = compute_or_window_volume_baseline(bars, or_minutes=15, lookback_days=20)

    # Now mutate day 25's (and later) OR volume dramatically and recompute; day 24's
    # baseline (which only depends on days 4-23) must be unchanged.
    target_day = pd.Timestamp(days[24])
    mutated_frames = list(frames)
    mutated_frames[25] = make_day_bars(
        days[25].strftime("%Y-%m-%d"), breakout_direction=None, or_volume_per_min=999_999
    )
    bars_mutated = pd.concat(mutated_frames)
    baseline_mutated = compute_or_window_volume_baseline(bars_mutated, or_minutes=15, lookback_days=20)

    assert baseline_original.loc[target_day] == baseline_mutated.loc[target_day]
