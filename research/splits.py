"""
Chronological train/validation/test splitting and walk-forward window
generation. Splits are strictly time-ordered (never shuffled) since the whole
point is to hold out a period the strategy/parameters have never been
evaluated against, in either direction -- validation must not leak into
training and the test set must not be touched until Phase 4.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from data.preprocessing import get_trading_days


@dataclass
class DataSplit:
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    test_start: str
    test_end: str


def chronological_split(start: str, end: str, train_frac: float = 0.6, val_frac: float = 0.2) -> DataSplit:
    """Splits [start, end] into three contiguous, non-overlapping periods by
    actual NYSE trading days (not calendar days, so the split respects
    holidays/weekends). `train_frac + val_frac` must be < 1; the remainder is
    the test period, which downstream code should treat as write-once/read-once:
    evaluate on it exactly once, at the end, and never re-tune afterward.
    """
    days = get_trading_days(start, end)
    n = len(days)
    train_end_idx = int(n * train_frac)
    val_end_idx = int(n * (train_frac + val_frac))

    return DataSplit(
        train_start=str(days[0].date()),
        train_end=str(days[train_end_idx - 1].date()),
        val_start=str(days[train_end_idx].date()),
        val_end=str(days[val_end_idx - 1].date()),
        test_start=str(days[val_end_idx].date()),
        test_end=str(days[-1].date()),
    )


def walk_forward_windows(
    start: str, end: str, train_days: int, test_days: int, step_days: int | None = None,
) -> list[DataSplit]:
    """Rolling-origin walk-forward windows: repeatedly train on `train_days`
    trading days and test on the following `test_days`, then roll forward by
    `step_days` (default: test_days, i.e. non-overlapping test windows) and
    repeat until the data runs out. Each window's test period is used exactly
    once and never overlaps a later window's train period going backward in
    time, so information never flows from a test window back into an earlier
    train window.
    """
    step_days = step_days or test_days
    days = get_trading_days(start, end)
    windows = []
    i = 0
    while i + train_days + test_days <= len(days):
        train_slice = days[i: i + train_days]
        test_slice = days[i + train_days: i + train_days + test_days]
        windows.append(
            DataSplit(
                train_start=str(train_slice[0].date()), train_end=str(train_slice[-1].date()),
                val_start=str(train_slice[-1].date()), val_end=str(train_slice[-1].date()),  # unused in walk-forward
                test_start=str(test_slice[0].date()), test_end=str(test_slice[-1].date()),
            )
        )
        i += step_days
    return windows
