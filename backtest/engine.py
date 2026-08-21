"""
Orchestrates the full day-by-day, symbol-by-symbol backtest: opening range ->
signal -> execution fill -> portfolio sizing/costs -> trade ledger.

This is the only place that stitches the other modules together, which keeps
each of them independently testable. The engine itself does no price-level
math -- it just feeds each day's bars through the pipeline in the correct
order and respects the requested date window (used for train/validation/test
splits and walk-forward windows: the engine only ever sees bars the caller
handed it, so restricting the input `bars_by_symbol` to a date range is
sufficient to prevent it from training/testing across the boundary).
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from backtest.execution import TradeFill, simulate_trade
from backtest.portfolio import Portfolio
from config import ExecutionParams, PortfolioParams, StrategyParams
from data.preprocessing import compute_or_window_volume_baseline
from strategies.opening_range import Signal, compute_opening_range, generate_signal


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity_curve: pd.Series
    fills: list[TradeFill]
    signals: list[Signal]
    n_days_considered: int
    n_days_skipped_no_or: int
    n_days_skipped_no_baseline: int


def run_backtest(
    bars_by_symbol: dict[str, pd.DataFrame],
    params: StrategyParams,
    execp: ExecutionParams,
    portp: PortfolioParams,
    start: str | None = None,
    end: str | None = None,
    require_volume_baseline: bool = False,
) -> BacktestResult:
    all_fills: list[TradeFill] = []
    all_signals: list[Signal] = []
    or_volumes: dict[tuple[str, pd.Timestamp], float] = {}

    n_considered = n_skip_or = n_skip_baseline = 0

    for symbol, bars in bars_by_symbol.items():
        if bars.empty:
            continue

        baseline_series = compute_or_window_volume_baseline(
            bars, params.or_minutes, params.volume_baseline_lookback_days
        )

        by_day = bars.groupby(bars.index.date)
        for date, day_bars in by_day:
            day_ts = pd.Timestamp(date)
            if start and day_ts < pd.Timestamp(start):
                continue
            if end and day_ts > pd.Timestamp(end):
                continue

            n_considered += 1
            opening_range = compute_opening_range(day_bars, symbol, params.or_minutes)
            if opening_range is None:
                n_skip_or += 1
                continue

            baseline_value = baseline_series.get(day_ts)
            baseline_value = None if (baseline_value is None or pd.isna(baseline_value)) else float(baseline_value)

            if require_volume_baseline and baseline_value is None:
                n_skip_baseline += 1
                continue

            or_volumes[(symbol, opening_range.day)] = opening_range.or_volume

            signal = generate_signal(day_bars, opening_range, params, baseline_value)
            if signal is None:
                continue
            all_signals.append(signal)

            fill = simulate_trade(signal, day_bars, params, execp)
            if fill is not None:
                all_fills.append(fill)

    portfolio = Portfolio(portp, execp)
    trades_df = portfolio.run(all_fills, or_volumes)

    if not trades_df.empty:
        equity_curve = pd.concat(
            [pd.Series([portp.starting_equity], index=[trades_df["entry_time"].min()]),
             trades_df.set_index("exit_time")["equity_after"]]
        ).sort_index()
    else:
        equity_curve = pd.Series([portp.starting_equity], index=[pd.Timestamp.now()])

    return BacktestResult(
        trades=trades_df,
        equity_curve=equity_curve,
        fills=all_fills,
        signals=all_signals,
        n_days_considered=n_considered,
        n_days_skipped_no_or=n_skip_or,
        n_days_skipped_no_baseline=n_skip_baseline,
    )


def run_direction_control(
    bars_by_symbol: dict[str, pd.DataFrame],
    signals: list[Signal],
    params: StrategyParams,
    execp: ExecutionParams,
    portp: PortfolioParams,
    seed: int = 0,
) -> pd.DataFrame:
    """Builds the correct null-hypothesis control group for testing whether
    breakout DIRECTION carries information, as opposed to testing against a
    naive "expectancy = 0" assumption.

    Why this is needed: the execution model is deliberately asymmetric (a 2:1
    reward:risk target, and a conservative same-bar tie-break that assumes the
    stop is hit whenever a bar's range could have hit either level first).
    Both of those mechanically bias expectancy away from zero even when
    direction carries NO information at all -- e.g. under a pure random walk,
    naive "is mean R-multiple significantly different from zero" tests can
    reject the null purely from execution mechanics, not from a real edge
    (this was caught by tests/test_engine_integration.py during development).

    The fix: take the REAL signals (same symbol, same signal_time, same
    signal_price, same OR levels -- i.e. the same underlying price path) and
    randomly re-assign each one's direction with 50/50 probability,
    independent of what the true breakout direction was. Run those through
    the IDENTICAL execution/portfolio pipeline, with the stop placed the same
    RISK DISTANCE from entry as the true setup implied (|signal_price -
    or_mid|), but on the correct side of entry for whichever direction was
    randomly assigned (see `stop_distance_override` in
    `backtest.execution.simulate_trade` -- reusing `or_mid` as a literal price
    level for a flipped direction would put the stop on the wrong side of
    entry, which is a degenerate/nonsensical setup, not a fair control).
    Because the control group experiences the exact same execution
    asymmetries, cost structure, and risk magnitude, comparing real trades to
    this control (via analysis.statistics.permutation_test_two_groups)
    isolates whether knowing the TRUE breakout direction adds value beyond
    what the execution mechanics alone would produce from a coin flip.
    """
    rng = np.random.default_rng(seed)
    control_signals = []
    for sig in signals:
        flipped_direction = rng.choice(["long", "short"])
        stop_distance = abs(sig.signal_price - sig.or_mid)
        control_signals.append((replace(sig, direction=flipped_direction), stop_distance))

    or_volumes: dict[tuple[str, pd.Timestamp], float] = {}
    fills: list[TradeFill] = []
    for sig, stop_distance in control_signals:
        day_bars = bars_by_symbol[sig.symbol]
        day_bars = day_bars[day_bars.index.date == sig.day.date()]
        fill = simulate_trade(sig, day_bars, params, execp, stop_distance_override=stop_distance)
        if fill is not None:
            fills.append(fill)
            or_volumes[(sig.symbol, sig.day)] = sig.or_volume

    portfolio = Portfolio(portp, execp)
    return portfolio.run(fills, or_volumes)
