"""
End-to-end smoke tests for the full pipeline (engine -> portfolio -> performance
-> statistics) run against SYNTHETIC data, not real market data. These exist to
catch integration bugs -- and especially look-ahead / data-leakage bugs -- that
per-module unit tests could miss:

1. On a PURE random walk with no injected drift, the strategy should show NO
   statistically significant edge. If it does, that is a strong signal of a
   bug (most likely a look-ahead leak) rather than a real discovery, since by
   construction there is no directional information in the underlying series.
2. On synthetic data with a deliberately injected continuation drift, the
   strategy SHOULD detect a positive, statistically significant edge -- this
   confirms the pipeline is actually capable of finding a real effect when
   one is present, i.e. that a null result on real data isn't just the
   pipeline being broken/insensitive.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.performance import compute_performance_summary
from analysis.statistics import permutation_test_two_groups
from backtest.engine import run_backtest, run_direction_control
from config import DEFAULT_EXECUTION, DEFAULT_PORTFOLIO, StrategyParams
from data.preprocessing import clean_symbol_bars, get_trading_days


def _make_synthetic_universe(
    n_days: int, drift_strength: float, seed: int, symbols=("SYNA", "SYNB"),
) -> dict[str, pd.DataFrame]:
    trading_days = get_trading_days("2023-01-02", "2023-01-02")  # placeholder, overwritten below
    business_days = pd.bdate_range("2023-01-02", periods=n_days)

    rng = np.random.default_rng(seed)
    out = {}
    for symbol in symbols:
        frames = []
        price = 100.0
        for day in business_days:
            idx = pd.date_range(f"{day.date()} 09:30", f"{day.date()} 16:00", freq="1min",
                                 tz="America/New_York", inclusive="left")
            n = len(idx)
            steps = rng.normal(0, 0.05, size=n)

            # Determine, from the opening-range bars alone (first 15), which side
            # of the open price is dominant -- then apply drift AFTER the OR
            # window in that same direction. This never uses information from
            # after the OR window to decide the OR window itself.
            or_end = 15
            or_walk = np.cumsum(steps[:or_end])
            direction = 1.0 if or_walk[-1] >= 0 else -1.0
            steps[or_end:] += direction * drift_strength

            path = price + np.cumsum(steps)
            close = path
            open_ = np.concatenate([[price], close[:-1]])
            high = np.maximum(open_, close) + rng.uniform(0, 0.05, n)
            low = np.minimum(open_, close) - rng.uniform(0, 0.05, n)
            volume = rng.uniform(5_000, 15_000, n)

            frames.append(pd.DataFrame(
                {"open": open_, "high": high, "low": low, "close": close,
                 "volume": volume, "symbol": symbol}, index=idx,
            ))
            price = close[-1]

        raw = pd.concat(frames)
        raw.index.name = "timestamp"
        trading_days = pd.DatetimeIndex(get_trading_days(str(business_days[0].date()), str(business_days[-1].date())))
        clean, _ = clean_symbol_bars(raw, trading_days)
        out[symbol] = clean
    return out


@pytest.mark.slow
def test_pure_random_walk_shows_no_spurious_edge():
    """Primary check: real trades vs. a direction-randomized control group built
    from the SAME signals (same timing/price/OR levels, direction re-assigned
    50/50) through the SAME execution pipeline. On pure random-walk data with
    no injected drift, true breakout direction carries no information, so real
    trades should be statistically indistinguishable from the control group.
    (A naive "is mean R-multiple significantly different from zero" test does
    NOT hold here -- see the CAVEAT in analysis.statistics.sign_flip_test;
    this is exactly why run_direction_control exists.)
    """
    bars = _make_synthetic_universe(n_days=120, drift_strength=0.0, seed=100)
    params = StrategyParams(min_baseline_days=1)  # short history in this synthetic test
    result = run_backtest(bars, params, DEFAULT_EXECUTION, DEFAULT_PORTFOLIO, require_volume_baseline=False)

    assert len(result.trades) > 20, "need enough trades for the significance check to be meaningful"

    control = run_direction_control(bars, result.signals, params, DEFAULT_EXECUTION, DEFAULT_PORTFOLIO, seed=7)
    perm = permutation_test_two_groups(
        result.trades["r_multiple"].values, control["r_multiple"].values, n_perm=5000, random_state=1
    )
    assert perm.p_value > 0.01, (
        f"Real trades differ significantly from the direction-randomized control (p={perm.p_value:.4f}) "
        "on data with NO injected directional edge -- this points at a look-ahead or data-leakage bug."
    )


@pytest.mark.slow
def test_injected_momentum_edge_is_detected():
    bars = _make_synthetic_universe(n_days=120, drift_strength=0.03, seed=101)
    params = StrategyParams(min_baseline_days=1)
    result = run_backtest(bars, params, DEFAULT_EXECUTION, DEFAULT_PORTFOLIO, require_volume_baseline=False)

    assert len(result.trades) > 20

    control = run_direction_control(bars, result.signals, params, DEFAULT_EXECUTION, DEFAULT_PORTFOLIO, seed=7)
    perm = permutation_test_two_groups(
        result.trades["r_multiple"].values, control["r_multiple"].values, n_perm=5000, random_state=1
    )
    summary = compute_performance_summary(
        result.trades, DEFAULT_PORTFOLIO.starting_equity,
        str(result.trades["entry_time"].min().date()), str(result.trades["exit_time"].max().date()),
    )
    assert result.trades["r_multiple"].mean() > control["r_multiple"].mean()
    assert perm.p_value < 0.05, "pipeline failed to detect a deliberately injected, real continuation edge"
