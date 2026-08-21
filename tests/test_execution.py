import pandas as pd

from backtest.execution import simulate_trade
from config import DEFAULT_EXECUTION, DEFAULT_PARAMS, ExecutionParams
from strategies.opening_range import Signal
from tests.conftest import make_day_bars


def _signal_from(df, symbol="TEST", direction="long", or_high=101.0, or_low=99.0):
    bi = 20
    return Signal(
        symbol=symbol, day=df.index[0].normalize(), direction=direction,
        signal_time=df.index[bi], signal_price=float(df.iloc[bi]["close"]),
        or_high=or_high, or_low=or_low, or_mid=(or_high + or_low) / 2,
        or_width=or_high - or_low, or_volume=150_000, relative_volume=1.5,
    )


def test_entry_fills_at_next_bar_open_not_signal_close():
    df = make_day_bars("2024-01-10", breakout_direction="long")
    sig = _signal_from(df)
    fill = simulate_trade(sig, df, DEFAULT_PARAMS, DEFAULT_EXECUTION)
    next_bar_open = df.iloc[21]["open"]
    # entry_price should be derived from next bar's open (plus cost haircut), not
    # from the signal bar's close.
    assert fill.entry_time == df.index[21]
    haircut = (DEFAULT_EXECUTION.half_spread_bps + DEFAULT_EXECUTION.slippage_bps) / 10_000
    assert abs(fill.entry_price - next_bar_open * (1 + haircut)) < 1e-9


def test_long_entry_costs_more_than_raw_price():
    df = make_day_bars("2024-01-10", breakout_direction="long")
    sig = _signal_from(df)
    fill = simulate_trade(sig, df, DEFAULT_PARAMS, DEFAULT_EXECUTION)
    raw_next_open = df.iloc[21]["open"]
    assert fill.entry_price > raw_next_open  # buying pays the spread/slippage, never receives it


def test_short_entry_receives_less_than_raw_price():
    df = make_day_bars("2024-01-10", breakout_direction="short")
    sig = _signal_from(df, direction="short")
    fill = simulate_trade(sig, df, DEFAULT_PARAMS, DEFAULT_EXECUTION)
    raw_next_open = df.iloc[21]["open"]
    assert fill.entry_price < raw_next_open  # selling short receives less, never more


def test_target_hit_exits_at_target():
    df = make_day_bars("2024-01-10", breakout_direction="long", post_breakout_drift=0.5)  # fast, clean move
    sig = _signal_from(df)
    fill = simulate_trade(sig, df, DEFAULT_PARAMS, DEFAULT_EXECUTION)
    assert fill.exit_reason == "target"
    assert fill.gross_pnl_per_share > 0


def test_stop_hit_exits_at_stop():
    # Price breaks out then immediately reverses hard through the OR midpoint stop.
    df = make_day_bars("2024-01-10", breakout_direction="long", post_breakout_drift=0.05)
    bi = 21
    df.iloc[bi, df.columns.get_loc("low")] = 90.0  # crashes through the 100.0 stop
    df.iloc[bi, df.columns.get_loc("close")] = 90.5
    sig = _signal_from(df)
    fill = simulate_trade(sig, df, DEFAULT_PARAMS, DEFAULT_EXECUTION)
    assert fill.exit_reason == "stop"
    assert fill.gross_pnl_per_share < 0


def test_same_bar_stop_and_target_defaults_to_conservative_stop():
    df = make_day_bars("2024-01-10", breakout_direction="long", post_breakout_drift=0.05)
    bi = 21
    # Same bar's range spans both the stop (100.0) and a would-be target far above.
    df.iloc[bi, df.columns.get_loc("low")] = 90.0
    df.iloc[bi, df.columns.get_loc("high")] = 300.0
    sig = _signal_from(df)
    fill = simulate_trade(sig, df, DEFAULT_PARAMS, DEFAULT_EXECUTION)
    assert fill.exit_reason == "stop"


def test_time_exit_when_neither_level_hit():
    df = make_day_bars("2024-01-10", breakout_direction="long", post_breakout_drift=0.001)  # barely drifts
    sig = _signal_from(df)
    fill = simulate_trade(sig, df, DEFAULT_PARAMS, DEFAULT_EXECUTION)
    assert fill.exit_reason in ("time", "session_end")
    assert fill.holding_minutes <= DEFAULT_PARAMS.max_holding_minutes


def test_zero_cost_execution_matches_raw_prices():
    """Sanity check: with spread/slippage/commission all zeroed out, the fill
    should reduce to the raw next-bar-open entry with no haircut."""
    df = make_day_bars("2024-01-10", breakout_direction="long")
    sig = _signal_from(df)
    zero_cost = ExecutionParams(half_spread_bps=0, slippage_bps=0, commission_per_share=0, reg_fee_bps_on_sell=0)
    fill = simulate_trade(sig, df, DEFAULT_PARAMS, zero_cost)
    assert abs(fill.entry_price - df.iloc[21]["open"]) < 1e-9
