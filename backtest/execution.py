"""
Trade execution simulation: turns a strategy Signal into a realistic per-share
fill outcome (entry price, exit price, exit reason), explicitly modeling that
you cannot transact at a theoretical signal price with zero cost or delay.

Key realism choices, all deliberate and documented (and all swept in Phase 5):

* Entry is filled at the NEXT bar's open, not the signal bar's close. The
  signal is only knowable once that bar's close prints, so trading at that
  exact price is a look-ahead / zero-latency fantasy. The next bar's open is
  the earliest price a real order could plausibly touch.
* Both entry and exit pay half the bid-ask spread plus a slippage allowance,
  always AGAINST the trader (buy higher, sell lower) -- never in the
  trader's favor.
* Stop/target checks use the bar's high/low, not just its close, since a
  stop or limit can be touched intrabar. If a single bar's range contains
  both the stop and the target, the conservative assumption is that the
  stop was hit first (worst case), per `ExecutionParams.conservative_same_bar_fill`.
* Time-based exit (max holding period or session close, whichever is first)
  exits at that bar's close, with the same spread/slippage penalty as any
  other market exit.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from config import ExecutionParams, StrategyParams
from strategies.opening_range import Signal


@dataclass
class TradeFill:
    symbol: str
    day: pd.Timestamp
    direction: str
    signal_time: pd.Timestamp
    signal_price: float
    entry_time: pd.Timestamp
    entry_price: float          # after spread + slippage
    stop_price: float
    target_price: float
    exit_time: pd.Timestamp
    exit_price: float           # after spread + slippage
    exit_reason: str            # "stop" | "target" | "time" | "session_end"
    holding_minutes: int
    risk_per_share: float
    gross_pnl_per_share: float
    relative_volume: float | None
    or_width: float


def _apply_cost(price: float, direction: str, side: str, execp: ExecutionParams) -> float:
    """Move `price` against the trader by (half-spread + slippage).
    side: "buy" or "sell" -- the actual transaction side of this leg
    (independent of long/short: a short's entry is a sell, its exit is a buy).
    """
    haircut_bps = execp.half_spread_bps + execp.slippage_bps
    haircut = haircut_bps / 10_000.0
    if side == "buy":
        return price * (1 + haircut)
    return price * (1 - haircut)


def simulate_trade(
    signal: Signal,
    day_bars: pd.DataFrame,
    params: StrategyParams,
    execp: ExecutionParams,
    stop_distance_override: float | None = None,
) -> TradeFill | None:
    """`day_bars` is the full day's session bars for this symbol (used to find the
    bar immediately after the signal bar, and to walk forward from there).
    Returns None if there is no bar after the signal (can't fill) -- rare given
    the 11:00 cutoff leaves ample bars before the 16:00 close.

    Stop placement: by default the stop is the fixed OR-midpoint PRICE LEVEL
    (`signal.or_mid`), matching the baseline model ("stop-loss at the opening
    range midpoint") -- this is only a sensible stop because a genuine breakout
    signal's entry is, by construction, on the correct side of or_mid (a long
    breakout enters above or_high > or_mid; a short breakout enters below
    or_low < or_mid). `stop_distance_override`, if given, instead places the
    stop at `entry_price -/+ stop_distance_override` (correct side for
    whichever `signal.direction` is passed) -- used by
    `backtest.engine.run_direction_control` to build a randomized-direction
    null control that preserves the true risk MAGNITUDE without inheriting a
    stop level that would be on the wrong side of entry for the flipped
    direction (an earlier version of that control reused `or_mid` directly for
    flipped trades, which silently produced a stop above entry for "long"
    controls and below entry for "short" controls -- a nonsensical, mislabeled
    near-instant "loss" that was actually a disguised win; caught by
    tests/test_engine_integration.py).
    """
    after_signal = day_bars[day_bars.index > signal.signal_time]
    if after_signal.empty:
        return None

    entry_bar_ts = after_signal.index[0]
    entry_bar = after_signal.loc[entry_bar_ts]
    entry_side = "buy" if signal.direction == "long" else "sell"
    entry_price = _apply_cost(float(entry_bar["open"]), signal.direction, entry_side, execp)

    if stop_distance_override is not None:
        stop_price = (
            entry_price - stop_distance_override if signal.direction == "long"
            else entry_price + stop_distance_override
        )
    else:
        stop_price = signal.or_mid
    risk_per_share = abs(entry_price - stop_price)
    if risk_per_share <= 0:
        return None  # degenerate OR (or_mid == entry) -- cannot size a stop, skip

    if signal.direction == "long":
        target_price = entry_price + params.reward_risk * risk_per_share
    else:
        target_price = entry_price - params.reward_risk * risk_per_share

    # Walk forward from the entry bar itself (its high/low AFTER the open can
    # still hit the stop or target the same minute) through the max holding window.
    window = day_bars[day_bars.index >= entry_bar_ts].iloc[: params.max_holding_minutes]

    exit_side = "sell" if signal.direction == "long" else "buy"
    exit_time = None
    exit_price = None
    exit_reason = None

    for ts, bar in window.iterrows():
        hit_stop, hit_target = _check_levels(bar, signal.direction, stop_price, target_price)
        if hit_stop and hit_target:
            if execp.conservative_same_bar_fill:
                exit_time, exit_price, exit_reason = ts, stop_price, "stop"
            else:
                exit_time, exit_price, exit_reason = ts, target_price, "target"
            break
        if hit_stop:
            exit_time, exit_price, exit_reason = ts, stop_price, "stop"
            break
        if hit_target:
            exit_time, exit_price, exit_reason = ts, target_price, "target"
            break

    if exit_time is None:
        # Neither level touched within the holding window -> time-based exit at
        # the last bar's close (or session end, whichever came first, since
        # `window` is already truncated to available session bars).
        last_ts = window.index[-1]
        last_bar = window.loc[last_ts]
        exit_time = last_ts
        exit_price = float(last_bar["close"])
        exit_reason = "time" if len(window) >= params.max_holding_minutes else "session_end"

    exit_price = _apply_cost(exit_price, signal.direction, exit_side, execp)

    gross_pnl_per_share = (
        (exit_price - entry_price) if signal.direction == "long" else (entry_price - exit_price)
    )
    holding_minutes = int((exit_time - entry_bar_ts) / pd.Timedelta(minutes=1)) + 1

    return TradeFill(
        symbol=signal.symbol,
        day=signal.day,
        direction=signal.direction,
        signal_time=signal.signal_time,
        signal_price=signal.signal_price,
        entry_time=entry_bar_ts,
        entry_price=entry_price,
        stop_price=stop_price,
        target_price=target_price,
        exit_time=exit_time,
        exit_price=exit_price,
        exit_reason=exit_reason,
        holding_minutes=holding_minutes,
        risk_per_share=risk_per_share,
        gross_pnl_per_share=gross_pnl_per_share,
        relative_volume=signal.relative_volume,
        or_width=signal.or_width,
    )


def _check_levels(bar: pd.Series, direction: str, stop_price: float, target_price: float) -> tuple[bool, bool]:
    if direction == "long":
        return bool(bar["low"] <= stop_price), bool(bar["high"] >= target_price)
    return bool(bar["high"] >= stop_price), bool(bar["low"] <= target_price)
