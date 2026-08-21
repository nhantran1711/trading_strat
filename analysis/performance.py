"""
Performance metrics computed from a closed-trade ledger (backtest.portfolio.Portfolio
output) and, where a time series is needed (Sharpe/Sortino/CAGR/drawdown), a
daily equity curve derived from that ledger.

All ledger-derived metrics (win rate, expectancy, profit factor, ...) are
computed straightforwardly from `trades` alone. Time-series metrics require
converting the event-driven trade ledger into a daily equity series first via
`daily_equity_curve`, aligned to the actual NYSE trading calendar so
non-trading days don't distort annualization.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from data.preprocessing import get_trading_days

TRADING_DAYS_PER_YEAR = 252
MINUTES_PER_SESSION = 390


def daily_equity_curve(trades: pd.DataFrame, starting_equity: float, start: str, end: str) -> pd.Series:
    trading_days = pd.DatetimeIndex(get_trading_days(start, end))
    if trades.empty:
        return pd.Series(starting_equity, index=trading_days)

    daily_pnl = trades.copy()
    daily_pnl["exit_date"] = pd.to_datetime(daily_pnl["exit_time"]).dt.tz_localize(None).dt.normalize()
    pnl_by_day = daily_pnl.groupby("exit_date")["net_pnl"].sum()

    equity = pd.Series(starting_equity, index=trading_days, dtype=float)
    cum_pnl = pnl_by_day.reindex(trading_days, fill_value=0.0).cumsum()
    equity = starting_equity + cum_pnl
    return equity


@dataclass
class PerformanceSummary:
    n_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    expectancy: float          # mean net_pnl per trade, in dollars
    expectancy_r: float        # mean R-multiple per trade
    profit_factor: float       # gross profit / gross loss
    total_return_pct: float
    cagr: float
    sharpe: float
    sortino: float
    max_drawdown_pct: float
    avg_holding_minutes: float
    exposure: float            # fraction of available session-minutes with a position open
    turnover: float            # total traded notional / average equity, over the period
    gross_pnl: float
    net_pnl: float
    total_costs: float


def compute_performance_summary(
    trades: pd.DataFrame, starting_equity: float, start: str, end: str
) -> PerformanceSummary:
    if trades.empty:
        return PerformanceSummary(
            n_trades=0, win_rate=float("nan"), avg_win=float("nan"), avg_loss=float("nan"),
            expectancy=float("nan"), expectancy_r=float("nan"), profit_factor=float("nan"),
            total_return_pct=0.0, cagr=0.0, sharpe=float("nan"), sortino=float("nan"),
            max_drawdown_pct=0.0, avg_holding_minutes=float("nan"), exposure=0.0, turnover=0.0,
            gross_pnl=0.0, net_pnl=0.0, total_costs=0.0,
        )

    wins = trades[trades["net_pnl"] > 0]
    losses = trades[trades["net_pnl"] <= 0]

    win_rate = len(wins) / len(trades)
    avg_win = wins["net_pnl"].mean() if len(wins) else 0.0
    avg_loss = losses["net_pnl"].mean() if len(losses) else 0.0
    expectancy = trades["net_pnl"].mean()
    expectancy_r = trades["r_multiple"].mean()

    gross_profit = wins["net_pnl"].sum()
    gross_loss = -losses["net_pnl"].sum()
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    equity = daily_equity_curve(trades, starting_equity, start, end)
    daily_returns = equity.pct_change().dropna()

    total_return_pct = (equity.iloc[-1] / equity.iloc[0]) - 1.0
    n_days = len(equity)
    years = n_days / TRADING_DAYS_PER_YEAR
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1 if years > 0 and equity.iloc[-1] > 0 else float("nan")

    sharpe = (
        daily_returns.mean() / daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        if daily_returns.std() > 0 else float("nan")
    )
    downside = daily_returns[daily_returns < 0]
    sortino = (
        daily_returns.mean() / downside.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        if len(downside) > 0 and downside.std() > 0 else float("nan")
    )

    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    max_drawdown_pct = drawdown.min()

    avg_holding_minutes = trades["holding_minutes"].mean()

    total_position_minutes = trades["holding_minutes"].sum()
    available_minutes = n_days * MINUTES_PER_SESSION
    exposure = total_position_minutes / available_minutes if available_minutes > 0 else float("nan")

    entry_notional = (trades["entry_price"] * trades["shares"]).sum()
    exit_notional = (trades["exit_price"] * trades["shares"]).sum()
    avg_equity = equity.mean()
    turnover = (entry_notional + exit_notional) / avg_equity if avg_equity > 0 else float("nan")

    gross_pnl = trades["gross_pnl"].sum()
    net_pnl = trades["net_pnl"].sum()
    total_costs = (trades["commission"] + trades["reg_fees"]).sum()

    return PerformanceSummary(
        n_trades=len(trades), win_rate=win_rate, avg_win=avg_win, avg_loss=avg_loss,
        expectancy=expectancy, expectancy_r=expectancy_r, profit_factor=profit_factor,
        total_return_pct=total_return_pct, cagr=cagr, sharpe=sharpe, sortino=sortino,
        max_drawdown_pct=max_drawdown_pct, avg_holding_minutes=avg_holding_minutes,
        exposure=exposure, turnover=turnover, gross_pnl=gross_pnl, net_pnl=net_pnl,
        total_costs=total_costs,
    )


def trade_return_distribution(trades: pd.DataFrame) -> pd.DataFrame:
    """Per-trade return distribution stats, both in R-multiples (risk-normalized,
    the more meaningful unit for a fixed-risk strategy) and as a fraction of
    equity at entry (account-impact terms)."""
    if trades.empty:
        return pd.DataFrame()
    r = trades["r_multiple"]
    pct_of_equity = trades["net_pnl"] / trades["equity_before"]
    stats = {}
    for name, series in [("r_multiple", r), ("pct_of_equity", pct_of_equity)]:
        stats[name] = {
            "mean": series.mean(),
            "std": series.std(),
            "skew": series.skew(),
            "kurtosis": series.kurtosis(),
            "min": series.min(),
            "p5": series.quantile(0.05),
            "p25": series.quantile(0.25),
            "median": series.median(),
            "p75": series.quantile(0.75),
            "p95": series.quantile(0.95),
            "max": series.max(),
        }
    return pd.DataFrame(stats)


def summary_to_dict(summary: PerformanceSummary) -> dict:
    return asdict(summary)
