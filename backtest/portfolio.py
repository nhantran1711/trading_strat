"""
Position sizing, transaction-cost application, and equity tracking.

Sizing is risk-based: each trade risks a fixed fraction of CURRENT equity
(not starting equity) against the entry-to-stop distance, so the position
size adapts as the account compounds. Equity is updated sequentially, trade
by trade, in entry-time order -- so a trade's size depends on the realized
P&L of every trade that has already CLOSED before it, not on trades that are
still open concurrently. This is a simplifying assumption for a
multi-symbol portfolio (it does not reserve capital for open positions) and
is documented as such; it does not look ahead, since it only ever uses
information from trades that already closed in the past.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import pandas as pd

from backtest.execution import TradeFill
from config import ExecutionParams, PortfolioParams


@dataclass
class ClosedTrade:
    symbol: str
    day: pd.Timestamp
    direction: str
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: pd.Timestamp
    exit_price: float
    exit_reason: str
    shares: int
    stop_price: float
    target_price: float
    risk_per_share: float
    holding_minutes: int
    gross_pnl: float
    commission: float
    reg_fees: float
    net_pnl: float
    r_multiple: float          # net_pnl / (risk_per_share * shares) -- costs included
    relative_volume: float | None
    or_width: float
    equity_before: float
    equity_after: float


def _size_shares(fill: TradeFill, equity: float, or_volume: float, params: PortfolioParams) -> int:
    risk_dollars = equity * params.risk_per_trade
    shares_by_risk = risk_dollars / fill.risk_per_share

    max_notional = equity * params.max_position_notional_frac
    shares_by_notional = max_notional / fill.entry_price

    shares_by_participation = or_volume * params.max_participation_of_or_volume

    shares = min(shares_by_risk, shares_by_notional, shares_by_participation)
    return max(int(shares), 0)


def _costs(fill: TradeFill, shares: int, execp: ExecutionParams) -> tuple[float, float]:
    commission = execp.commission_per_share * shares * 2  # entry leg + exit leg
    # Regulatory fees apply on the "sell" notional of the round trip: exit
    # notional for a long (sell to close), entry notional for a short (sell to open).
    sell_notional = (fill.exit_price if fill.direction == "long" else fill.entry_price) * shares
    reg_fees = sell_notional * (execp.reg_fee_bps_on_sell / 10_000.0)
    return commission, reg_fees


class Portfolio:
    def __init__(self, params: PortfolioParams, execp: ExecutionParams):
        self.params = params
        self.execp = execp
        self.equity = params.starting_equity
        self.trades: list[ClosedTrade] = []

    def run(self, fills: list[TradeFill], or_volumes: dict[tuple[str, pd.Timestamp], float]) -> pd.DataFrame:
        """`or_volumes` maps (symbol, day) -> that day's opening-range volume, used
        to cap position size by participation. Processes fills in entry-time
        order so equity compounds causally.
        """
        ordered = sorted(fills, key=lambda f: f.entry_time)
        for fill in ordered:
            or_vol = or_volumes.get((fill.symbol, fill.day), float("inf"))
            shares = _size_shares(fill, self.equity, or_vol, self.params)
            if shares <= 0:
                continue

            gross_pnl = fill.gross_pnl_per_share * shares
            commission, reg_fees = _costs(fill, shares, self.execp)
            net_pnl = gross_pnl - commission - reg_fees

            equity_before = self.equity
            self.equity += net_pnl

            risk_dollars = fill.risk_per_share * shares
            r_multiple = net_pnl / risk_dollars if risk_dollars > 0 else 0.0

            self.trades.append(
                ClosedTrade(
                    symbol=fill.symbol,
                    day=fill.day,
                    direction=fill.direction,
                    entry_time=fill.entry_time,
                    entry_price=fill.entry_price,
                    exit_time=fill.exit_time,
                    exit_price=fill.exit_price,
                    exit_reason=fill.exit_reason,
                    shares=shares,
                    stop_price=fill.stop_price,
                    target_price=fill.target_price,
                    risk_per_share=fill.risk_per_share,
                    holding_minutes=fill.holding_minutes,
                    gross_pnl=gross_pnl,
                    commission=commission,
                    reg_fees=reg_fees,
                    net_pnl=net_pnl,
                    r_multiple=r_multiple,
                    relative_volume=fill.relative_volume,
                    or_width=fill.or_width,
                    equity_before=equity_before,
                    equity_after=self.equity,
                )
            )
        return self.trades_df()

    def trades_df(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame(columns=[f.name for f in ClosedTrade.__dataclass_fields__.values()])
        return pd.DataFrame([asdict(t) for t in self.trades])
