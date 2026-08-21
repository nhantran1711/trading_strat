import pandas as pd

from backtest.execution import TradeFill
from backtest.portfolio import Portfolio
from config import DEFAULT_EXECUTION, PortfolioParams


def _fill(entry_time, exit_time, entry_price=100.0, exit_price=102.0, direction="long", risk_per_share=1.0):
    return TradeFill(
        symbol="TEST", day=pd.Timestamp(entry_time).normalize(), direction=direction,
        signal_time=pd.Timestamp(entry_time) - pd.Timedelta(minutes=1), signal_price=entry_price,
        entry_time=pd.Timestamp(entry_time), entry_price=entry_price,
        stop_price=entry_price - risk_per_share, target_price=entry_price + 2 * risk_per_share,
        exit_time=pd.Timestamp(exit_time), exit_price=exit_price, exit_reason="target",
        holding_minutes=10, risk_per_share=risk_per_share,
        gross_pnl_per_share=(exit_price - entry_price) if direction == "long" else (entry_price - exit_price),
        relative_volume=1.5, or_width=2.0,
    )


def test_risk_based_sizing_risks_configured_fraction_of_equity():
    params = PortfolioParams(starting_equity=100_000, risk_per_trade=0.01,
                              max_position_notional_frac=1.0, max_participation_of_or_volume=1.0)
    execp = DEFAULT_EXECUTION
    portfolio = Portfolio(params, execp)
    fill = _fill("2024-01-10 09:51", "2024-01-10 10:01", risk_per_share=1.0)
    trades = portfolio.run([fill], or_volumes={("TEST", fill.day): 1_000_000})
    # risk_per_trade=1% of 100k = $1000 risk budget / $1 risk-per-share = 1000 shares
    assert trades.iloc[0]["shares"] == 1000


def test_participation_cap_limits_size_on_thin_or_volume():
    params = PortfolioParams(starting_equity=100_000, risk_per_trade=0.5,  # deliberately huge risk ask
                              max_position_notional_frac=1.0, max_participation_of_or_volume=0.01)
    execp = DEFAULT_EXECUTION
    portfolio = Portfolio(params, execp)
    fill = _fill("2024-01-10 09:51", "2024-01-10 10:01", risk_per_share=1.0)
    thin_or_volume = 5_000
    trades = portfolio.run([fill], or_volumes={("TEST", fill.day): thin_or_volume})
    assert trades.iloc[0]["shares"] <= thin_or_volume * 0.01


def test_equity_compounds_sequentially_across_trades():
    params = PortfolioParams(starting_equity=100_000, risk_per_trade=0.01,
                              max_position_notional_frac=1.0, max_participation_of_or_volume=1.0)
    execp = DEFAULT_EXECUTION
    portfolio = Portfolio(params, execp)
    fill1 = _fill("2024-01-10 09:51", "2024-01-10 10:01", entry_price=100, exit_price=102, risk_per_share=1.0)
    fill2 = _fill("2024-01-11 09:51", "2024-01-11 10:01", entry_price=100, exit_price=102, risk_per_share=1.0)
    trades = portfolio.run([fill1, fill2], or_volumes={
        ("TEST", fill1.day): 1_000_000, ("TEST", fill2.day): 1_000_000,
    })
    assert trades.iloc[0]["equity_before"] == 100_000
    assert trades.iloc[1]["equity_before"] == trades.iloc[0]["equity_after"]


def test_costs_reduce_net_pnl_below_gross_pnl():
    params = PortfolioParams(starting_equity=100_000, risk_per_trade=0.01,
                              max_position_notional_frac=1.0, max_participation_of_or_volume=1.0)
    portfolio = Portfolio(params, DEFAULT_EXECUTION)
    fill = _fill("2024-01-10 09:51", "2024-01-10 10:01", risk_per_share=1.0)
    trades = portfolio.run([fill], or_volumes={("TEST", fill.day): 1_000_000})
    row = trades.iloc[0]
    assert row["net_pnl"] < row["gross_pnl"]
    assert row["commission"] > 0
    assert row["reg_fees"] > 0
