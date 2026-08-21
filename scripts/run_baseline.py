"""
Phase 1 -- Baseline: the simplest possible version of the hypothesis, no
optimization, evaluated on the research portion (train+validation) of the
data only. The test period is deliberately never loaded here -- see
research/splits.py and Phase 4 (out-of-sample testing).

Usage:
    python scripts/run_baseline.py

Requires data already fetched via scripts/fetch_data.py (which itself
requires ALPACA_API_KEY/ALPACA_SECRET_KEY). If no cached data is found, this
prints setup instructions and exits rather than fabricating results.
"""
from __future__ import annotations

import json
import logging
import sys

sys.path.insert(0, ".")

import pandas as pd

from analysis.performance import compute_performance_summary, summary_to_dict, trade_return_distribution
from analysis.statistics import bootstrap_ci, permutation_test_two_groups
from analysis.visualization import plot_equity_curve, plot_drawdown, plot_trade_return_distribution
from backtest.engine import run_backtest, run_direction_control
from config import DATA_CACHE_DIR, DEFAULT_EXECUTION, DEFAULT_PARAMS, DEFAULT_PORTFOLIO, REPORTS_DIR, UNIVERSE
from data.loader import AlpacaBarLoader
from data.preprocessing import clean_symbol_bars, get_trading_days
from research.splits import chronological_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_and_clean_universe(symbols: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    loader = AlpacaBarLoader()
    trading_days = pd.DatetimeIndex(get_trading_days(start, end))
    bars_by_symbol = {}
    for symbol in symbols:
        raw = loader.get_bars(symbol, start, end)  # cache-only if already fetched; else raises/empty
        if raw.empty:
            logger.warning("No cached data for %s -- run scripts/fetch_data.py first.", symbol)
            continue
        clean, dropped = clean_symbol_bars(raw, trading_days)
        if len(dropped):
            logger.info("%s: dropped %d low-coverage day(s): %s", symbol, len(dropped), list(dropped.index.date))
        bars_by_symbol[symbol] = clean
    return bars_by_symbol


def main():
    if not any(DATA_CACHE_DIR.glob("*/*.parquet")):
        logger.error(
            "No cached market data found in %s. Run:\n"
            "  1) copy .env.example to .env and fill in ALPACA_API_KEY / ALPACA_SECRET_KEY\n"
            "  2) .venv/Scripts/python.exe scripts/fetch_data.py\n"
            "then re-run this script.",
            DATA_CACHE_DIR,
        )
        sys.exit(1)

    # Determine the full available range from the cache metadata, then hold out
    # the test period (Phase 4) -- Phase 1 only ever touches train+validation.
    from config import DEFAULT_END_DATE, DEFAULT_START_DATE

    split = chronological_split(DEFAULT_START_DATE, DEFAULT_END_DATE, train_frac=0.6, val_frac=0.2)
    logger.info("Data split: train [%s, %s], val [%s, %s], test [%s, %s] (test untouched in Phase 1)",
                split.train_start, split.train_end, split.val_start, split.val_end,
                split.test_start, split.test_end)

    bars_by_symbol = load_and_clean_universe(UNIVERSE, split.train_start, split.val_end)
    if not bars_by_symbol:
        logger.error("No usable data loaded for any symbol. Aborting.")
        sys.exit(1)

    result = run_backtest(
        bars_by_symbol, DEFAULT_PARAMS, DEFAULT_EXECUTION, DEFAULT_PORTFOLIO,
        start=split.train_start, end=split.val_end, require_volume_baseline=False,
    )

    logger.info(
        "Days considered=%d, skipped (no OR)=%d, skipped (no baseline)=%d, signals=%d, fills=%d, trades=%d",
        result.n_days_considered, result.n_days_skipped_no_or, result.n_days_skipped_no_baseline,
        len(result.signals), len(result.fills), len(result.trades),
    )

    if result.trades.empty:
        logger.warning("Zero trades generated -- check data coverage and parameters. No further analysis to run.")
        sys.exit(0)

    summary = compute_performance_summary(
        result.trades, DEFAULT_PORTFOLIO.starting_equity, split.train_start, split.val_end
    )
    logger.info("Performance summary:\n%s", json.dumps(summary_to_dict(summary), indent=2, default=str))

    dist = trade_return_distribution(result.trades)
    logger.info("Trade return distribution:\n%s", dist)

    boot = bootstrap_ci(result.trades["r_multiple"].values, n_boot=10_000)
    logger.info("Bootstrap 95%% CI on mean R-multiple: %.4f [%.4f, %.4f]",
                boot.point_estimate, boot.ci_low, boot.ci_high)

    control_trades = run_direction_control(
        bars_by_symbol, result.signals, DEFAULT_PARAMS, DEFAULT_EXECUTION, DEFAULT_PORTFOLIO, seed=7
    )
    perm = permutation_test_two_groups(
        result.trades["r_multiple"].values, control_trades["r_multiple"].values, n_perm=10_000
    )
    logger.info(
        "Direction-control permutation test: real mean R=%.4f vs control mean R=%.4f, p-value=%.4f "
        "(H0: knowing the TRUE breakout direction adds nothing beyond the execution model's own "
        "structural asymmetry)",
        result.trades["r_multiple"].mean(), control_trades["r_multiple"].mean(), perm.p_value,
    )

    plot_equity_curve(result.equity_curve, title="Phase 1 Baseline -- Equity Curve")
    plot_drawdown(result.equity_curve, title="Phase 1 Baseline -- Drawdown")
    plot_trade_return_distribution(result.trades)

    result.trades.to_csv(REPORTS_DIR / "phase1_trades.csv", index=False)
    (REPORTS_DIR / "phase1_summary.json").write_text(
        json.dumps(summary_to_dict(summary), indent=2, default=str)
    )
    logger.info("Saved trade ledger, summary, and plots to %s", REPORTS_DIR)


if __name__ == "__main__":
    main()
