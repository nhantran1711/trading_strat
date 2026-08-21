"""
Runs Phases 1-5 end to end as a batch script (mirrors research/opening_range_research.ipynb)
and writes a single JSON results bundle to reports/full_research_results.json plus the
plots, for compiling the final RESEARCH_REPORT.md. Test period is only touched in the
clearly-marked Phase 4 section, run once.
"""
from __future__ import annotations

import json
import logging
import sys

sys.path.insert(0, ".")

import pandas as pd

from analysis.performance import compute_performance_summary, summary_to_dict, trade_return_distribution
from analysis.statistics import block_bootstrap_ci, multiple_testing_correction, permutation_test_two_groups
from analysis.visualization import (
    plot_drawdown, plot_equity_curve, plot_metric_by_bucket, plot_trade_return_distribution,
)
from backtest.engine import run_backtest, run_direction_control
from config import (
    DATA_CACHE_DIR, DEFAULT_END_DATE, DEFAULT_EXECUTION, DEFAULT_PARAMS, DEFAULT_PORTFOLIO,
    DEFAULT_START_DATE, REPORTS_DIR, UNIVERSE, ExecutionParams, StrategyParams,
)
from data.loader import AlpacaBarLoader
from data.preprocessing import clean_symbol_bars, get_trading_days
from research.splits import chronological_split, walk_forward_windows

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_universe(start, end):
    loader = AlpacaBarLoader()
    trading_days = pd.DatetimeIndex(get_trading_days(start, end))
    out = {}
    for symbol in UNIVERSE:
        raw = loader.get_bars(symbol, start, end)
        if raw.empty:
            continue
        clean, _ = clean_symbol_bars(raw, trading_days)
        out[symbol] = clean
    return out


def summary_row(label, trades, start, end):
    if trades.empty:
        return {"label": label, "n_trades": 0}
    s = compute_performance_summary(trades, DEFAULT_PORTFOLIO.starting_equity, start, end)
    d = summary_to_dict(s)
    d["label"] = label
    return d


def main():
    results = {}

    split = chronological_split(DEFAULT_START_DATE, DEFAULT_END_DATE, train_frac=0.6, val_frac=0.2)
    results["split"] = split.__dict__
    logger.info("Split: %s", split)

    bars_full = load_universe(DEFAULT_START_DATE, DEFAULT_END_DATE)
    assert bars_full, "no data loaded"

    # ---------------- Phase 1: baseline (train+val) ----------------
    p1 = run_backtest(bars_full, DEFAULT_PARAMS, DEFAULT_EXECUTION, DEFAULT_PORTFOLIO,
                       start=split.train_start, end=split.val_end)
    results["phase1"] = summary_row("phase1_baseline_train_val", p1.trades, split.train_start, split.val_end)
    results["phase1"]["n_signals"] = len(p1.signals)
    results["phase1"]["dist"] = trade_return_distribution(p1.trades).to_dict()

    plot_equity_curve(p1.equity_curve, title="Phase 1 Baseline (train+val) -- Equity Curve", filename="phase1_equity.png")
    plot_drawdown(p1.equity_curve, title="Phase 1 Baseline (train+val) -- Drawdown", filename="phase1_drawdown.png")
    plot_trade_return_distribution(p1.trades, filename="phase1_return_dist.png")

    control = run_direction_control(bars_full, p1.signals, DEFAULT_PARAMS, DEFAULT_EXECUTION, DEFAULT_PORTFOLIO, seed=7)
    perm1 = permutation_test_two_groups(p1.trades["r_multiple"].values, control["r_multiple"].values, n_perm=20_000, random_state=1)
    results["phase1"]["direction_control_p_value"] = perm1.p_value
    results["phase1"]["direction_control_real_mean_r"] = float(p1.trades["r_multiple"].mean())
    results["phase1"]["direction_control_control_mean_r"] = float(control["r_multiple"].mean())

    boot = block_bootstrap_ci(p1.trades, value_col="net_pnl", block_col="day", n_boot=10_000)
    results["phase1"]["block_bootstrap_net_pnl"] = {"point": boot.point_estimate, "lo": boot.ci_low, "hi": boot.ci_high}
    logger.info("Phase 1 done: expectancy_r=%.4f, direction_control_p=%.4f",
                results["phase1"]["expectancy_r"], perm1.p_value)

    # ---------------- Phase 2: conditional analysis (train+val) ----------------
    trades = p1.trades.copy()
    has_rv = trades["relative_volume"].notna()
    if has_rv.sum() > 20:
        trades.loc[has_rv, "rel_volume_bucket"] = pd.qcut(trades.loc[has_rv, "relative_volume"], 4, duplicates="drop").astype(str)
    trades["or_width_bucket"] = pd.qcut(trades["or_width"], 4, duplicates="drop").astype(str)
    trades["breakout_hour"] = pd.to_datetime(trades["entry_time"]).dt.hour
    trades["day_of_week"] = pd.to_datetime(trades["entry_time"]).dt.day_name()

    phase2 = {}
    for col in ["rel_volume_bucket", "or_width_bucket", "breakout_hour", "symbol", "day_of_week"]:
        sub = trades.dropna(subset=[col])
        if sub[col].nunique() < 2:
            continue
        try:
            plot_metric_by_bucket(sub, col, value_col="r_multiple", filename=f"phase2_{col}.png")
        except Exception as e:
            logger.warning("plot failed for %s: %s", col, e)
        means = sub.groupby(col)["r_multiple"].mean().sort_values()
        counts = sub.groupby(col)["r_multiple"].count()
        lo_key, hi_key = means.index[0], means.index[-1]
        lo = sub[sub[col] == lo_key]["r_multiple"].values
        hi = sub[sub[col] == hi_key]["r_multiple"].values
        res = permutation_test_two_groups(hi, lo, n_perm=10_000, random_state=3)
        phase2[col] = {
            "bucket_means": means.to_dict(), "bucket_counts": counts.to_dict(),
            "lowest_bucket": str(lo_key), "highest_bucket": str(hi_key),
            "p_value": res.p_value, "observed_diff": res.observed_stat,
        }
        logger.info("Phase 2 [%s]: lowest=%s (mean=%.3f) vs highest=%s (mean=%.3f) p=%.4f",
                    col, lo_key, lo.mean(), hi_key, hi.mean(), res.p_value)
    results["phase2"] = phase2

    # ---------------- Phase 3: multiple-testing correction ----------------
    names = ["phase1_direction_control"] + [f"phase2_{k}" for k in phase2]
    pvals = [perm1.p_value] + [v["p_value"] for v in phase2.values()]
    reject, adj_p = multiple_testing_correction(pvals, method="fdr_bh", alpha=0.05)
    results["phase3"] = {
        "tests": names, "raw_p": pvals, "fdr_adjusted_p": [float(x) for x in adj_p],
        "significant_at_0.05": [bool(x) for x in reject],
    }
    logger.info("Phase 3 correction table: %s", list(zip(names, pvals, adj_p, reject)))

    # ---------------- Phase 4: out-of-sample (validation, then TEST once) ----------------
    val_only = run_backtest(bars_full, DEFAULT_PARAMS, DEFAULT_EXECUTION, DEFAULT_PORTFOLIO,
                             start=split.val_start, end=split.val_end)
    results["phase4_validation"] = summary_row("phase4_validation_only", val_only.trades, split.val_start, split.val_end)

    test_only = run_backtest(bars_full, DEFAULT_PARAMS, DEFAULT_EXECUTION, DEFAULT_PORTFOLIO,
                              start=split.test_start, end=split.test_end)
    results["phase4_test"] = summary_row("phase4_test_period_RUN_ONCE", test_only.trades, split.test_start, split.test_end)
    logger.info("Phase 4: validation expectancy_r=%s, TEST expectancy_r=%s",
                results["phase4_validation"].get("expectancy_r"), results["phase4_test"].get("expectancy_r"))

    # ---------------- Phase 5a: walk-forward ----------------
    windows = walk_forward_windows(DEFAULT_START_DATE, DEFAULT_END_DATE, train_days=126, test_days=42)
    wf_trades = []
    window_summaries = []
    for w in windows:
        r = run_backtest(bars_full, DEFAULT_PARAMS, DEFAULT_EXECUTION, DEFAULT_PORTFOLIO,
                          start=w.test_start, end=w.test_end)
        if not r.trades.empty:
            wf_trades.append(r.trades)
            s = compute_performance_summary(r.trades, DEFAULT_PORTFOLIO.starting_equity, w.test_start, w.test_end)
            window_summaries.append({"test_start": w.test_start, "test_end": w.test_end,
                                      "n_trades": s.n_trades, "expectancy_r": s.expectancy_r, "net_pnl": s.net_pnl})
    results["phase5_walk_forward_windows"] = window_summaries
    if wf_trades:
        wf_all = pd.concat(wf_trades)
        results["phase5_walk_forward_aggregate"] = summary_row(
            "phase5_walk_forward_aggregate", wf_all, windows[0].test_start, windows[-1].test_end
        )
    logger.info("Phase 5a: %d walk-forward windows", len(windows))

    # ---------------- Phase 5b: robustness sweeps (train+val only) ----------------
    sweeps = []
    for or_minutes in [10, 15, 20, 30]:
        p = StrategyParams(or_minutes=or_minutes)
        r = run_backtest(bars_full, p, DEFAULT_EXECUTION, DEFAULT_PORTFOLIO, start=split.train_start, end=split.val_end)
        if not r.trades.empty:
            s = compute_performance_summary(r.trades, DEFAULT_PORTFOLIO.starting_equity, split.train_start, split.val_end)
            sweeps.append({"param": "or_minutes", "value": or_minutes, "n_trades": s.n_trades,
                            "expectancy_r": s.expectancy_r, "sharpe": s.sharpe, "net_pnl": s.net_pnl})

    for rr in [1.0, 1.5, 2.0, 3.0]:
        p = StrategyParams(reward_risk=rr)
        r = run_backtest(bars_full, p, DEFAULT_EXECUTION, DEFAULT_PORTFOLIO, start=split.train_start, end=split.val_end)
        if not r.trades.empty:
            s = compute_performance_summary(r.trades, DEFAULT_PORTFOLIO.starting_equity, split.train_start, split.val_end)
            sweeps.append({"param": "reward_risk", "value": rr, "n_trades": s.n_trades,
                            "expectancy_r": s.expectancy_r, "sharpe": s.sharpe, "net_pnl": s.net_pnl})

    for max_hold in [30, 45, 60, 90, 120]:
        p = StrategyParams(max_holding_minutes=max_hold)
        r = run_backtest(bars_full, p, DEFAULT_EXECUTION, DEFAULT_PORTFOLIO, start=split.train_start, end=split.val_end)
        if not r.trades.empty:
            s = compute_performance_summary(r.trades, DEFAULT_PORTFOLIO.starting_equity, split.train_start, split.val_end)
            sweeps.append({"param": "max_holding_minutes", "value": max_hold, "n_trades": s.n_trades,
                            "expectancy_r": s.expectancy_r, "sharpe": s.sharpe, "net_pnl": s.net_pnl})

    for extra_bps in [0, 2, 5, 10]:
        execp = ExecutionParams(half_spread_bps=DEFAULT_EXECUTION.half_spread_bps,
                                 slippage_bps=DEFAULT_EXECUTION.slippage_bps + extra_bps,
                                 commission_per_share=DEFAULT_EXECUTION.commission_per_share,
                                 reg_fee_bps_on_sell=DEFAULT_EXECUTION.reg_fee_bps_on_sell)
        r = run_backtest(bars_full, DEFAULT_PARAMS, execp, DEFAULT_PORTFOLIO, start=split.train_start, end=split.val_end)
        if not r.trades.empty:
            s = compute_performance_summary(r.trades, DEFAULT_PORTFOLIO.starting_equity, split.train_start, split.val_end)
            sweeps.append({"param": "extra_slippage_bps", "value": extra_bps, "n_trades": s.n_trades,
                            "expectancy_r": s.expectancy_r, "sharpe": s.sharpe, "net_pnl": s.net_pnl})

    for threshold in [None, 1.0, 1.5, 2.0, 3.0]:
        p = StrategyParams(relative_volume_threshold=threshold)
        r = run_backtest(bars_full, p, DEFAULT_EXECUTION, DEFAULT_PORTFOLIO, start=split.train_start, end=split.val_end,
                          require_volume_baseline=True)
        if not r.trades.empty:
            s = compute_performance_summary(r.trades, DEFAULT_PORTFOLIO.starting_equity, split.train_start, split.val_end)
            sweeps.append({"param": "relative_volume_threshold", "value": str(threshold), "n_trades": s.n_trades,
                            "expectancy_r": s.expectancy_r, "sharpe": s.sharpe, "net_pnl": s.net_pnl})

    results["phase5_robustness_sweeps"] = sweeps
    logger.info("Phase 5b: %d sweep variants", len(sweeps))

    out_path = REPORTS_DIR / "full_research_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    logger.info("Wrote %s", out_path)
    p1.trades.to_csv(REPORTS_DIR / "phase1_trades.csv", index=False)
    logger.info("DONE")


if __name__ == "__main__":
    main()
