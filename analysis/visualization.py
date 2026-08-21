"""
Plotting helpers for the research report. All functions save a PNG to
`reports/` and return the path, rather than calling plt.show(), since this
project is driven from scripts/notebooks in a headless research environment.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import REPORTS_DIR

REPORTS_DIR.mkdir(exist_ok=True)


def plot_equity_curve(equity: pd.Series, title: str = "Equity Curve", filename: str = "equity_curve.png") -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(equity.index, equity.values, linewidth=1.2)
    ax.set_title(title)
    ax.set_ylabel("Equity ($)")
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    path = REPORTS_DIR / filename
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_drawdown(equity: pd.Series, title: str = "Drawdown", filename: str = "drawdown.png") -> Path:
    running_max = equity.cummax()
    drawdown = (equity / running_max - 1.0) * 100
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(drawdown.index, drawdown.values, 0, color="firebrick", alpha=0.5)
    ax.set_title(title)
    ax.set_ylabel("Drawdown (%)")
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    path = REPORTS_DIR / filename
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_trade_return_distribution(
    trades: pd.DataFrame, column: str = "r_multiple", title: str = "Trade Return Distribution (R-multiples)",
    filename: str = "return_distribution.png",
) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(trades[column].dropna(), bins=40, color="steelblue", edgecolor="white")
    ax.axvline(0, color="black", linewidth=1)
    ax.axvline(trades[column].mean(), color="firebrick", linestyle="--", label=f"mean={trades[column].mean():.3f}")
    ax.set_title(title)
    ax.set_xlabel(column)
    ax.legend()
    ax.grid(alpha=0.3)
    path = REPORTS_DIR / filename
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_metric_by_bucket(
    trades: pd.DataFrame, bucket_col: str, value_col: str = "net_pnl",
    title: str | None = None, filename: str | None = None,
) -> Path:
    """Bar chart of mean `value_col` (with bootstrap-free +/- 1 SEM error bars)
    grouped by `bucket_col` -- the core chart for Phase 2 conditional analysis
    (e.g. expectancy by relative-volume quintile, by ticker, by day-of-week).
    """
    grouped = trades.groupby(bucket_col)[value_col]
    means = grouped.mean()
    sems = grouped.sem()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(means.index.astype(str), means.values, yerr=sems.values, capsize=4, color="steelblue")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title(title or f"Mean {value_col} by {bucket_col}")
    ax.set_ylabel(f"Mean {value_col}")
    ax.grid(alpha=0.3, axis="y")
    fig.autofmt_xdate()
    path = REPORTS_DIR / (filename or f"{value_col}_by_{bucket_col}.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_bootstrap_distribution(
    boot_values: np.ndarray, observed: float, ci_low: float, ci_high: float,
    title: str = "Bootstrap Distribution", filename: str = "bootstrap_distribution.png",
) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(boot_values, bins=60, color="slategray", alpha=0.8)
    ax.axvline(observed, color="firebrick", linewidth=2, label=f"observed={observed:.4f}")
    ax.axvline(ci_low, color="black", linestyle="--", label="95% CI")
    ax.axvline(ci_high, color="black", linestyle="--")
    ax.axvline(0, color="gray", linewidth=1)
    ax.set_title(title)
    ax.legend()
    path = REPORTS_DIR / filename
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path
