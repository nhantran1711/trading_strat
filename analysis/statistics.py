"""
Statistical significance testing for the trade ledger: is the observed edge
distinguishable from noise, and how much of that inference survives once we
account for having looked at many candidate variables (Phase 2/5)?

Nothing here p-hacks: the permutation/bootstrap functions are generic and
operate on whatever return series they're given -- the discipline of NOT
re-running them until a favorable answer appears, and of correcting for
multiple comparisons across Phase 2's conditional-analysis variables, lives
in how the research notebook calls these functions, not in the functions
themselves.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests


@dataclass
class BootstrapResult:
    point_estimate: float
    ci_low: float
    ci_high: float
    std_error: float
    n_boot: int


def bootstrap_ci(
    values: np.ndarray | pd.Series,
    stat_fn=np.mean,
    n_boot: int = 10_000,
    ci_level: float = 0.95,
    random_state: int = 42,
) -> BootstrapResult:
    """Simple i.i.d. bootstrap of `stat_fn` over `values` (default: the mean).
    Appropriate when trades can reasonably be treated as exchangeable; use
    `block_bootstrap_ci` instead when trades cluster by day/regime and that
    correlation should be preserved under resampling.
    """
    values = np.asarray(values)
    values = values[~np.isnan(values)]
    rng = np.random.default_rng(random_state)
    n = len(values)
    boot_stats = np.empty(n_boot)
    for i in range(n_boot):
        sample = values[rng.integers(0, n, size=n)]
        boot_stats[i] = stat_fn(sample)

    alpha = 1 - ci_level
    lo, hi = np.quantile(boot_stats, [alpha / 2, 1 - alpha / 2])
    return BootstrapResult(
        point_estimate=float(stat_fn(values)),
        ci_low=float(lo),
        ci_high=float(hi),
        std_error=float(boot_stats.std(ddof=1)),
        n_boot=n_boot,
    )


def block_bootstrap_ci(
    trades: pd.DataFrame,
    value_col: str = "net_pnl",
    block_col: str = "day",
    stat_fn=np.mean,
    n_boot: int = 10_000,
    ci_level: float = 0.95,
    random_state: int = 42,
) -> BootstrapResult:
    """Bootstrap by resampling whole DAYS (with all that day's trades attached),
    not individual trades -- preserves within-day correlation (e.g. multiple
    symbols breaking out on the same broad-market-driven day) instead of
    treating same-day trades as independent draws, which would understate
    the true standard error.
    """
    days = trades[block_col].unique()
    rng = np.random.default_rng(random_state)
    n_days = len(days)
    boot_stats = np.empty(n_boot)
    grouped = {d: g[value_col].values for d, g in trades.groupby(block_col)}

    for i in range(n_boot):
        sampled_days = rng.choice(days, size=n_days, replace=True)
        sample_values = np.concatenate([grouped[d] for d in sampled_days])
        boot_stats[i] = stat_fn(sample_values)

    alpha = 1 - ci_level
    lo, hi = np.quantile(boot_stats, [alpha / 2, 1 - alpha / 2])
    return BootstrapResult(
        point_estimate=float(stat_fn(trades[value_col].values)),
        ci_low=float(lo),
        ci_high=float(hi),
        std_error=float(boot_stats.std(ddof=1)),
        n_boot=n_boot,
    )


@dataclass
class PermutationResult:
    observed_stat: float
    p_value: float
    n_perm: int
    null_mean: float
    null_std: float


def sign_flip_test(returns: np.ndarray | pd.Series, n_perm: int = 10_000, random_state: int = 42) -> PermutationResult:
    """Tests H0: the return series has zero-mean, direction-agnostic edge (i.e.
    breakout direction carries no information -- a long breakout and a short
    breakout are equally likely to be winners) by randomly flipping the sign
    of each trade's return and re-computing the mean, many times. Appropriate
    because "direction predicts continuation" implies a specific SIGN
    (winning trades should be systematically positive once you've already
    conditioned on being long above / short below the range) -- under the
    null that direction is uninformative, each trade's realized return is
    exchangeable with its negation.

    CAVEAT (discovered via tests/test_engine_integration.py): this test
    implicitly assumes the return distribution would be roughly symmetric
    around zero under "no edge". That assumption breaks for THIS strategy's
    execution model, because a 2:1 reward:risk target plus a conservative
    same-bar stop/target tie-break both mechanically bias expectancy away
    from zero even when direction carries no information whatsoever (pure
    random-walk data reproducibly fails this test at p<0.001 in the test
    suite). Use this function for a quick/generic check, but treat
    `backtest.engine.run_direction_control` + `permutation_test_two_groups`
    as the primary significance test for THIS strategy, since it compares
    real trades against a control group that experienced the identical
    execution asymmetries and cost structure, isolating direction's actual
    information content instead of conflating it with execution-mechanics bias.
    """
    returns = np.asarray(returns)
    returns = returns[~np.isnan(returns)]
    observed = returns.mean()

    rng = np.random.default_rng(random_state)
    n = len(returns)
    null_stats = np.empty(n_perm)
    for i in range(n_perm):
        signs = rng.choice([-1, 1], size=n)
        null_stats[i] = (returns * signs).mean()

    p_value = float(np.mean(np.abs(null_stats) >= abs(observed)))
    return PermutationResult(
        observed_stat=float(observed), p_value=p_value, n_perm=n_perm,
        null_mean=float(null_stats.mean()), null_std=float(null_stats.std()),
    )


def permutation_test_two_groups(
    group_a: np.ndarray | pd.Series, group_b: np.ndarray | pd.Series,
    n_perm: int = 10_000, random_state: int = 42,
) -> PermutationResult:
    """Standard two-sample permutation test for a difference in means (e.g.
    high-relative-volume trades vs low-relative-volume trades in Phase 2):
    pools both groups, randomly reassigns labels of the same group sizes,
    and asks how often a difference this large or larger arises by chance.
    """
    a = np.asarray(group_a)
    a = a[~np.isnan(a)]
    b = np.asarray(group_b)
    b = b[~np.isnan(b)]
    observed_diff = a.mean() - b.mean()

    pooled = np.concatenate([a, b])
    n_a = len(a)
    rng = np.random.default_rng(random_state)
    null_diffs = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(pooled)
        null_diffs[i] = perm[:n_a].mean() - perm[n_a:].mean()

    p_value = float(np.mean(np.abs(null_diffs) >= abs(observed_diff)))
    return PermutationResult(
        observed_stat=float(observed_diff), p_value=p_value, n_perm=n_perm,
        null_mean=float(null_diffs.mean()), null_std=float(null_diffs.std()),
    )


def multiple_testing_correction(p_values: list[float], method: str = "fdr_bh", alpha: float = 0.05):
    """Adjusts a family of p-values (e.g. every Phase 2 conditional-variable
    test, or every Phase 5 robustness-sweep parameter) for multiple
    comparisons. Default is Benjamini-Hochberg FDR control; pass
    method="bonferroni" or "holm" for stricter family-wise control.
    Returns (reject: bool array, adjusted_p_values: array).
    """
    reject, adj_p, _, _ = multipletests(p_values, alpha=alpha, method=method)
    return reject, adj_p
