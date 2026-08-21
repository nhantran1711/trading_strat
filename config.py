"""
Central configuration for the opening-range breakout research project.

Every research parameter that could plausibly be "optimized" lives here (or is passed
as an explicit override into the relevant function) so that Phase 5 robustness testing
can sweep them systematically instead of them being buried as magic numbers in code.

Nothing in this file should be treated as a validated final choice -- these are the
BASELINE values specified in the research brief, not the result of any optimization.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from zoneinfo import ZoneInfo

# ---------------------------------------------------------------------------
# Universe & paths
# ---------------------------------------------------------------------------

UNIVERSE: list[str] = ["SPY", "QQQ", "NVDA", "META", "AMZN"]

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_CACHE_DIR = PROJECT_ROOT / "data_cache"
REPORTS_DIR = PROJECT_ROOT / "reports"

EASTERN = ZoneInfo("America/New_York")

# ---------------------------------------------------------------------------
# Session definition
# ---------------------------------------------------------------------------

MARKET_OPEN = "09:30"
MARKET_CLOSE = "16:00"

# ---------------------------------------------------------------------------
# Strategy parameters (BASELINE -- see research brief; treat as research
# parameters to be swept in Phase 5, not fixed truths).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StrategyParams:
    # Opening range window, minutes after the 09:30 open.
    or_minutes: int = 15

    # No new entries after this time of day (ET, "HH:MM").
    cutoff_time: str = "11:00"

    # Reward:risk multiple for the fixed take-profit target.
    reward_risk: float = 2.0

    # Maximum holding period in minutes, regardless of stop/target.
    max_holding_minutes: int = 60

    # Relative-volume lookback: number of PRIOR trading days used to build the
    # historical baseline for "unusually high volume" at the OR window. Using
    # a rolling trailing window (not the full sample) keeps the baseline
    # walk-forward safe -- a day's baseline never uses that day or later days.
    volume_baseline_lookback_days: int = 20

    # Relative-volume threshold candidates for Phase 2/5 sweeps. Baseline model
    # does NOT filter on this -- Phase 1 takes every breakout regardless of
    # volume, to first establish whether breakout direction has ANY edge
    # before conditioning on volume at all.
    relative_volume_threshold: float | None = None  # None = no filter (Phase 1 baseline)

    # Minimum number of prior trading days required before a symbol is eligible
    # for a signal (so the volume baseline is not computed from too little data).
    min_baseline_days: int = 20


DEFAULT_PARAMS = StrategyParams()


# ---------------------------------------------------------------------------
# Execution / cost model (BASELINE assumptions -- documented, not fitted).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExecutionParams:
    # Entry fill: the strategy signals on a 1-min bar's CLOSE. We do not allow
    # filling at that same close (unrealistic -- by the time the bar closes and
    # the signal is computed, that price is already gone). Fill is simulated at
    # the NEXT bar's open, adjusted for slippage/spread below.
    fill_on_next_bar_open: bool = True

    # Half bid-ask spread paid on entry and exit, in basis points of price.
    # SPY/QQQ/mega-cap names typically quote 1-2 cent spreads on multi-hundred
    # dollar stocks -> a few bps; this is a deliberately conservative baseline,
    # swept in Phase 5.
    half_spread_bps: float = 2.0

    # Additional slippage (market impact / latency) beyond the spread, in bps,
    # applied against the trader on both entry and exit.
    slippage_bps: float = 2.0

    # Commission per share (Alpaca-style U.S. equities are commission-free for
    # retail, but SEC Section 31 + FINRA TAF fees still apply on sells). Model
    # as a flat per-share fee to stay conservative and broker-agnostic.
    commission_per_share: float = 0.005

    # Regulatory fees (SEC + FINRA TAF), applied on the SELL side notional,
    # in basis points. Small but included for completeness/realism.
    reg_fee_bps_on_sell: float = 0.08

    # Same-bar stop/target ambiguity: if a single bar's range contains both the
    # stop and the target, assume the WORSE outcome (stop hit first). This is
    # the conservative convention used throughout -- never assume the best case.
    conservative_same_bar_fill: bool = True


DEFAULT_EXECUTION = ExecutionParams()


# ---------------------------------------------------------------------------
# Position sizing / portfolio
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PortfolioParams:
    starting_equity: float = 100_000.0

    # Risk per trade as a fraction of current equity, sized off the
    # entry-to-stop distance (i.e. true dollar risk, not notional).
    risk_per_trade: float = 0.005

    # Cap position notional as a fraction of equity, in case a very tight stop
    # would otherwise imply an unrealistically large share count.
    max_position_notional_frac: float = 0.25

    # Cap position size as a fraction of the opening-range volume, to avoid
    # implying the strategy could trade a size the market could not actually
    # absorb without moving price further than modeled.
    max_participation_of_or_volume: float = 0.01


DEFAULT_PORTFOLIO = PortfolioParams()


# ---------------------------------------------------------------------------
# Data date ranges (used by scripts/fetch_data.py). ~2-3 years, but the
# Alpaca free/IEX feed's actual usable history for 1-min bars is discovered
# at fetch time and logged -- do not assume this full window is available.
# ---------------------------------------------------------------------------

DEFAULT_START_DATE = "2022-08-01"
DEFAULT_END_DATE = "2025-08-01"
