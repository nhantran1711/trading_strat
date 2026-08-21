# Opening-Range Breakout Momentum: A Research Project

This project investigates one specific, falsifiable hypothesis and nothing else:

> When a liquid US stock breaks its first 15-minute opening range with unusually high
> volume, the breakout direction has a statistically significant tendency to continue
> over the following 30-60 minutes.

The goal is to determine whether this is true, not to produce a profitable-looking
backtest. See "Research Philosophy" below.

## Status

**Complete. Answer: no convincing evidence of a tradable edge.** See
`reports/RESEARCH_REPORT.md` for the full write-up (all 12 sections, real numbers).

Real SIP (consolidated-tape) 1-minute data for SPY/QQQ/NVDA/META/AMZN, 2022-08-01 to
2025-08-01, was pulled from Alpaca and run through Phases 1-5. Headline result: the
baseline strategy has negative expectancy net of costs (Sharpe -3.48, profit factor
0.70), the effect replicates (negatively) in validation, in a test period touched
exactly once, and in 13 of 14 walk-forward windows, and the core causal test — does the
true breakout direction beat a coin flip once the execution model's own asymmetry is
controlled for — is not statistically significant (p=0.089).

Along the way, an important data-quality issue was caught and fixed before Phase 1 ran:
Alpaca's free IEX feed captures only ~2% of real trading volume for these symbols
(verified directly), which would have silently invalidated the entire volume-based
analysis — the pipeline was switched to the SIP feed. A separate bug in the direction-
control significance test (stop placed on the wrong side of entry for ~half of
flipped-direction control trades) was caught by the synthetic-data integration tests
before any real-data analysis was affected. Both are documented in
`reports/RESEARCH_REPORT.md` and in code comments where fixed.

Every module — data loading/cleaning, opening range + signal generation, execution
simulation, portfolio/cost accounting, performance metrics, and statistical
significance testing — is covered by unit tests, plus two end-to-end integration tests
that run the full pipeline on synthetic data specifically to catch look-ahead/data-
leakage bugs (see "Testing philosophy" below).

## Setup

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt   # Windows
# source .venv/bin/activate && pip install -r requirements.txt   # macOS/Linux

cp .env.example .env
# Edit .env: fill in ALPACA_API_KEY and ALPACA_SECRET_KEY.
# Free, no funding required: https://alpaca.markets -> Paper Trading -> API Keys.
# Alpaca issues these as a PAIR generated together (Key ID + Secret) -- the secret is
# only shown once at generation time. If you've lost it, regenerate a new pair.
```

## Running

```bash
# 1. Pull and cache ~2-3 years of 1-minute bars for SPY, QQQ, NVDA, META, AMZN.
#    (Actual usable history depends on your Alpaca plan/feed -- logged at fetch time.)
.venv/Scripts/python.exe scripts/fetch_data.py

# 2. Run the Phase 1 baseline backtest (no optimization) on the train+validation
#    period only -- the test period is held out untouched until Phase 4.
.venv/Scripts/python.exe scripts/run_baseline.py

# 3. Run the full test suite (works without any API credentials -- uses synthetic data).
.venv/Scripts/python.exe -m pytest tests/ -v

# 4. Open the research notebook for the full Phase 1-5 walkthrough.
.venv/Scripts/python.exe -m jupyter notebook research/opening_range_research.ipynb

# Or, to reproduce the exact results in reports/RESEARCH_REPORT.md as a single batch run
# (baseline, conditional analysis, significance tests, out-of-sample, walk-forward, robustness):
.venv/Scripts/python.exe scripts/run_full_research.py
```

## Project layout

```text
config.py                    # every research/execution/portfolio parameter, in one place
data/
  loader.py                  # Alpaca fetch + parquet cache
  preprocessing.py           # session alignment, causal VWAP, walk-forward-safe volume baseline
strategies/
  opening_range.py           # OR computation + breakout signal generation (pure, no look-ahead)
backtest/
  execution.py                # realistic fill simulation (spread, slippage, commission, stops)
  portfolio.py                # risk-based position sizing, cost application, equity tracking
  engine.py                   # orchestrates data -> signal -> execution -> portfolio per day
analysis/
  performance.py              # win rate, expectancy, Sharpe/Sortino, drawdown, exposure, turnover...
  statistics.py                # bootstrap CIs, permutation tests, multiple-testing correction
  visualization.py             # equity curve, drawdown, return distribution, conditional-analysis plots
research/
  splits.py                    # chronological train/val/test split + walk-forward windows
  opening_range_research.ipynb # Phase 1-5 notebook
scripts/
  fetch_data.py                 # data acquisition CLI
  run_baseline.py                # Phase 1 baseline CLI
tests/                           # unit + synthetic-data integration tests (no API needed)
```

## Data methodology & known limitations (disclosed up front)

- **Universe**: fixed to SPY, QQQ, NVDA, META, AMZN per the research brief. These are
  all currently-liquid, currently-listed instruments that existed across the full
  lookback window — this is a **declared survivorship-bias limitation**: no delisted,
  acquired, or failed names are considered, and results should not be assumed to
  generalize to a broader or historically-accurate universe.
- **Adjustment**: bars are fetched split/dividend-adjusted (`adjustment="all"`) directly
  from Alpaca; `data/preprocessing.detect_price_discontinuities` flags any residual
  >20% overnight gaps for manual audit (a legitimate large gap event, or a sign an
  adjustment didn't apply correctly).
- **Session handling**: only regular-session bars (09:30-16:00 ET) are used; premarket/
  after-hours bars are excluded. Missing minutes within a session are filled as a flat
  bar at the previous close with zero volume and flagged `is_synthetic=True` (a
  disclosed "no trade printed" assumption, not an invented price move); days with more
  than 5% of the session missing are dropped entirely and logged.
- **Market calendar**: holidays and half-days are sourced from the official NYSE
  calendar (`pandas_market_calendars`), not inferred from gaps in the data — so a real
  data-vendor outage is distinguishable from an expected market holiday.
- **VWAP / volume baseline**: both are computed causally — VWAP is a running
  cumulative that resets each session (a value at minute *t* only ever reflects bars up
  to and including *t*); the relative-volume baseline for day *N* is the trailing mean
  of the prior `volume_baseline_lookback_days` days' opening-range volume, strictly
  excluding day *N* itself and every future day. See
  `tests/test_preprocessing.py::test_or_volume_baseline_excludes_current_and_future_days`
  for a test that directly proves this.

## Execution & cost assumptions (disclosed, and swept in Phase 5)

- Entry fills at the **next bar's open** after the signal bar's close (the signal is
  only knowable once that bar closes — filling at that same close is a zero-latency
  fantasy).
- Both entry and exit pay `half_spread_bps + slippage_bps` **against** the trader
  (never in their favor) — see `config.ExecutionParams`.
- Commission modeled per-share plus SEC/FINRA regulatory fees on the sell leg, even
  though Alpaca-style retail equities are nominally "commission-free" — the reg fees
  are real and small but non-zero.
- If a single bar's range could have hit both the stop and the target, the
  **conservative** assumption is that the stop was hit first (worst case) — never the
  best case.
- Position sizing is risk-based (a fixed fraction of current equity risked per trade,
  sized off the entry-to-stop distance), capped by both a max-notional-of-equity limit
  and a max-participation-of-opening-range-volume limit, so the model never implies
  trading a size the market couldn't plausibly absorb.

All of Phase 1's numbers are reported **gross and net of these costs**; the net number
is what matters for the final conclusion.

## Testing philosophy

Unit tests (`tests/test_opening_range.py`, `test_execution.py`, `test_preprocessing.py`,
`test_portfolio.py`, `test_statistics.py`) check individual modules in isolation,
including several tests written specifically to prove the no-look-ahead property (e.g.
`test_signal_scan_never_uses_future_bars`, `test_vwap_is_causal_not_using_future_bars`,
`test_or_volume_baseline_excludes_current_and_future_days`).

`tests/test_engine_integration.py` runs the **entire pipeline end-to-end on synthetic
data** as a bug-detection guard, independent of whether real data ever shows an edge:

- On a pure random walk with **no injected drift**, real trades must be statistically
  indistinguishable from a direction-randomized control group (same timing/price/OR
  levels, direction re-assigned by coin flip). If they're *not* indistinguishable, that
  points at a bug, since there is no directional information in the underlying series
  by construction.
- On synthetic data with a **deliberately injected** continuation drift, the pipeline
  must detect a real, statistically significant edge — confirming a null result on real
  data later would reflect the data, not a broken/insensitive pipeline.

This first test actually caught a real bug during development: an early version of the
direction-randomized control reused the opening-range midpoint as a literal stop PRICE
even for the flipped direction, which silently put the stop on the wrong side of entry
for ~half the control trades (a "stop" that was actually a disguised win). Fixed in
`backtest/execution.py` (`stop_distance_override`) and `backtest/engine.py`
(`run_direction_control`) — the control now preserves the true risk *distance* from the
original setup but places it on the correct side of entry for whichever direction is
being tested. This is exactly the kind of self-inflicted, easy-to-miss bug the research
philosophy below asks you to keep hunting for once real data is in play.

## Research philosophy

**Do not try to make the strategy profitable. Try to disprove the hypothesis.**

If the baseline loses money net of realistic costs, that is the finding — document it
and investigate why, don't iterate on parameters until something turns green. If it
appears profitable, the burden of proof is on ruling out: look-ahead bias, overfitting,
data leakage, survivorship bias, unrealistic execution, selection bias, multiple-testing
problems, and regime dependence, in that order, before calling it a real edge. Every
Phase 5 robustness sweep should be read the same way: a strategy that only works at one
suspiciously specific parameter value is evidence against it, not for it.

The final report (produced once real data is available — see Status above) will answer,
explicitly, without softening a "no": **is there convincing evidence that high-volume
opening-range breakouts provide a tradable intraday edge after realistic transaction
costs?**
