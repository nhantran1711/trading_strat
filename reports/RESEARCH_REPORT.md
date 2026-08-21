# Research Report: Opening-Range Breakout Momentum

**Status: complete. Answer: NO — the hypothesis is not supported.**

Real 1-minute SIP (consolidated-tape) data for SPY, QQQ, NVDA, META, AMZN, 2022-08-01 to
2025-08-01, was pulled from Alpaca Markets and run through all five research phases:
baseline, conditional analysis, formal significance testing, out-of-sample validation on an
untouched test period, and walk-forward + robustness sweeps. The baseline strategy loses money,
net of realistic transaction costs, consistently across the training period, the validation
period, the held-out test period, and 13 of 14 independent walk-forward windows. The specific
causal claim under test — that the *true* breakout direction predicts continuation, beyond what
the execution model's own structural asymmetry would produce from a coin flip — is not
statistically significant. See Section 12 for the full reasoning.

---

## 1. Hypothesis

> When a liquid US stock breaks its first 15-minute opening range with unusually high volume,
> the breakout direction has a statistically significant tendency to continue over the following
> 30-60 minutes.

This is the only hypothesis under test. No alternative strategies were considered.

## 2. Data methodology

- **Universe**: SPY, QQQ, NVDA, META, AMZN — fixed by the research brief, not selected based on
  any backtest result.
- **Source**: Alpaca Markets 1-minute OHLCV bars, split/dividend-adjusted (`adjustment="all"`),
  **SIP (consolidated) feed** — see the important data-quality note below.
- **Lookback**: 2022-08-01 to 2025-08-01 (~3 years), full range successfully retrieved.
  ~2.6-2.9M raw 1-minute bars per symbol were fetched.
- **Session**: regular trading hours only (09:30-16:00 ET); premarket/after-hours excluded.
- **Cleaning**: NYSE trading-calendar-aware (`pandas_market_calendars`). After cleaning, only
  3-5 low-coverage days were dropped per symbol out of ~700 trading days in the full range — and
  every single one is a genuine NYSE half-day (day after Thanksgiving, July 3rd) or a data-vendor
  gap on Thanksgiving itself, not an unexplained gap.
- **Known limitations, disclosed up front**:
  - *Survivorship bias*: the universe consists of instruments that existed and were liquid
    across the entire lookback window. No delisted/failed names are considered. Findings do not
    generalize beyond this fixed, currently-liquid universe.
  - *Small universe*: five tickers is not enough to make cross-sectional claims about "stocks in
    general" — only about these five, over this period.

### Data-quality finding worth flagging on its own

The project initially fetched data on Alpaca's free **IEX** feed. Direct comparison showed IEX
captures only a small fraction of real volume — e.g. META on 2024-06-03 showed **193,856 shares**
of IEX volume vs. **10,843,911 shares** on the SIP (consolidated) feed, roughly 2% coverage. On
the IEX feed, 72-91% of trading days were being dropped by the >5%-missing-minutes cleaning rule
for QQQ/META specifically, and the volume figures central to this whole hypothesis would have
been unreliable even on the days that survived. This was caught before any result was computed,
and the pipeline was switched to the SIP feed (confirmed accessible on this account) before
Phase 1 ran. Anyone reproducing this analysis on a free/IEX-only Alpaca plan should expect
materially different — and less trustworthy — results, particularly for anything volume-related.

Full detail: `README.md` -> "Data methodology & known limitations".

## 3. Strategy definition (baseline, Phase 1 — no optimization)

- Opening range: high/low of the first 15 minutes (09:30-09:45 ET).
- Long signal: first 1-minute close above the opening-range high. Short signal: first 1-minute
  close below the opening-range low. At most one signal per symbol per day, no new signals after
  11:00 ET.
- Stop-loss: opening-range midpoint.
- Take-profit: 2:1 reward:risk from the actual fill price.
- Max holding period: 60 minutes.
- Volume filter: **none** in the Phase 1 baseline — Phase 1 tests direction-only continuation
  before conditioning on volume at all, per the research brief ("do not assume that a particular
  volume threshold is optimal").

Full detail: `config.StrategyParams` (`config.py`), `strategies/opening_range.py`.

## 4. Execution assumptions

- Fills simulated at the next 1-minute bar's open after the signal bar's close (not the signal
  close itself).
- Cost haircut on both entry and exit: 2bps half-spread + 2bps slippage (4bps round-trip-per-leg),
  always against the trader.
- Commission $0.005/share plus SEC/FINRA regulatory fees (0.08bps) on the sell leg.
- Conservative same-bar stop/target tie-break (assume stop hit first when a single bar's range
  could have triggered either).
- Risk-based position sizing (0.5% of equity risked per trade off the entry-to-stop distance),
  capped at 25% of equity notional and 1% participation of opening-range volume.

Full detail: `config.ExecutionParams` / `config.PortfolioParams`, `backtest/execution.py`,
`backtest/portfolio.py`.

## 5. Backtest results (Phase 1 — Baseline, train+validation: 2022-08-01 to 2024-12-20)

| Metric | Value |
|---|---|
| Trades | 2,817 |
| Win rate | 39.5% |
| Avg win | $99.07 |
| Avg loss | -$91.66 |
| Expectancy | -$16.37/trade (-0.208R) |
| Profit factor | 0.70 |
| Gross P&L | -$40,606 |
| Total costs | $5,505 |
| **Net P&L** | **-$46,111** (on $100,000 starting equity) |
| Total return | -47.1% |
| CAGR | -23.3% |
| Sharpe | -3.48 |
| Sortino | -6.09 |
| Max drawdown | -47.1% |
| Avg holding time | 39.8 min (of a 60 min cap — most trades hit stop/target before the clock) |
| Exposure | 47.7% of session-minutes with a position open |
| Turnover | 1,401x average equity |

Plots: `reports/phase1_equity.png`, `phase1_drawdown.png`, `phase1_return_dist.png`. Full trade
ledger: `reports/phase1_trades.csv`.

The loss is **not primarily a transaction-cost story**: gross P&L is already -$40,606 before the
$5,505 of modeled costs. Costs make it worse, but the strategy loses on the underlying price
action itself.

Return distribution (R-multiples): mean -0.208, median -0.494, right-skewed (skew +0.71) as
expected for a 2:1 target/1R-stop structure — most trades are small losses, a minority are ~+2R
wins, but there aren't enough winners to overcome the 2:1-vs-~1:2-win-rate arithmetic.

## 6. Statistical analysis (Phase 3)

**Primary significance test — direction-randomized control** (isolates whether the *true*
breakout direction adds information beyond the execution model's own structural asymmetry; see
`README.md` -> "Testing philosophy" for why this is the correct null, not a naive
mean-equals-zero test):

- Real trades mean R-multiple: **-0.2083**
- Control (direction randomly re-assigned, same timing/price/risk-distance) mean R-multiple:
  **-0.2565**
- Observed difference: +0.048R (real trades slightly *less bad* than the control)
- Permutation p-value: **0.089** (n=20,000 permutations) — **not significant** at p<0.05
- FDR-adjusted (across the full family of Phase 1+2 tests below): **p=0.106** — still not
  significant

Real trades outperform the randomized-direction control by a small margin, but that margin is
not statistically distinguishable from noise. This is the single most important number in this
report: **there is no statistically significant evidence that knowing the true breakout
direction is better than guessing**, once the execution model's own asymmetric R:R structure is
held constant between the two groups.

**Block-bootstrap (resampled by day) 95% CI on mean net P&L per trade**: -$16.37
[-$23.01, -$9.66] — reliably negative; zero is not in the interval.

**Phase 2 conditional-variable tests** (each compares the highest- vs. lowest-performing bucket
of that variable; all n≥500 except breakout_hour):

| Variable | Worst bucket (mean R) | Best bucket (mean R) | Raw p | FDR-adjusted p | Significant? |
|---|---|---|---|---|---|
| Relative volume (quartile) | 0.90-1.12x: -0.264 | >1.12x: -0.139 | 0.023 | 0.034 | Yes |
| OR width (quartile) | narrowest: -0.328 | widest: -0.091 | 0.0001 | 0.0003 | Yes |
| Breakout hour | 09:xx: -0.224 | 11:xx (n=2): +0.083 | 0.735 | 0.735 | No |
| Symbol | SPY: -0.386 | NVDA: -0.099 | <0.0001 | <0.0001 | Yes |
| Day of week | Monday: -0.306 | Friday: -0.130 | 0.0048 | 0.0096 | Yes |

**Read this table carefully**: four of these five splits are statistically significant even
after FDR correction — but in **every single case, the "best" bucket is still net-negative**.
Relative volume, OR width, symbol, and day-of-week all have a real, non-random relationship with
*how bad* the loss is, but none of them flips the strategy into profitability on its own. This is
a meaningfully different (and much weaker) finding than "an edge exists in the high-relative-
volume subset" — it means the structural loss shrinks under certain conditions without
disappearing. See Section 10 for the one sweep result that did cross into positive territory, and
why it doesn't change the conclusion.

## 7. Drawdown / risk analysis

Max drawdown -47.1% over the train+validation period, essentially equal to the total return
figure because the equity curve declines close to monotonically rather than recovering (Sharpe
-3.48, Sortino -6.09 reflect this — losses are not just larger than gains on average, they are
persistent). See `reports/phase1_drawdown.png`. Exposure of 47.7% and turnover of ~1,401x
average equity reflect the strategy's high trade frequency (2,817 trades across 5 symbols over
~440 trading days) relative to its short average holding period (~40 minutes) — this is a
high-turnover, cost-sensitive strategy structurally, which the Section 10 slippage sweep
confirms.

## 8. Out-of-sample results (Phase 4)

Same fixed baseline parameters, no re-tuning, evaluated on periods not used in Phase 1-3:

| Period | Trades | Expectancy_R | Profit factor | Sharpe | Net P&L |
|---|---|---|---|---|---|
| Train+Val (Phase 1, for reference) | 2,817 | -0.208 | 0.70 | -3.48 | -$46,111 |
| Validation only (2024-05-17 to 2024-12-20) | 704 | -0.143 | 0.79 | -2.23 | -$9,150 |
| **Test (2024-12-23 to 2025-08-01, touched once)** | **685** | **-0.112** | **0.89** | **-0.81** | **-$5,301** |

The result **replicates directionally out of sample**: negative expectancy in training, negative
in validation, and negative in the test period that was held out and touched exactly once, after
Phases 1-3 were finalized. The test period is the *least* bad of the three (profit factor closest
to 1.0, Sharpe closest to 0), which is consistent with normal period-to-period variation around a
persistently negative-to-flat true effect — not with the strategy "working" out of sample.

## 9. Walk-forward results (Phase 5a)

14 rolling windows (126 trading days train / 42 trading days test, non-overlapping test windows,
spanning the full 2022-08-01 to 2025-08-01 range):

- **13 of 14 windows: negative expectancy_R** (range: -0.042 to -0.365)
- **1 of 14 windows positive**: 2025-02-04 to 2025-04-03, +0.186R, +$4,602 on 201 trades
- Aggregated across all windows (2,736 trades): expectancy -0.197R, profit factor 0.73,
  Sharpe -2.30, net P&L -$50,170, max drawdown -51.3%

One positive window out of fourteen is exactly what unconditional noise around a negative-to-flat
true effect would produce — it is not evidence of a regime where the strategy "turns on." The
walk-forward aggregate (-0.197R) closely matches the original train+val baseline (-0.208R),
which is itself evidence that the negative result is stable across time and not an artifact of
the particular chronological train/val/test split chosen in Phase 1.

## 10. Robustness analysis (Phase 5b)

All sweeps run on train+validation only (test period untouched). Full table in
`reports/full_research_results.json` -> `phase5_robustness_sweeps`.

- **Opening-range duration** (10/15/20/30 min): expectancy_R ranges -0.231 to -0.171. Longer
  ranges are mildly less bad but never positive.
- **Reward:risk** (1.0/1.5/2.0/3.0): expectancy_R tightly clustered, -0.201 to -0.211 —
  essentially insensitive to this parameter. The negative result is not an artifact of the
  specific 2:1 choice.
- **Max holding period** (30/45/60/90/120 min): expectancy_R ranges -0.212 to -0.188. Longer
  holds are mildly less bad but never positive.
- **Added slippage** (+0/2/5/10bps beyond baseline): expectancy_R degrades sharply and
  monotonically, -0.208 → -0.288 → -0.391 → -0.535. The strategy is highly sensitive to
  execution cost assumptions in the bad direction — it does not become more attractive under any
  more-favorable-than-baseline cost assumption tested (0 extra bps *is* the baseline, already
  negative).
- **Relative-volume threshold** (None/1.0/1.5/2.0/3.0x, `require_volume_baseline=True`):
  expectancy_R improves monotonically and substantially as the threshold rises — None: -0.204
  (n=2,719), 1.0x: -0.173 (n=1,005), 1.5x: -0.147 (n=244), 2.0x: -0.090 (n=84), **3.0x: +0.129
  (n=31, Sharpe 0.28, net P&L +$616)**.

**On that last row specifically** — this is the one place in the entire study where a variant
crosses into positive territory, and it deserves the scrutiny the research philosophy demands
rather than being reported as a discovery:

1. **n=31 trades** is far too small to support a statistical claim; a strategy with this few
   observations does not have a meaningfully estimated expectancy.
2. It emerged from a **5-value sweep**, not a single pre-registered parameter choice — exactly
   the multiple-testing scenario Phase 3 exists to guard against, and it has not been through
   that correction.
3. It was computed on **train+validation data already used** for every other test in this report,
   with **no out-of-sample confirmation** — the test period was already spent once in Phase 4 and
   is not re-used here to avoid exactly the kind of post-hoc mining this finding would otherwise
   represent.
4. It is precisely the pattern the brief warns about: *"a strategy that only works at one
   extremely specific parameter value should be treated as suspicious."* A monotonically
   improving trend that only tips positive at the most extreme, sparsest-data point of five is
   the textbook shape of a threshold effect combined with a shrinking sample, not a validated
   discovery.

The honest reading: relative trading volume has a real, statistically detectable relationship
with how well (or how badly) an opening-range breakout performs — consistent with *part* of the
hypothesis — but nothing in this study demonstrates a validated, out-of-sample, adequately-
sampled positive edge at any volume threshold.

## 11. Failure modes considered

- **Look-ahead bias**: actively checked throughout development, not just at the end.
  `strategies/opening_range.py` and `data/preprocessing.py` include tests that directly prove
  causality (truncating a day after the signal bar reproduces an identical signal; VWAP and the
  volume baseline are provably unaffected by later bars/days). No evidence found.
- **Overfitting**: Phase 5b shows the negative result is stable across parameter choices rather
  than concentrated at the baseline's specific values — the opposite of what overfitting to the
  baseline parameters would look like.
- **Data leakage**: the test period (2024-12-23 to 2025-08-01) was not loaded or inspected until
  Phase 4, after Phase 1-3 were finalized on train+validation only, and was evaluated exactly
  once. The Phase 5b relative-volume-threshold=3.0 result was deliberately NOT re-tested against
  the test period, to avoid spending it a second time chasing that result (see Section 10).
- **Survivorship bias**: present and disclosed (Section 2) — a real, acknowledged limitation of
  the fixed five-ticker universe, not something the results can be corrected for after the fact.
- **Unrealistic execution**: the Section 10 slippage sweep shows results get *worse*, not better,
  under more conservative cost assumptions — if anything, the baseline assumptions (2bps
  half-spread + 2bps slippage) are already lenient relative to how sensitive the strategy is to
  this parameter.
- **Selection bias**: the opening-range/breakout definition, universe, and baseline parameters
  were fixed by the research brief before any backtest ran, not chosen after seeing results.
- **Multiple-testing problems**: directly addressed in Phase 3 (FDR correction across the Phase
  1+2 test family) and in the explicit treatment of the relative-volume-threshold=3.0 result in
  Section 10, which is flagged rather than reported as a finding.
- **Regime dependence**: checked via the Phase 5a walk-forward analysis — the negative result is
  broadly persistent (13/14 windows) rather than concentrated in one period, and the one positive
  window is consistent with ordinary noise, not a distinct favorable regime.

## 12. Final conclusion

**Question:** Is there convincing evidence that high-volume opening-range breakouts provide a
tradable intraday edge after realistic transaction costs?

**Answer: No.**

The baseline strategy — first 15-minute opening-range breakout, opening-range-midpoint stop, 2:1
reward:risk target, 60-minute max hold, no entries after 11:00 ET — loses money net of realistic
transaction costs on SPY, QQQ, NVDA, META, and AMZN over 2022-08-01 to 2025-08-01 (Sharpe -3.48,
profit factor 0.70, expectancy -0.21R/trade). This holds up in the validation period, in a test
period touched exactly once after all methodology was finalized, and in 13 of 14 independent
walk-forward windows spanning the full three years. The specific causal claim under test —
that knowing the true breakout direction beats a coin flip, once the execution model's own
structural asymmetry is controlled for — is not statistically significant (p=0.089, FDR-adjusted
p=0.106).

There is a real, statistically significant (FDR-corrected) relationship between relative trading
volume and how large the loss is — consistent with *part* of the hypothesis — but even the
best-performing volume quartile in the primary analysis remains net-negative, and the one
parameter combination that crossed into positive territory (relative volume ≥3x baseline) did so
on a 31-trade sample from an unconfirmed, non-pre-registered sweep, which does not meet a
reasonable bar for "convincing evidence" under the multiple-testing and small-sample scrutiny
this report applies to itself.

**No changes were made to the methodology, universe, or baseline parameters after seeing
unfavorable results**, in keeping with the research brief's explicit instruction not to iterate
toward a profitable-looking answer.
