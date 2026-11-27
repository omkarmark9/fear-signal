# The trading model (model 2)

Model 1, the fear model, measures how afraid the AI stocks are. Model 2
reads model 1's outputs at every close and decides what to trade at the
next open: which names, how much, for how long, and whether shares or
options. It trades only the 134 names in the universe, long only, with
options limited to level-2 approval: long calls, long puts, cash-secured
puts and covered calls.

A trade is labelled **high conviction** only if its rule:
1. was declared before any backtest;
2. passed every gate on 2024–2025;
3. agreed in the 2026 holdout with t ≥ 1, in a single read;

and the ticket itself meets its family's ticket conditions. Everything else
is shown as information (Tier 0).

<!-- RESULTS -->

## What model 2 reads from model 1

| model-1 output | how model 2 uses it |
|---|---|
| systemic-day label (≥ 10% of names dipped ≥ 2 own-σ) | **which kind of shock** it is: market-wide selling (liquidity) or stock-specific (information) |
| x, the drawdown-to-low in own σ | whether a name took part in the shock |
| σ, the robust own scale | the yardstick for the beta residual u, its segment-relative version v and the news gap γ, and the base of risk sizing |
| M_pct, market fear | the stress dose (SR3, Tier 3) and the state of the RF overlay |
| filtered regime probabilities, A, s | regime-adjusted forward volatility, for sizing and for judging whether option IV is rich |
| q_next | position caps (5% per name when q_next ≥ 0.10) and the tail-risk monitor |

Backtests use **point-in-time** fear readings. At each quarter's first
session the model is refit on data before it (`trading/pit.py`), so no
2024–2026 reading saw its own future. Live tickets use the frozen v3
model; the two agree to about 0.05 on the M_pct scale in late 2026.

## The declared families (`research/trading/`)

Three independent designers proposed eight families without looking at
any post-signal return. Their lenses were empirical asset pricing, risk
management and options practice. A judge declared three, with ten
variants in total. `candidates.md` holds every rule and threshold.

* **SR, systemic-shock liquidity reversal (long).**
  - Rule: on a systemic day, buy the ≤ 10 names hit ≥ 2σ whose beta- and segment-adjusted residual is ≤ −1.5σ, excluding names that gapped down on their own news.
  - Hold 5 sessions; variants with 10 sessions, a stress dose (M_pct ≥ 1) and a beta-only residual.
  - Mechanism: on broad fear days much of the selling carries no information, and liquidity providers are paid to absorb it, more so under stress (Campbell, Grossman & Wang 1993; Nagel 2012).
* **ND, idiosyncratic news drift (bearish information).**
  - Rule: on a calm day, a stock-specific overnight gap-down of ≥ 2σ with a ≥ 3σ low.
  - It is measured as information, because the account cannot short.
  - Live it means selling the name if it is held (a typed exit), or puts at Tier 2.
* **RF, fear-gated de-risking (a risk tool; no return claim).**
  - Rule: scale the core AI holdings down when market fear is high.
  - It is adopted only if it cuts variance by ≥ 15% with no measurable utility cost and with no worse CVaR or drawdown.

## Protocol

1. **Declaration.** It was committed (4341c9b) before any rule was run. The
   implementation reproduces every one of the judge's signal-side event
   counts exactly.
2. **Pre-backtest audit.** Four independent auditors (spec, look-ahead,
   accounting, statistics) worked under a no-returns rule. Ten minor issues
   were found and fixed before the dev run (`research/trading/audit.md`).
   The largest was a mislabelled half-day (2026-07-02).
3. **Placebo calibration.** Each declared variant kept its real signal
   days and names per day, but the names were drawn at random. The gate
   was run on 300 such worlds. Its family-wise false-adoption rate is
   below.
4. **Dev selection, 2024-01-02..2025-12-31.** Gates (a)–(d) applied:
   - (a) t ≥ 1 for the 1-SE-chosen variant;
   - (b) calibrated SPA_c within Holm (0.05 each);
   - (c) the mechanism signatures;
   - (d) a positive mean at 25 bps per side and with a one-day delay.
5. **Holdout, 2026-01-02..2026-11-27, read once.** It is unlocked by the
   hash of the committed selection. It is not pristine: an exploratory
   event study touched 2026 with different rules.
6. **Prospective.** Every live ticket is logged before its session.

## Execution and accounting (`trading/engine.py`)

* **Timing.** A signal at the close of t executes at the open of t+1 and
  exits at the open H sessions later. The robustness check uses t+2.
* **Costs.** 10 bps per side, with a 25 bps stress test.
* **Tradability.** A close at t, an open at t+1, and a 20-session median
  dollar volume of at least $5M.
* **Gate series.** The calendar-time portfolio (Fama 1998): each day,
  equal weight across all open positions, each measured against β × the
  equal-weight basket bought at the same open.
* **Benchmark.** It nets out the AI-boom tailwind that every long position
  in this hindsight-chosen universe enjoys.

## Live tickets, tiers and sizing

| tier | requires | risk budget per ticket |
|---|---|---|
| 0, display | anything less than Tier 1 | none, never traded |
| 1, pilot | adopted, and the 2026 holdout agrees | ≤ 0.25% of NAV (1-s.d. loss) |
| 2, standard | adopted, and 2026 t ≥ 1 | ≤ 0.5% of NAV |
| 3, high conviction | Tier 2 plus the ticket conditions: the strongest third of dev scores, M_pct ≥ 1, no earnings in the window, not a half-day | ≤ 1% of NAV |

**Sizing**, declared and never a selection input:
- quarter-Kelly on a shrunk per-trade IR with an N(0, 0.05²) prior;
- notional = risk budget ÷ the regime-adjusted volatility over the
  holding period;
- caps per name, per segment and in total;
- drawdown brakes.

On dev the fear model's volatility yardstick beat EWMA(0.94) on QLIKE at
5 and 20 sessions by about 7 and 6 s.e., so sizing uses it
(`outputs/trading/sizing_calibration.json`).

**Options**, live only. SR can be expressed as a cash-secured put when
the put's IV is at least 1.25× the model's forward volatility, after a
real expired-contract check enables that branch. One contract ties up
100 × the strike in cash, so in a small account most SR tickets are
shares.

```bash
fearsignal trading-placebo                 # gate calibration on placebo families
fearsignal trading-dev                     # once: writes config/trading_selected.json
fearsignal trading-holdout --sha256 <sha256 of config/trading_selected.json>   # once
fearsignal trading-live --days 6           # both models on recent sessions, and the next session
```
