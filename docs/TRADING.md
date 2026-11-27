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

## Results (2026-11-27)

**Today the trading model issues no high-conviction trade.** One family
is active at Tier 1, ND, which says to step aside from a stock after a
stock-specific news gap-down. No family reached Tier 2 or 3.

### Placebo calibration of the gate (`outputs/trading/placebo.json`)

Every declared variant was rebuilt 300 times, keeping its real signal days
and names per day but drawing the names at random. The gate then adopted
SR 5.0%, ND 4.0% and either family 9.0% of the time (MC s.e. 1.65%), at
B = R = 199. At the real run's B = R = 999, an exact calculation puts
this about 1.17× higher: roughly 5.9%, 4.7% and 10.5%. All of these are
before the signature and robustness gates.

### Dev selection, 2024-01-02..2025-12-31 (`outputs/trading/dev/`)

| family | chosen | gate series | event excess | calibrated SPA p | signatures | 25 bps / delay | verdict |
|---|---|---|---|---|---|---|---|
| SR, systemic reversal (long) | SR1 | −3.4 bp/day, t −0.86 | −0.30% per 5 sessions (490 events) | 1.00 | S1 ✗, S2 ✓, S3 ✗ | ✗ | **not adopted** |
| ND, news drift (bearish information) | ND1 | +17.5 bp/day, t 2.54 | +2.32% per 20 sessions (145 events) | 0.020 | N1 ✓, N2 ✓, N3 ✓ | ✓ | **adopted** |
| RF, de-risking overlay | RF1 | variance ratio 0.842; utility +2.7 bp/day | — | (not a Holm family) | A1 ✓, A2 ✓, A3 ✓ | ✓ | **adopted** |

* **SR.** All four variants are negative against both the beta-adjusted and the
  plain basket. The S2 contrast "passes" only because the same rule on
  non-systemic days does even worse (−0.54% per event).
* **ND.** It holds up across several checks:
  - the intercept beats realized beta (a = +1.66%);
  - the drift is gradual: 0.67% accrues by the close of session 2;
  - it needs the news gap: 2.32% with a gap against 1.23% without one;
  - it survives the same-stock baseline (+1.64%).

  One prediction failed. The most displaced third (v ≤ −5.15) drifted
  least, 0.40% against 3.31% for the rest. That also undercuts ND's
  declared Tier-3 ticket condition. Dose in x went the predicted way: 3.0%
  at x ≥ 4 against −0.5% at 3 ≤ x < 4.
* **RF.** Its reported signatures all hold:
  - the basket's return per unit of variance is lower when de-risked;
  - multi-week episodes contribute the gain;
  - a one-day delay keeps 99% of it.

  Its Sharpe gain over 100% basket (1.55 against 1.32) is not significant
  (95% CI −0.15 to +0.56).

### Holdout, 2026-01-02..2026-11-27, read once (`outputs/trading/holdout/`)

| family | 2026 | verdict |
|---|---|---|
| ND1 (confirmation read) | +6.4 bp/day, t 0.61; +2.08% per event (54 events) | agrees, with t < 1: **Tier 1 (pilot)** |
| RF1 (confirmation read) | variance ratio 0.992; utility −2.7 bp/day (−0.8 s.e.); CVaR −5.9% against −5.6%; drawdown −35.8% against −34.0% | **not confirmed: OFF** |
| SR1 (descriptive) | +19.8 bp/day, t 2.09; +0.36% per event (299 events) | **Tier 0**: failed dev, so it stays unadopted |

### What the results mean

* **The bearish information signal is the one with evidence.** A stock that gaps down on
  its own news, on a day the rest of the AI complex is calm, has
  underperformed its beta-adjusted benchmark by about 2% over the next 20
  sessions. That held in 2024–25 (t 2.5), and 2026 gave a similar size per
  event (+2.1%) on too few events to confirm it. The account cannot
  short, so the live use is a **typed exit**: sell the name if it is held,
  and don't buy it for 20 sessions. Puts would need Tier 2.
* **The liquidity-reversal idea did not survive its own test.** SR lost
  in 2024–25 and then made money in 2026. A rule whose sign flips between
  periods is exactly what the protocol exists to keep out of a
  high-conviction label. A new SR version could be declared now and
  judged prospectively. Adopting it on the 2026 result would turn the
  holdout into a selection sample.
* **Fear-gated de-risking did not cut risk in 2026.** It helped in 2024–25,
  but in 2026 its variance ratio was 0.99 and its tails were slightly
  worse, so the overlay stays off. Why it failed has not been analysed.

### Caveats

* **Hindsight-chosen universe.** These are 2026's AI winners. That biases
  long rules up. It works against bearish rules like ND, because names
  that kept falling after bad news are under-represented. It also means
  only the prospective log is a clean test.
* **The 2026 holdout is not pristine.** An exploratory event study touched
  2026 with different rules (buying Z ≥ 2.5 and M ≥ 2 on raw returns).
* **Sample sizes.** ND has 145 dev and 54 holdout events. RF has 12 dev
  stress episodes.
* **Options.** No option trade is allowed at the current tiers. In a
  $10,000 account a cash-secured put is usually impossible anyway: one
  contract ties up 100 × the strike.


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
