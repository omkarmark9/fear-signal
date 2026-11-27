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

**Today the trading model issues no high-conviction trade.** One family,
ND, is at Tier 1 (pilot), and it stands on its 2024–25 evidence alone. Its
live use is a **rotation**: when a new signal fires, sell the flagged
name at the next open and hold the rest of the AI basket in its place for
20 sessions. No family reached Tier 2 or 3. An independent results review
(`research/trading/review.md`) reproduced every number below exactly from
the hash-locked inputs. It then stress-tested them, and its findings are
part of this section.

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
| SR, systemic reversal (long) | SR1 | calendar-time −3.4 bp/day, t −0.86 | −0.30% per 5 sessions (490 events) | 1.00 | S1 ✗, S2 ✓, S3 ✗ | ✗ | **not adopted** |
| ND, news drift (bearish information) | ND1 | calendar-time +17.5 bp/day, t 2.54 | +2.32% per 20 sessions (145 events); clustered t 1.80 by day, 1.57 by episode | 0.020 (0.057 on the 1/H series) | N1 ✓, N2 ✓, N3 ✓ | ✓ | **adopted, marginally** |
| RF, de-risking overlay | RF1 | variance ratio 0.842; utility +2.7 bp/day | — | (not a Holm family) | A1 ✓, A2 ✓, A3 ✓ | ✓ | **adopted, on one episode** |

**SR.** All four variants are negative against both the beta-adjusted
and the plain basket. Against random names on the same days, SR1 sits at
the 52nd percentile, so its selection carried no information. The S2
contrast "passes" only because the same rule on non-systemic days did
even worse.

**ND: real on dev, but marginal.**
- **For it:**
  - against random names drawn on the same days, it sits at the 99.3rd percentile;
  - it holds at 50 bps (t 1.93), with β = 1 (t 2.74) and with a one-day delay (t 2.10);
  - the drift is gradual: 0.67% accrues by the close of session 2;
  - it needs the news gap: 2.32% against 1.23% without one;
  - it survives the same-stock baseline (+1.64%);
  - names that became 2026's big winners hurt it rather than help it.
- **Against it:**
  - the per-event t is 1.6–1.8 clustered, not 2.5;
  - the pass on gate (b) hinges on single names. Dropping any one of ALAB (2025-09-24), AEHR (2025-01-14), MU (2024-06-27) or BIDU (2024-10-03) lifts the calibrated SPA p to 0.06–0.11, outside Holm. Gate (a), t ≥ 1, holds for every single removal (t 1.88–2.95);
  - the effect is concentrated: software and semicap supply 89% of the per-event sum, and 2024 is stronger than 2025 (t 2.54 against 1.12);
  - the declared dose prediction in v failed: the most displaced third drifted least, 0.40% against 3.31%. That also undercuts ND's declared Tier-3 condition.
- **Partly seen in advance:** the earlier exploratory event study had already computed outcomes that overlap 94 of ND's 145 dev events (56% share the event day), and ND's edge sits in those events (+3.31% against +0.51%). See the caveats.

**RF.** Its dev pass rests on one episode:
- without April 2025 the variance ratio is 0.954;
- a stationary bootstrap gives P(ratio > 0.85) = 0.53;
- its reported signatures hold (lower return per unit of variance when
  de-risked, multi-week episodes, 99% of the gain kept with a delay);
- the Sharpe gain (1.55 against 1.32) is not significant (95% CI −0.15
  to +0.56).

### Holdout, 2026-01-02..2026-11-27, read once (`outputs/trading/holdout/`)

| family | 2026 | verdict |
|---|---|---|
| ND1 (confirmation read) | calendar-time +6.4 bp/day, t 0.61; +2.08% per event (54 events, clustered t 1.0) | "agrees" (mean > 0) with t < 1: **Tier 1 (pilot)**, by the declared rule. The read itself is uninformative; see below. |
| RF1 (confirmation read) | variance ratio 0.992; utility −2.7 bp/day (−0.8 s.e.); CVaR −5.9% against −5.6%; drawdown −35.8% against −34.0% | **not confirmed: OFF** |
| SR1 (descriptive) | calendar-time +19.8 bp/day, t 2.09; +0.36% per event (299 events, clustered t 0.5–0.8) | **Tier 0**: failed dev |

**Why ND's 2026 read confirms nothing:**
- 40% of random-name draws on the same days also "agree";
- one lone position (INOD, the only one open 2026-11-03..11-25) supplies 4.1 of the 6.4 bp/day;
- entering one day later gives −8.5 bp/day;
- one earnings cluster (17 events in April–June) supplies 88% of the per-event sum;
- the gradual-drift signature fails in 2026 (45% of the drift arrives by session 2).

The Tier-1 label is the declared tier mechanically applied. It is a pilot
on dev evidence, not a confirmation.

**SR's 2026 gain comes from two news jumps:** APP (+70%) and SIMO (+54%),
both through overnight gaps inside the hold. Without them the per-event
mean is −0.06%. The median is −0.06%, and the clustered t is 0.5–0.8. So
SR shows no per-event edge in either period, and nothing here justifies a
new SR version.

**RF's 2026 failure is not a reversal.** Its 0.99 ratio matches dev with
April 2025 left out (0.95). It fails at every γ, at 25 bps, with a delay
and against the causal static mix.

### What the results mean

* **The one signal with evidence is bearish, and it is a rotation.** On
  2024–25, a stock that gapped down on its own news on a calm day trailed
  β × the AI basket by about 2% over the next 20 sessions, an effect that
  random names do not produce. That payoff is selling the name and
  holding the basket. Selling to cash would have earned nothing: −0.52%
  per event on dev, +0.42% in 2026, because the basket rose meanwhile. A
  sell started inside an already-open window was never tested. So live:
  - a new ND signal means rotating out of the name, if it is held, at the
    next open;
  - an open window means only "do not add".
* **Long liquidity reversal (SR) and fear-gated de-risking (RF) did not
  survive their own tests,** and both stay off.

### Caveats

* **Hindsight-chosen universe.** These are 2026's AI winners. That biases
  long rules up. It works against bearish rules like ND, and the data
  agree: the later big winners hurt ND. It also means only the
  prospective log is a clean test.
* **Neither test sample is pristine for ND.** The exploratory event study
  (commit 9ead1bb, before this model existed) measured 20-session outcomes
  after Z ≥ 2.5 episodes. Those overlap 65% of ND's dev events and 48% of
  its 2026 events, and the designers cited its result. The 2026 holdout
  was also touched by that study.
* **Sample sizes.** ND has 145 dev and 54 holdout events. RF's dev pass is
  one crash.
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
  - It is measured as information, because the account cannot short. The
    payoff tested is β × the basket minus the name.
  - Live, a new signal means rotating out of the name at the next open, if
    it is held, into the rest of the AI basket for 20 sessions (a typed
    exit). An open window means "do not add". Puts are allowed only at
    Tier 2.
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
