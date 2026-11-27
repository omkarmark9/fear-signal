# Trading model: declared candidate families

Declared 2026-11-27, before any backtest of any rule below. The judge's
output is in `declaration.json`, verbatim. This file states how it is
implemented, in `src/fearsignal/trading/families.py`. Where the declaration
needed an interpretation, the choice is listed under "Implementation notes"
and was fixed before any result existed.

**Inputs, fixed by hash** (git-ignored files in `data/raw/`):

| file | SHA-256 |
|---|---|
| `pit_signals.parquet` (point-in-time fear outputs) | `b02560aa774022b180b4b512d160ce6a1411e7c2e46073d09c16dce243cc0096` |
| `pit_market.parquet` | `18d389b8095c5b4391da753593a9e84e99ea47ece7aee9583726e73bc3b104bb` |
| `pit_refits.json` | `65b2ba587c748a1148005d6c910e4af7c2f2aa61a613e21f0e8f4440217c9257` |
| `trade_ohlcv.parquet` (prices) | `6cae3d1d6851cb7c2222d311ed439b2b6089f640def2172d71a58769055cfa25` |

## Common definitions (`signals.rule_features`)

Every quantity is known at the close of t.

| symbol | definition |
|---|---|
| r, σ, x, eligible | the fear model's own features: log return, robust σ (126 sessions, today excluded), drawdown-to-low in σ, eligibility |
| systemic_t | share_hit2_t ≥ 0.10, where share_hit2_t is the share of eligible names with x ≥ 2 |
| half_day_t | NYSE early close: 2024-07-03, 2024-11-29, 2024-12-24, 2025-07-03, 2025-11-28, 2025-12-24, 2026-07-02, 2026-11-27 |
| rB_t | equal-weight mean of eligible names' r |
| β_i,t | ⅔ × OLS slope of r_i on rB over t−126..t−1 (≥ 100 observations, same rows as the stock) + ⅓ |
| u_i,t | (r_i,t − β_i,t · rB_t) / σ_i,t |
| v_i,t | u minus the median u of eligible names in the segment; the all-eligible median if the segment has fewer than 3 |
| γ_i,t | ln(adj open_t / adj close_t−1) / σ_i,t |
| g_seg, g_all | γ minus the segment median (same fallback), or minus the all-eligible median |
| price_ok | raw close_t ≥ $5 |
| liquid20 | median of raw close × volume over t−20..t−1 ≥ $20M |
| M_pct_t | point-in-time market fear in backtests, frozen v3 live |

**Execution** (`engine.py`, fixed before this declaration):
- buy at the open of t+1 and sell at the open of t+1+H;
- costs of 10 bps per side, with a 25 bps stress test;
- engine tradability: a close at t, an open at t+1, and a 20-session median dollar volume of at least $5M;
- no re-entry while held, and a per-day cap in priority order (`signals.suppress_reentry`).

**Gate series:** the calendar-time portfolio `ct_excess_net`. Each session,
every open position gets equal weight, measured against β_i,t × the
equal-weight basket bought at the same open, net of costs. The series is 0
on sessions with no open position. The 1/H-cohort series is reported too.

## SR: systemic-shock liquidity-provision reversal (long; Holm slot 1 of 2)

**Rule.** At the close of t, name i qualifies if all of these hold:
1. systemic_t and not half_day_t;
2. eligible, price_ok and liquid20;
3. x ≥ 2;
4. v ≤ −1.5;
5. g_seg > −2;
6. not already held (no re-entry until t+H);
7. it is among the 10 new names with the most negative v that day.

Side long, beta benchmark.

**Variants, in 1-SE order:**

| variant | change | H |
|---|---|---|
| SR1 | as stated | 5 |
| SR2 | as stated | 10 |
| SR3 | only days with M_pct_t ≥ 1.0 | 5 |
| SR4 | u instead of v, g_all instead of g_seg, cap ranked by u | 5 |

**Required signature** (gate c), on the chosen variant's dev events:
- **S1, not realized beta.** The OLS of event excess e on [1, R_B], with CR1 errors clustered by signal day, has intercept a > 0.
- **S2, shock-type contrast.** The mean e exceeds that of the identical rule on non-systemic days.
- **S3, hindsight guard.** The mean of e − base_i(H) is > 0, where base_i(H) is stock i's mean H-session beta-adjusted excess over every dev signal day (`backtest.same_stock_baseline`).

**Reported** (not gating):
- the slope of e on M_pct;
- front-loading over sessions 1–3;
- permanence at session 20;
- the news-excluded set (g_seg ≤ −2);
- v terciles;
- liquidity halves;
- leave-one-cluster-out;
- SR4 by segment.

**Signal-side counts, reproduced exactly by the implementation:**

| | dev events / days | 2026 events / days |
|---|---|---|
| SR1 | 490 / 122 | 294 / 70 |
| SR2 | 459 / 118 | 272 / 66 |
| SR3 | 328 / 77 | 146 / 37 |
| SR4 | 659 / 123 | 442 / 73 |
| S2 contrast | 473 / 242 | |
| news-excluded set | 131 | |

**Options expression** (live only, never in selection):
- Candidate: the put nearest −0.35 delta, at the first expiry at least H and at most H+5 sessions after entry.
- Filters: spread ≤ 10% of mid, OI ≥ 250, mid ≥ $0.30, no earnings before expiry, put-call IV gap ≤ 5 points.
- Richness: ρ = IV·√(D/365) / σ̃_i(h).
- Choice:
  - ρ ≥ 1.25: a cash-secured put, or a buy-write if call IV ≥ put IV + 2;
  - ρ ≤ 0.90: a +0.60-delta call;
  - otherwise: shares.
- The CSP branch is enabled only if a real expired-contract check passes: of up to 12 contracts, the median ρ ≥ 1.25 and at least 8 have ρ ≥ 1.0.

## ND: idiosyncratic news-shock drift (bearish information; Holm slot 2 of 2)

**Rule.** At the close of t, name i qualifies if all of these hold:
1. not systemic_t and not half_day_t;
2. eligible, price_ok and liquid20;
3. x ≥ 3;
4. v ≤ −2;
5. g_seg ≤ −2 (a stock-specific overnight gap-down: news outside market hours);
6. no re-entry;
7. it is among the 5 new names with the most negative v.

It is not traded as shares (the account cannot short). The information
measured is β·R_B − R_i − round-trip cost (side `short_info`, beta
benchmark).

**Variants, in 1-SE order:**

| variant | H |
|---|---|
| ND1 | 20 |
| ND2 | 40 |
| ND3 | 10 |

**Required signature:**
- **N1.** Intercept a > 0, as in S1.
- **N2, gradual.** The mean short-side excess from the entry open to the close of session 2 (gross) is at most ⅓ of the mean H-session e, and that mean is > 0.
- **N3, information split.** The mean e exceeds that of the no-news contrast (condition 5 becomes g_seg > −1).

**Reported:**
- the contrast on systemic days;
- the second-half share;
- the slope on log dollar volume;
- dose in v and x;
- the same-stock-adjusted sign.

**Counts:**

| | dev events / days | 2026 events |
|---|---|---|
| ND1 | 145 / 104 | 54 |
| ND2 | 135 / 98 | 50 |
| ND3 | 154 / 110 | 58 |
| N3 contrast | 96 | |
| systemic contrast | 105 | |

**Live use, if adopted:**
- a typed exit: sell held shares at t+1 and don't re-buy before t+1+H;
- a long put, at tier ≥ 2 and only if ρ ≤ 0.90.

## RF: fear-gated de-risking overlay (risk forecaster; no Holm slot)

**Exposure.** E_t ∈ [0.25, 1] on the daily-rebalanced equal-weight basket,
decided at full-session closes. A half-day keeps E_t−1.
- σ̂_t: a recursive HAR forecast of the basket's next-5-session daily volatility (`validate.har_design`), fitted on rows s ≤ t−5 from the first complete row, with at least 60 rows.
- σ*_t: the median of σ̂_s over s ≤ t.

**Variants:**
- **RF1, staircase.** E = 0.5 if M_pct ≥ 1.5. Otherwise, if M_pct < 1.0 and E < 1, E rises by 0.25.
- **RF2, gated volatility target.** A stress state switches on when M_pct ≥ 1.5 and off at the first full-session close with M_pct < 1.0. While on, E = clip(σ*/σ̂, 0.25, 1). While off, E rises 0.25 per full session with M_pct < 1.0, and holds while 1.0 ≤ M_pct < 1.5.
- **RF3, comparator without fear input.** The target is clip(σ*/σ̂, 0.25, 1), with a 0.10 no-trade band.

**Test.** `engine.run_exposure` at the open of t+1, with costs of 10 bps × |ΔE|.

**Statistic.** d_t = u(net_t) − u(Ē·cc_t), with u(r) = r − (γ/2)r² and γ = 5.
Ē is the variant's mean intraday exposure over the window.

**Adoption** (it replaces SPA and Holm, because nothing is claimed about
returns). Choose by the 1-SE rule on mean d, then require all three:
- **A1.** Var(net)/Var(Ē·cc) ≤ 0.85.
- **A2.** Mean d ≥ −0.5 s.e. at 10 bps, at 25 bps and with a one-day delay.
- **A3.** CVaR95 and maximum drawdown are no worse than the static mix's.

"Confirmed" also requires A1 and A2 on 2026. RF never produces a
high-conviction trade.

## Gates and selection, for SR and ND

**Window.** Dev is 2024-01-02..2025-12-31. Signals stop so that every
cohort exits by 2025-12-31.

**Gates:**
- **(a)** The variant chosen by the 1-SE rule has t ≥ 1. Its s.e. is the stationary-bootstrap long-run s.e., mean block 20.
- **(b)** The calibrated Hansen SPA_c p-value (B = R = 999, block 20) passes Holm across SR and ND at family-wise 0.10, each threshold capped at 0.05. With m = 2, both thresholds are 0.05.
- **(c)** Every required signature item holds.
- **(d)** The mean stays positive at 25 bps per side and with a one-day delay.

**Placebo calibration**, run before the dev run and reported, not a gate:
- 300 replications, B = R = 199.
- Each declared variant keeps its real signal days and names per day, but the names are random draws from the basket's universe.
- The placebo omits gates (c) and (d), which can only remove adoptions. Its rate is that of a B = R = 199 gate; see SP-4 for the B = R = 999 adjustment.

## Holdout and tiers

**Holdout.** The selected variants are read once on 2026-01-02..2026-11-27,
unlocked by the SHA-256 of the committed `config/trading_selected.json`.
- **Agrees:** mean daily excess > 0.
- **Strong:** t ≥ 1.
- **Not pristine:** an exploratory event study touched 2026 with different rules (buying Z ≥ 2.5 and M ≥ 2 on raw returns).

**Families not adopted on dev** (added before any dev result existed). The
1-SE-chosen variant is still read once on 2026, as a descriptive figure.
It cannot raise the family above Tier 0. RF's chosen variant is read the
same way, and only an adopted RF can be confirmed.

**Tiers.** Only Tier 3 is "high conviction":

| tier | requires | risk budget cap |
|---|---|---|
| 0 | anything below Tier 1 | 0 |
| 1 | adopted, holdout agrees | 0.25% of NAV |
| 2 | Tier 1, and 2026 t ≥ 1 | 0.5% of NAV |
| 3 | Tier 2, plus the ticket conditions | 1% of NAV |

Tier 3 ticket conditions for SR:
- the score is in the dev bottom third;
- M_pct ≥ 1;
- no earnings in the window;
- not a half-day;
- the option conditions hold, if an option is used.

The sizing framework is quarter-Kelly on a shrunk per-trade IR with an
N(0, 0.05²) prior, plus the caps in `declaration.json`. It is the live
implementation and is never a selection input.

## Implementation notes (interpretations fixed before any result)

1. **S2 contrast.** Only the shock type is swapped. The half-day exclusion, any M_pct condition, H, the cap and the priority are kept. This reproduces the judge's count of 473 events on 242 days exactly.
2. **N2.** The partial excess to the close of session 2 is gross of costs, and the H-session e is net. This makes the "at most ⅓" test harder to pass, not easier.
3. **liquid20.** The median uses the values available in t−20..t−1, requiring at least 15.
4. **RF2 switch-off day.** On the day the stress state switches off (M_pct < 1.0), the staircase step applies that same day.
5. **Reported diagnostics** use only dev windows that end by 2025-12-31. For example, permanence at session 20 drops events whose 20th session is in 2026.
6. **Placebo benchmark.** The placebo uses the plain basket, because random names have exactly zero expected excess against it. The real gate uses the beta benchmark as declared.

## Amendments before the dev run (from the pre-backtest audit)

An independent audit workflow checked the committed implementation against
`declaration.json` through four lenses: spec conformance, look-ahead,
accounting, and statistics and protocol. It had one hard rule: compute no
return that follows a signal on real data. Each finding below was verified
and fixed before the placebo, the dev run or the holdout read existed.
The audit's transcript is summarised in `research/trading/audit.md`.

| id | finding | fix | effect on dev / 2026 |
|---|---|---|---|
| SC-1, LA-1 | The declared half-day list included 2026-07-02, but that was a full NYSE session: July 4, 2026 is a Saturday, NYSE closed on July 3, and only the bond market closed early on July 2. Its volume was 1.1× normal, against 0.3–0.7× on every true half-day. The list also omitted 2026-12-24, a real 1 pm close. | The declaration's own definition, "NYSE early close", governs, and the list is corrected. Model 1's forecast list (`forecast.HALF_DAYS`) is corrected the same way. | Dev: none, since all dev half-days are genuine. 2026 signal counts become SR1 299, SR2 276, SR3 151, SR4 446 (from 294, 272, 146, 442); ND is unchanged. |
| SC-2 | RF's HAR fit included 10 fabricated zero basket returns from January 2023, before any name was tradable. | `basket_legs` reports no basket (NaN) on sessions with no tradable name. | RF's σ̂ moves by up to 6% in dev; RF's statistics are computed on the corrected fit. |
| SC-3 | RF2 did not switch its stress state on at a half-day close with M_pct ≥ 1.5. The declaration says "at any close". | RF2's state can switch on at a half-day close; E is still held. | None in dev or 2026 (no such half-day); latent for live use. |
| SC-4, LA-3 | Execution prices ignored the universe's listing dates, so TLN's OTC prints and OKLO's SPAC shell entered the benchmark basket in 2023–24. | `market_data.load` drops rows before each `trade_start`, exactly as the fear panel does. | The H1-2024 basket and RF legs. Signals are unchanged. |
| SC-5 | The input hashes were recorded but never enforced, and a data refresh could overwrite them. | `config/trading_inputs.json` locks the four declared files plus the three fear-panel files. trading-placebo, trading-dev and trading-holdout refuse to run on any mismatch. Live price refreshes go to a separate file. | None. |
| SC-6 | Several declared "reported" diagnostics were not produced. | Added: the plain-basket benchmark and the 1/H series for every variant; SR4's segment breakdown when SR4 beats SR1; RF's Moreira–Muir ratio, episode contributions, the delay-kept share of the variance gain, and the bootstrap Sharpe difference. | Reported only. |
| LA-2 | No early closes were listed after 2026-11-27. | The list covers 2026-12-24 and 2027-11-26. Dating future sessions refuses to run past the tables' coverage. | Live only. |
| AC-1 (major) | The calendar-time gate series counted each position as a full member of the day's average on its two partial days. An entrant has no overnight return and an exiter has no intraday return, so on every turnover day part of the capital sat idle and costs were diluted. Means were shrunk by 3–11% depending on H, which tilts the 1-SE choice toward long holds. This is not the declared "equal weight across all open positions" portfolio. | The series is now computed per leg. The overnight leg averages over positions held at the previous close and charges exit costs. The intraday leg averages over positions held after the open and charges entry costs. The two legs compound. The placebo was rerun on the corrected series. | Every gate statistic. Per-event excess, the 1/H series and the signatures are unchanged. |
| SP-1 (major) | The holdout read did not check that it ran the code the dev selection ran, so an uncommitted rule edit would pass. | The dev run records HEAD and the `src/fearsignal` tree hash in the selection. The holdout read refuses unless the tree is clean and identical. Its start is logged before any 2026 outcome is computed, and its lock file is written before the outcome files. | None. |
| SP-2 (major) | The declaration check covered only 3 files, so gate, engine or feature code could change unnoticed. | trading-placebo and trading-dev refuse unless `src/`, `research/trading/`, `config/` and `tests/` are clean, and they record the code state. | None. |
| SP-3 (major) | `trading-live` would print realized P/L of 2026 holdout events before any read. | Live hides every outcome with a signal on or before 2026-11-27 until a holdout read exists under the committed selection. | None (no live run before the read). |
| SP-4 | The placebo (B = R = 199) was called an upper bound for the real gate (B = R = 999). An exact calculation for the calibrated test puts the real gate's size about 1.17× higher, and the placebo's plain-basket benchmark does not cover beta-estimate error. | The wording is corrected. The rerun on the corrected engine gives SR 5.0%, ND 4.0% and family-wise 9.0% (MC s.e. 1.65%), so about 5.9%, 4.7% and 10.5% at B = R = 999, before gates (c) and (d). | Reporting only. |
| SP-5 | Locking `live.parquet` meant a routine live refresh would block the holdout read. | Backtests read a frozen byte copy, `live_frozen_20261127.parquet`, with the same hash. | None. |
| SP-6 | Live tiers could reuse a holdout read made under a different selection. | A tier above 0, or RF ON, needs a confirmation read made under the committed selection. | Live only. |
| LA-4 | The live report could silently misalign if the price cache and the fear panel ended on different sessions. | It refuses to run unless both end on the same session. | Live only. |
