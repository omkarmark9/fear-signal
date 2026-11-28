# H_VOL: does model 1's volatility forecast price options? (pre-registration)

Written 2026-11-28, **before any option price for this test was collected**.
The only option data seen so far is one pilot bar: the MSFT 2025-03-21
$410 call, used to check that history is retrievable.

## Why this test

Model 1's one validated skill is forecasting volatility. Its
regime-adjusted yardstick σ̃ beat EWMA(0.94) on QLIKE:
- on dev, by about 7 s.e. (h 5) and 6 s.e. (h 20);
- on 2026, out of sample with dev calibration, by −0.046 and −0.060 QLIKE
  (t −1.30 and −1.35).

Options are priced off implied volatility (IV). If σ̃ carries information
about future realised volatility (RV) that IV lacks, options are cheap
when σ̃ > IV and rich when σ̃ < IV. This is the volatility-forecast version
of Goyal & Saretto (2009) and the variance-risk-premium literature
(Bollerslev, Tauchen & Zhou 2009).

## Sample (fixed now)

- **Names (10):** AMD, MU, INTC, QCOM, AMAT, ORCL, PLTR, CRM, TSM, MSFT.
  All are liquid, have listed monthly options, and had no split in
  2024–2026.
- **Expiries:** the standard monthly expiry (third Friday) from 2024-02
  through 2026-11.
- **Entry session:** the last session at least 28 calendar days before
  expiry.
- **Contracts:** the call and the put at the listed strike nearest the
  underlying's raw close at entry.
- **Prices:** the option's daily close on the entry session, from
  Robinhood's historical bars for expired contracts.
- **Exclusions:** a pair whose entry-session bar is missing, or whose
  price breaks no-arbitrage bounds, is excluded and counted.

## Quantities

- **IV:** the average Black–Scholes implied volatility of the call and the
  put (r = 4%, no dividend yield), with T in calendar days / 365.
- **σ̃:** model 1's regime-adjusted forward volatility over the session
  count to expiry. It uses the dev-calibrated κ, point-in-time
  probabilities and refits, and is annualised.
- **RV:** realised volatility of daily log returns from entry to expiry,
  annualised.
- **Straddle return:** held to expiry, (|S_T − K| − C − P) / (C + P),
  with a round-trip cost of 5% of premium (half-spreads on two legs).

## Hypotheses and tests

**Dev:** entries with expiry ≤ 2025-12-31. **2026:** expiries
2026-01..2026-11, read once.

| id | hypothesis | statistic | claimed if |
|---|---|---|---|
| H_VOL1 | σ̃ adds information about RV beyond IV | OLS of log RV on [1, log IV, log σ̃], s.e. clustered by expiry month; the coefficient c on log σ̃ | dev one-sided p < 0.05 for c > 0 |
| H_VOL2 | straddles bought when σ̃ ≥ IV earn more than those bought when σ̃ < IV | the difference in mean net straddle return between the two groups, with a bootstrap over expiry months (2,000 draws, months resampled) | dev one-sided p < 0.05 for a positive difference |
| H_VOL3 | straddles bought when σ̃ ≥ IV make money | the mean net return of that group, bootstrapped over months | reported (no claim unless H_VOL2 holds) |

**Decision.**
- If H_VOL2 holds on dev and its 2026 difference is > 0, the rule "buy
  the straddle when σ̃ ≥ IV" enters the prospective test as a Tier 1
  pilot.
- Otherwise it is recorded as not supported.

There is no parameter search: the 28-day entry, the ATM strike and the
1.0 ratio are fixed here.

## Operational clarifications

These were added before any sample price was collected. Two bars had been
seen by then, both from checks that the API works: the MSFT pilot bar and
one AMD bar (the 2024-02-16 $175 pair, from a format check).

- **Holiday expiries.** When the third Friday is a market holiday, the
  standard expiry is the session before it. In this window that is
  2025-04-17 (Good Friday) and 2026-06-18 (Juneteenth). The entry session
  is dated from that expiry.
- **Strike rule.** The candidates are round(S/g)·g for g in {0.5, 1, 2.5,
  5, 10}, where S is the raw close at entry. They are tried in order of
  distance from S, lower strike first on ties. The first candidate listed
  as both a call and a put for that expiry is used. If the strike grid
  near the money is any of those spacings, this is the nearest listed
  strike.
- **Missing bars.** A bar flagged `interpolated` counts as missing.
- **Forecast scale.** Comparisons use total volatility over the option's
  life, as follows.
  - σ̃_tot = κ₂₀ · σ̃(h), with h the number of sessions after entry up to
    and including expiry, and κ₂₀ the dev calibration.
  - IV_tot = IV · √T.
  - RV_tot = √(Σ r²) over those h sessions.
  - "σ̃ ≥ IV" means σ̃_tot ≥ IV_tot.
  - The H_VOL1 regression uses the logs of these three totals.
- **Settlement price.** S_T is the raw close on the expiry session.
- **Secondary analysis (reported, never a claim).** H_VOL1 and H_VOL2 are
  repeated on the pairs whose life contains no earnings reaction session,
  taken from the v2 earnings calendar. σ̃ has no earnings information,
  while IV prices earnings, so this shows how much of any effect comes
  from earnings.

## Amendment 1 (2026-11-28, after the first collection pass, before any analysis)

The first pass chose the nearest strike among all strikes ever listed for
the expiry. Robinhood's instrument list includes strikes the exchange added
**after** the entry session. For those strikes, the entry-session bar is a
placeholder: either flagged interpolated, or flat at the 0.01 minimum tick
with no flag. For example, the MSFT 2026-09-18 $482.5 call has gap-fill bars
until 2026-08-21, a flat 0.01 bar on that entry day, and real prices from
2026-08-24. This affected 136 of the 340 pairs.

Such a contract could not have been bought at entry. Dropping these pairs
would also select on the later price path, because strikes near the entry
spot tend to be added when the stock stays near it.

**The amended rule** is to use the nearest strike **listed at entry**.
- Candidates are tried in the same order as before.
- The first candidate is used whose call and put both have a real
  entry-session bar, meaning it is not interpolated and its close is above
  0.01.
- If no candidate qualifies, the pair is excluded and counted as
  `not_listed_at_entry`.

A second collection pass applies this rule to the 136 pairs. The first
pass's raw records are kept in `data/raw/hvol_pass1.jsonl`. Like all
vendor data, they are not committed; their SHA-256 is in
`config/hvol_inputs.json`.

At this point no IV, forecast, realized volatility or straddle return had
been computed for any pair.

## Results (2026-11-28; each period read once, logged in `outputs/trading/v2/HVOL_RUNLOG.md`)

Outputs are `outputs/trading/v2/hvol_dev.json` and `hvol_2026.json`.
Per-pair derived quantities are in `hvol_pairs_*.csv`; vendor prices are
omitted.

**Sample.** 339 of 340 pairs are usable. MU 2025-03-21 is excluded: its
only remaining candidate, the $100 strike, had a real call bar at entry,
but the put had no entry bar. Amendment 1 needs both legs. Of the 339, 204 strikes come from pass 1
and 135 from pass 2. Nothing broke the no-arbitrage bounds. The median
absolute put-call parity residual is 1.0% of premium on dev and 0.6% in
2026.

| | dev (expiries 2024-02..2025-12) | 2026 (2026-01..2026-11) |
|---|---|---|
| pairs (months) | 229 (23) | 110 (11) |
| median IV | 40% | 51% |
| mean log(RV/IV), so RV ran above IV | +0.032 | +0.082 |
| share with σ̃ ≥ IV | 62% | 79% |
| **H_VOL1**: c on log σ̃, beyond log IV | 0.12 (s.e. 0.12, t 0.97, p 0.17) | 0.18 (s.e. 0.22, t 0.82, p 0.21) |
| coefficient b on log IV | 0.87 (t 7.2) | 0.86 (t 3.9) |
| **H_VOL2**: mean net straddle, σ̃ ≥ IV minus σ̃ < IV | +14.5 points (+17.0% vs +2.5%), **p 0.053**, 95% CI [−2.2, +32.2] | +27.6 points (+33.6% vs +6.0%), p 0.11 |
| **H_VOL3**: mean net, σ̃ ≥ IV group | +17.0%, p 0.028 | +33.6%, p 0.017 |
| mean net straddle, all pairs (median) | +11.4% (−8.0%) | +27.8% (+11.8%) |
| secondary, no earnings in life: H_VOL2 | +22.7 points, p 0.050 (124 vs 28 pairs) | +40.8 points, p 0.028 (67 vs 8 pairs) |

**Decision (as declared).**
- H_VOL1 is not claimed in either period.
- H_VOL2 fails on dev: p = 0.053 against the 0.05 threshold. So the
  straddle rule is **not supported** and does not enter the prospective
  test, even though its 2026 difference is positive.
- H_VOL3 is reported only.

**What it means.**
- **σ̃ against IV.** Model 1's forecast adds no *significant* information
  about realized volatility beyond implied volatility.
  - Its coefficient is positive in both periods (c = 0.12 and 0.18).
  - IV carries most of the forecast.
- **σ̃ against EWMA.** The forecast beats a historical-volatility
  forecaster (EWMA), on dev and in the 2026 QLIKE re-check. Against the
  options market the evidence is inconclusive; see the post-hoc
  diagnostic below.
- **The straddle sorting.** The σ̃ ≥ IV group did better in both periods.
  That is a consistent direction, but by the declared rule neither period
  passes on its own.
- **Long volatility paid in general.** Realized volatility ran above
  implied in both periods, so the average ATM straddle on these names
  made money. On dev, though, the median straddle lost.
- **What model 1 did and did not show.**
  - **The level.** Model 1 often put forecast volatility above IV: σ̃ ≥ IV
    in 62% and 79% of pairs. That group carried the gains, +17.0% and
    +33.6% against +2.5% and +6.0%.
  - **The sorting.** What failed is the declared test, the
    sorting between the two groups.
  - **The caveat.** σ̃'s level factor κ was fitted in-sample on dev
    realized volatility over all 134 names, and in 2026 σ̃ ran about 10%
    above realized volatility. So the level signal is not clean evidence.
  - **The usual sign.** The variance risk premium usually runs the other
    way (IV > RV). No rule is drawn from the period's long-volatility
    gains.

**Post-hoc diagnostic (not pre-registered; `hvol.posthoc_qlike`,
`outputs/trading/v2/hvol_posthoc_qlike.json`).**
- QLIKE of each forecast's variance against RV at the option horizon,
  with month-clustered differentials against raw IV. **None of the
  differentials is significant.** The "IV × dev level" factor is fitted on
  the dev pairs themselves, so that row is in-sample on dev.

  | forecast | dev | 2026 |
  |---|---|---|
  | σ̃ | 0.346 (+0.096 vs IV, t 1.5) | 0.266 (+0.045, t 0.4) |
  | raw IV | 0.249 | 0.222 |
  | IV × dev level factor 1.127 | 0.209 (−0.040, t −1.3) | 0.166 (−0.056, t −1.5) |
  | geometric mean of σ̃ and IV | 0.260 (+0.011, t 0.4) | 0.189 (−0.033, t −0.5) |

- **Reading.**
  - The point estimates favour listed IV over σ̃ (differential t 1.5 on
    dev, 0.4 in 2026), and the blend of the two did better than raw IV in
    2026.
  - Whether IV would make a better sizing yardstick is a hypothesis for a
    future pre-registered test, not a finding.
  - The declared ρ = IV/σ̃ put filter is a cap on what to pay, not an
    edge.

## Review corrections (2026-11-28)

An independent review (`REVIEW_hvol_prospective.md`) recomputed every
number above from the raw files with its own code, and all of them match.
"Not supported" stands. The wording of the post-hoc diagnostic, the
reading of the level signal and the MU label were corrected above
(HV-1, HV-3, HV-5). The review also added these notes:

- **Monte Carlo precision (HV-2).**
  - With B = 2,000 and seed 0, both fixed in code at 61fa3ab before the
    read, dev H_VOL2 has p = 0.053.
  - The converged p (B = 400,000) is 0.054. Over 400 seeds at B = 2,000,
    the p has an s.d. of 0.005, and 18% of seeds fall below 0.05.
  - The declared draw decides. The result is a near miss either way.
- **κ sensitivity (HV-2, post hoc).** H_VOL2 depends strongly on the
  level of σ̃, because κ moves the threshold:

  | κ₂₀ | dev p | 2026 difference |
  |---|---|---|
  | 0.852 | 0.014 | +1.3 points |
  | 0.90 | 0.079 | +9.8 points |
  | 0.952 (declared) | 0.053 | +27.6 points |
  | 1.00 | 0.150 | +21.6 points |

  So the size of the "consistent direction" is fragile.
- **The interpolated flag (HV-6).**
  - Pass 2 checked 238 candidate strikes, of which 202 were listed. 93 of
    the 202 listed strikes had no `interpolated` field on their bars; the
    other 36 candidates were not listed, so they had no bars at all.
    (An earlier version said "129 of the 238 bars", which counted the
    unlisted candidates as bars.)
  - Every flagged placeholder had closes of exactly 0.01: 16 pairs, or
    32 bars, since both legs were flagged each time (15 pairs from pass 1
    and 1 from pass 2). So the "close above 0.01" test caught every known
    placeholder.
  - Pass-2 pairs' parity residuals (median 0.81%) match pass 1's (0.88%).
- **Dropped draws (HV-8).** A bootstrap draw with no pair in one group is
  dropped. The 2026 secondary used 1,999 of 2,000 draws.
- **Stale prints (HV-9).** The same entry close sets both the sort (IV)
  and the return's denominator. Stale prints therefore bias H_VOL2 and
  H_VOL3 slightly towards finding an effect, so "not supported" is
  conservative. The parity residuals are small: 1.0% of premium on dev,
  0.6% in 2026.
- **Provenance (HV-7).**
  - A rerun now records the price and model inputs, the earnings calendar,
    κ and the code state.
  - For the recorded reads, every one of those inputs matched its lock
    when the review checked them.
  - The reads ran at 61fa3ab: the inputs lock is that commit's, and the
    outputs were committed at 5a96922.
- **Exclusion counts (HV-4).** They are now kept per period. No pair was
  excluded after collection in either period. Dev's one exclusion, MU
  2025-03-21, is `not_listed_at_entry` (an earlier version said "both
  periods had none").
