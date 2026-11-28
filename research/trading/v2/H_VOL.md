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
