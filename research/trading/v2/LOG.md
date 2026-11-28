# Trading model v2: research log (checkpoints)

Newest entries at the bottom. Each entry gives the time (UTC), what ran,
the result, the commit, and the next step. To resume after a reset, read
this file and `PLAN.md`, then continue from the last "next" line.

## 2026-11-28, start

**State.** v1 is complete:
- dev selection 3065e17;
- holdout b3be90a;
- review fixes through 6ad09dc.

ND1 is at Tier 1 (pilot, rotation); SR is at Tier 0; RF is off.

**Done.** PLAN.md declares:
- the permutation space: 108 Round-1 trials plus 10 v1 trials;
- the statistics: StepM, SPA, PBO/CSCV, DSR;
- the mechanical selection rule.

It was written before any v2 trial ran.

**Next.**
1. Fetch the earnings calendar for 2023-12..2026-11 (background agent).
2. Build `trading/multitest.py` and `trading/explore.py`, with tests.
3. Register the v1 trials.
4. Run Round 1.

## 2026-11-28, framework built; Round 1 run (except E4)

**Done.**
- `trading/multitest.py`: StepM, PBO/CSCV, DSR, effective trials (eeba2f6).
- `engine.run_weights` (eeba2f6).
- `trading/explore.py`, the runner and append-only registry (f3fcbd7).
- Registered: the 10 v1 trials plus 96 Round-1 trials. Total 106, in
  `outputs/trading/v2/trials.jsonl` and `trial_series.parquet`.

**Round-1 results on dev** (t of each trial's performance series):

| family | t range | note |
|---|---|---|
| E1, news drift (bearish information) | +0.56 to +1.98 | positive in all 32; best calm, x ≥ 3, g ≤ −1.5, H 40 |
| E2, good-news drift (long) | −3.11 to −0.17 | negative in all 16: good-news gaps on calm days reverse (MAX/lottery-like) |
| E3, systemic reversal (long) | −1.77 to +1.54 | v ≤ −1.5 negative; v ≤ −2.5 positive (best H 10) |
| E5, fear factor | −1.35 to −0.20 | Z and I rank identically, so 4 pairs are duplicates |
| P, risk-based construction | −0.62 to +0.81 | no Sharpe gain; EWMA k = 2 best |
| T, overlays | −0.58 to +1.33 | staircase on 1.0, cut 0.5 best; vol target negative |
| V1 | SR −1.30..−0.60, ND +1.65..+2.02, RF +0.88..+1.21 | 1/H series |

**Next.**
1. Run E4 once the earnings calendar is in.
2. Compute the registry statistics: StepM, PBO, DSR.
3. Round 2, adaptive and logged:
   - E2 faded (both signs counted);
   - idiosyncratic-shock and IVOL/MAX avoidance;
   - more extreme E3.

## 2026-11-28, Rounds 2–5 run: 191 trials registered

**Earnings data.** 1,657 report dates (Robinhood calendar, 2023-11..2026-11).
After dropping stale unverified dates, 1,651 remain, covering 1,049
reaction sessions in dev. The SHA-256 is locked in
`config/trading_v2_inputs.json`.

**Results on dev**, as t of each trial's performance series:

| round | family | result |
|---|---|---|
| R1 | E4, earnings reaction | bad reactions as bearish information +1.8 to +2.9 (all 6 positive); good reactions held long −0.6 to −2.2 (all 6 negative) |
| R2 | E2F, good-news gap-ups faded | up to +2.28 |
| R2 | E6, idiosyncratic gap of either sign | +2.60 (H 10) |
| R2 | MAX / IVOL avoidance | −0.35 / −0.96: the classic anomalies do not hold here |
| R2 | E3 with v ≤ −3 | negative |
| R2 | ensembles | E1 +1.81, E6 +2.13, E1+E2F +2.04 |
| R2 | clean basket (gaps, H 20), long-only | **+2.72** |
| R3 | **E7**, earnings reaction of either sign ≥ 3σ, bearish information | **+3.17 (H 20), +3.65 (H 40)** |
| R3 | clean basket on earnings flags | +0.2 to +2.1; H 20 gaps remains the best clean basket |
| R3 | theme ensembles | +2.25 / +2.42 |
| R4 | E7 conditioned on model 1 | calm market +3.09 against high fear +1.36; Z ≥ 2 +2.92 against Z < 2 +1.92; inverse-vol weights +3.38; thr 4: +2.77; H 60: +2.73. No gain over plain E7. |
| R5 | E8, EPS surprise | good surprises long −2.0 (negative); bad weak |
| R5 | E9, beat and drop | +2.26 (H 40) |
| R5 | E10, announcement premium | +1.74 held through the report, +0.20 if not |

**Theme.** A stock-specific news shock is followed by underperformance
against β × basket for 20–40 sessions, *whichever its sign*.
- Bad news drifts on.
- Good news reverses.

Six independently built families agree. Model 1's conditioning adds
nothing to the news rule.

**Registry statistics after Round 3** (165 trials):
- effective independent trials 11.4;
- StepM rejects E7 at H 40 and H 20 (critical t 3.06);
- PBO 0.32;
- DSR of E7 H40 0.84, short of the 0.95 required.

**Next.**
1. Recompute the registry statistics at 191 trials.
2. Run candidate diagnostics on the StepM-rejected trials: placebo,
   same-stock, intercept and plain basket, by year, leave-one-episode-out,
   top-event dependence.
3. Apply the declared selection rule.

**Declared now, before any 2026 read of v2.** If no trial passes the
whole rule, the StepM-rejected trials are still read on 2026 once, as
descriptive figures. They stay at Tier 0 and become the pre-registered
hypotheses for the prospective test.

## 2026-11-28, exploration closed; declared selection applied

**Rounds 6–7.**
- E11 no-news losers bought as reversals: t −3.5 to −3.9; faded: +3.16.
- E11 no-news winners as bearish information: +3.30.
- Segment shocks: negative.
- Model 1's q_next spike: about 0.
- The unified any-shock rule (E14): only +0.05 to +1.67, and its clean
  basket is null. The theme does not survive unification, which argues
  that its strong members are partly products of how they were cut.

**Registry.** 215 trials, 14.6 effective; StepM critical t 3.14; PBO 0.184.

**StepM rejects five trials** (`outputs/trading/v2/selection.json`):

| trial | t | Sharpe | DSR | other gates |
|---|---|---|---|---|
| E7 H 20, thr 3 | 3.17 | 2.05 | 0.70 | episode t 1.79 < 2 |
| E7 H 40, thr 3 | 3.65 | 2.13 | 0.74 | all others pass |
| E7 H 40, inverse vol | 3.38 | 2.04 | 0.70 | all others pass |
| E11 winners, H 5 | 3.30 | 2.47 | 0.87 | all others pass |
| E11F, H 10 | 3.16 | 1.82 | 0.58 | all others pass |

**Selection result: NO v2 CANDIDATE.** All five fail the deflated-Sharpe
rule (≥ 0.95). As declared above, the five are now read on 2026 once,
descriptively. They stay at Tier 0 and become pre-registered hypotheses
for the prospective test.
