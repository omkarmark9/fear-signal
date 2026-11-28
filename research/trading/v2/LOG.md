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

## 2026-11-28, descriptive 2026 read of the five StepM survivors: none replicates

`outputs/trading/v2/holdout_2026.json`; its start was logged in
`HOLDOUT_RUNLOG.md` before anything was computed.

| trial | dev t | 2026 t | 2026 placebo percentile | 2026 event mean (median) |
|---|---|---|---|---|
| E7 H 20 | 3.17 | 0.57 | 0.745 | +1.00% (+1.79%) |
| E7 H 40 | 3.65 | −0.07 | 0.555 | +0.70% (+1.21%) |
| E7 H 40, inverse vol | 3.38 | −0.24 | 0.500 | +0.70% |
| E11 winners H 5 | 3.30 | −0.50 | 0.330 | −0.94% |
| E11F H 10 | 3.16 | −0.38 | 0.475 | −0.75% |

**Conclusion.**
- The deflated-Sharpe gate was right to reject all five. They passed
  StepM, the placebo, the intercept test and both dev years, and still
  did not replicate.
- After 215 trials, v2 finds no return edge that survives.
- Model 1's validated skill is risk forecasting.

**Declared before running it.** The live sizing yardstick's QLIKE
advantage over EWMA (dev: −0.109 at h 5, −0.107 at h 20) is re-checked
on 2026 once. Same code, h ∈ {5, 20}, realized windows ending by
2026-11-27.

**Next.**
1. The 2026 QLIKE re-check.
2. A feasibility pilot for reconstructing historical implied volatility
   from Robinhood's expired contracts: can model 1's forecast volatility
   price options better than IV?

## 2026-11-28, the 2026 QLIKE re-check confirms the risk skill out of sample

`outputs/trading/sizing_qlike_2026.json`. It uses dev κ and dev code;
nothing was fitted on 2026.

| h | regime QLIKE | EWMA QLIKE | diff | s.e. | t | days |
|---|---|---|---|---|---|---|
| 5 | 0.4603 | 0.5059 | −0.0457 | 0.0351 | −1.30 | 223 |
| 20 | 0.2333 | 0.2934 | −0.0602 | 0.0447 | −1.35 | 208 |

The direction holds. The size is about half the dev advantage, and with
one year it is not significant on its own.

## 2026-11-28, H_VOL is pre-registered and its data collected

`research/trading/v2/H_VOL.md` asks whether model 1's forecast volatility
prices options: whether σ̃ adds information about realized vol beyond
implied vol, and whether straddles bought when σ̃ ≥ IV beat the rest.
- **Sample:** 10 split-free names × 34 monthly expiries (2024-02..2026-11),
  ATM call and put, entry 28 days before expiry.
- **Committed before any sample price was collected:**
  - d9629a1, the design;
  - ee71c7f, the operational details.

**Collection pass 1** used 7 background agents to fetch Robinhood
expired-contract daily bars. 136 of the 340 pairs came back with
placeholder entry bars, either interpolated or flat at 0.01. Those
strikes were listed after the entry day.

**Amendment 1** (f76facb) was made before any computation: use the
nearest strike that was listed at entry. Pass 2 (4 agents) applies it to
the 136 pairs.

Vendor data stays uncommitted. `config/hvol_inputs.json` locks it by
hash.

## 2026-11-28, H_VOL read: not supported (dev p 0.053); IV beats σ̃

Pass 2 found a listed-at-entry strike for 135 of the 136 placeholder
pairs. Mechanical checks of both passes found no violations, and the
inputs were locked at 61fa3ab before any computation. Each period was
then read once.

| | dev | 2026 |
|---|---|---|
| H_VOL1: σ̃'s information beyond IV | t 0.97, p 0.17 | t 0.82, p 0.21 |
| H_VOL2: straddles, σ̃ ≥ IV minus the rest | +14.5 points, p 0.053 | +27.6 points, p 0.11 |
| all straddles, mean net | +11.4% | +27.8% |

The declared rule says not supported. The full table and the post-hoc
QLIKE diagnostic are in `H_VOL.md`.
- Model 1's forecast beats EWMA but not implied volatility.
- Realized volatility ran above IV throughout 2024–26, which is a
  period effect.

**Next.**
- The model-2 prospective declaration: ND1, the risk-skill check, and the
  v2 and H_VOL hypotheses as exploratory.
- The v2 docs.
