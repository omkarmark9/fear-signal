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
