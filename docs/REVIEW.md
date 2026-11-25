# Independent correctness review

A separate agent with a single job, checking correctness, reviewed the code
twice. It was given the specification and the repository, not the author's
conclusions, and was told to verify by independent re-implementation rather
than by reading. It could not edit the repository. Before the freeze it was
also barred from any 2026 data.

## Round 1 — before the freeze (code at `c37ec96`)

**Found correct by independent re-implementation:** the forward filter,
backward pass, posteriors, pair counts and likelihood (agreement within
1e-10 or better, across late starts, interior and trailing gaps, and every
emission variant); the EM fixed points (L-BFGS on the full likelihood gains
≤ 1.3e-11); the features (identical σ and observation types, x within
3e-14); causality of the whole signal path (bit-identical past under future
perturbation); severity, quantile maps, bands and I; tail-forecast
alignment; the HAC formulas; the adjustment-basis alignment; and freeze
serialisation (|ΔZ| ≤ 9e-5).

**Findings and resolutions** (all fixed in `0260923` before the freeze, then
the walk-forward was rerun in `7dc63bc`):

| # | severity | finding | resolution |
|---|---|---|---|
| F1 | major | H1b (v2 vs v1 on VXN) was pre-registered but never computed | paired bootstrap of v2 M against v1-as-published legacy M on common days |
| F2 | major | HAC tests on overlapping 20-day returns rejected 22% at nominal 5% | Hodrick (1992) 1B reverse regression (simulated size 5–6%); Diebold–Mariano with the HLN correction for H4 (4.7%); size tests added |
| F3 | major | one walk-forward fit hit the EM cap silently; a quoted π was not an MLE | convergence flag and warning; freeze refuses a non-converged fit; docs corrected; selection unchanged |
| F4 | major | the x² EWMA baseline was weak (one censored day contributes 100) | \|x\| EWMA added as the primary baseline; skill claims restated |
| F5 | minor | stitched 5-session changes straddled fold boundaries | changes taken inside each test block (dev gain +0.017 → +0.037) |
| F6 | minor | wrong scored sample size in the docs | corrected (378 sessions; 48,862 stock-days in the log-score comparison, 48,851 in the Brier comparison) |
| F7 | minor | holdout lock gaps (hash check, VXN loader, download, run-once) | all gated on a committed frozen spec; second run refused without a logged reason |
| F8 | minor | v1 reference maps refit at holdout time | frozen in the spec; dev file verified against its manifest |
| F9 | minor | `dropna` before `diff(5)` could stretch changes | differences on the full calendar |
| F10 | minor | steep lower-tail extrapolation of the quantile map | lower tail held flat |
| F11 | minor | bootstrap replicates under-converged | 3000-iteration cap (199 of 200 converge) |
| F12 | minor | EM starts used v1's full-sample estimates | round, theory-motivated starting values |
| F13 | minor | calibration standard errors treated stock-days as independent | clustered by day; 60-session blocks for levels |
| F14 | minor | docstring stated the nesting backwards | fixed |
| F15 | note | H3 nearly mechanical; H2's significance clause has low power | stated in the pre-registration before the run; H3b added |

## Round 2 — after the single holdout run (code at `ecd7574`)

**Found correct:** the run used the frozen code and spec unchanged
(`git diff 8714f68 1fa610a -- src tests config` is empty, hash matches), and
refit nothing. An independent filter driven only by `config/frozen.json`
reproduces `market_fear.csv` and all 109,216 `stock_fear` rows, and all 200
replicate bands, to the 4-decimal rounding. The dev/holdout adjustment-basis
alignment held on the real December-2025 overlap: identical raw prints, an
adjusted ratio constant per ticker, and first-2026 returns equal to a
single-pull computation to 1e-7. Every headline number (H1–H6) and alert
count was recomputed independently and matches the JSON. VXN's 2026 gaps are
all NYSE holidays, so no five-session change is stretched. All 11 censored
2026 days are genuine collapses.

| # | severity | finding | resolution |
|---|---|---|---|
| P1 | minor | the H2 pass (slope 0.8014 vs floor 0.8, s.e. 0.118) is fragile; "held its calibration" overstated it; table showed 0.80 | slope printed to 3 decimals; leave-one-day-out fragility (0.741–0.830, 83 of 225 single-day drops fall below 0.8) reproduced and reported; wording corrected |
| P2 | minor | H1b labelled "fail" though pre-registered as "claimed only if"; H5's same-sign clause skipped when dev was not passed | claim-type labels for H1b/H2b; dev reference now required for H5 |
| P3 | minor | partial SNDK bars on 11-20 and 11-23 (correctly masked); breadth described as a count | footnote added; breadth described as an expected share |
| P4 | trivial | F6 above cited the Brier sample size as the log-score sample size | corrected |

## Round 3 — the v3 revision (code at `07f7739`)

A fresh reviewer checked the v3 code, a v2 regression, the v3 artifacts and
every number in the v3 documents. It was interrupted by a container restart
and resumed from its transcript.

**Found correct:**
- The HMM with stacked, regime-indexed transitions agrees with an
  independent log-space implementation within 1e-12. That covers the
  log-likelihood, filtered and smoothed marginals, pair counts split by
  regime, and the EM update. The t → t+1 timing is consistent across the
  forward, backward and pair-count steps. 2-D transitions are bitwise
  unchanged.
- The shock-type label is causal and is carried by each bootstrap block's
  original dates.
- v2's frozen spec still reproduces `market_fear.csv` byte for byte, the
  stock table's content and `holdout_results.json` with no difference.
- Every v3 artifact recomputes exactly: in-sample signal history,
  estimation reference, post-hoc diagnostics, the F5 challenger and
  `compare_family` (gain 0.0024054, s.e. 0.0011113).
- Recursive HAR and Clark–West recompute exactly (t 3.8999), and the
  within/between calibration split is right.
- The P-H1 tightening's code matches the pre-registration, and its tables
  match the simulation.
- F5's declaration commit precedes its machinery and its result.

| # | severity | finding | resolution |
|---|---|---|---|
| V1 | major | a stock split between the holdout pull and the prospective pull would make `align_adjustment_basis` raise, blocking the single pre-registered read | alignment now estimates a per-ticker split factor (close and low) and dividend factor (adjusted close) and checks that each is constant on the overlap; tested with a 2:1 split plus a dividend |
| V2 | minor | a ticker with rows on both sides of the overlap but none inside it was silently left on the old basis | now an error |
| V3 | minor | the prospective read did not check the pre-registered hashes, the comparator hash, the data manifests or the universe | all are checked (`config.PREREGISTERED_SHA256`); a "started" entry is logged before any prospective bar loads, so a crash cannot hide an attempt |
| V4 | minor | P-H2's leave-one-day-out range and within/between slopes were pre-registered but not computed | `validate.tail_calibration_detail`, reported as `P_H2_detail` |
| V5 | minor | F5's signature described as "held in part", with panic said to go "as predicted"; it held in 1 of 3 folds, and panic went the predicted way in 2 of 3 | corrected in V3.md, with the review's gain-by-shock-type breakdown; P-F5 states that criterion (c) is reported, not re-tested |
| V6 | minor | "M ≥ 2" in the forward-return summary used the linear M, not the alert definition | uses `M_pct`; only those four summary values changed |
| V7 | minor | README and V3.md overstatements (both v3 changes "gated"; the 11-23 difference attributed only to re-estimation; alert days "identical"); five objects scored against a cap of four not disclosed; F5 hash missing from the ledger | corrected |
| V8 | minor | the pre-registration header no longer literally true after its amendment; the bootstrap s.d. bias misattributed; the power comment in `config.py` out of date | corrected |
| V9 | minor | two v3 outputs had no committed generator; `frozen_v3.json` predates the `persistence_by_shock` field | generators added (`fearsignal v3-reference`); reproducibility note in V3.md |
| V10 | latent | `bootstrap.parameter_table` failed on stacked transitions; `good[: t-4]` could wrap for t < 4 | fixed |
