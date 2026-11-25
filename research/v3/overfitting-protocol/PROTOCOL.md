# v3 overfitting-control protocol (recommended: option P1)

Status, 2026-11-24. The 2026 holdout (225 sessions, 2026-01-02..2026-11-23) was
used once by the frozen v2 spec (SHA-256 7be905b1...). It is spent: anything
chosen by looking at it is not out-of-sample evidence. v2 stays the published
model until a v3 passes this protocol. The options considered (P0-P3) and the
tests are declared in `candidates.md` (written before any of the numbers
below were computed).

**Deviations from `candidates.md`, all driven by dev-only simulations and none
by 2026:**
(1) P1's family level was declared as raw SPA_c at 0.10. At the dev
covariance that gate falsely adopts 21% of the time (`c3_calib.json`), so the
adopted rule is the calibrated SPA_c at 5%, with Holm at family-wise 0.10
across issues.
(2) Added afterwards as checks: bootstrap-t and calibrated SPA variants,
leave-one-episode-out, the B2 power DGP, and one labelled 2026 post-hoc
diagnostic (S0 vs S1 log score).

## 0. The evidence that shapes the protocol

All numbers come from scripts in this directory. Dev = the committed walk-forward
(378 stitched test sessions, 48,862 stock-days, `outputs/dev/walkforward_daily.npz`).

1. **The one-standard-error rule alone does not control false adoption.**
   With the four declared stock candidates S2-S5 and the incumbent S1,
   resampled at the dev covariance (stationary bootstrap of the demeaned
   differentials, block 10) under the least-favourable null (all four exactly
   as good as S1), "adopt if some candidate beats the incumbent by at least one
   NW(10) s.e." adopts a change **47.5%** of the time (R = 1000;
   `c3_calib.json`). It is 33.9% when the K=2 model keeps its observed
   (bad) mean, and 9.9-11.9% even when every candidate is truly worse by its
   observed amount. Parametric Monte Carlo agrees: 36% / 47% / 54% for K = 4 / 8 / 12
   (`mc_size_power.json`).
2. **Daily loss differentials are dominated by a handful of crash days.**
   Kurtosis 15-80; the five largest days carry 57-77% of the variance of the
   S2, S3 and S5 differentials (2025-01-27 DeepSeek; 2025-04-03/04/07 tariffs)
   and 67% of S0's (2024-08-01/02/05 yen-carry unwind; 2025-04-10). Real
   evidence about candidate differences rests on two or three market episodes.
3. **So the textbook SPA_c is oversized here.** At the dev covariance,
   nominal 5% SPA_c rejects a true null 12.9% of the time with Hansen's fixed
   studentisation (least-favourable null; 6.7% in the mixed null) and 9-12%
   with the bootstrap-t (`c3_calib.json`, `c3_rules_bt.log`). In the
   parametric Monte Carlo (Student-t(5), AR(1) 0.3, cross-correlation 0.6,
   T = 380) it is 6.9 / 8.5 / 9.1% for K = 4 / 8 / 12 (MC s.e. 0.8-0.9%). That
   is mostly noise in the studentising variance: iid data with block 1 give
   5.3%, the same data with block 10 give 7.5%, and the true variance gives
   5.7% (`mc_size_diag.json`). The bootstrap-t repairs the parametric case
   (3.8-6.8%; `mc_bootstrap_t.json`) but not the crash-day case. The mechanism:
   a sample that happens to miss the one day on which a candidate lost badly
   shows a positive mean with a small variance, and no bootstrap built from that
   sample can learn the rare day exists.
4. **Calibrating the level in the bootstrap world mostly fixes it.**
   `spa.spa_calibrated` (prepivoting, i.e. a double bootstrap; the calibrated
   p is the maximum over two null worlds: least-favourable, and the SPA_c null)
   was rerun from scratch on each of 200 synthetic dev-like samples. At
   nominal 5% it rejected 5.0% under the least-favourable null and 3.5% under
   the mixed null (MC s.e. 1.5%), with 67% power for an S2-like candidate
   better by +0.003 nats/obs (`c3_prepivot2_*.json`). A one-world version
   was not enough (7.0% / 12.0%; `c3_prepivot_size.json`). On the real S2-S5
   family the implied raw level is **0.009**: a raw SPA_c p must be below about
   0.01 to count as 5%. At equal calibrated size the fixed studentisation was
   more powerful than the bootstrap-t (+0.003 nats/obs: 88% vs 70% S2-like,
   70% vs 55% S3-like, 57% vs 38% S5-like; `c3_calib.json`), so it is the
   default.
5. **Detectable effects are large relative to every v2 alternative.** With the
   calibrated 5% gate, the dev walk-forward detects a log-score gain of
   +0.002 nats/obs only 57-59% of the time and +0.003 67-89% of the time,
   for the best-measured differential (S2-like, s.e. 0.00073). For S3- or
   S5-like differentials (s.e. 0.0010-0.0012) +0.003 gives 57-70%. So 80%
   power needs about **0.003-0.004 nats/obs**. The alternatives v2 rejected
   differed from S1 by 0.0005-0.001. A v3 stock-layer candidate has to be a
   big improvement, or the dev data cannot see it.
6. **Leave-one-episode-out (LOEO) adds no control.** Adding "the gain
   survives dropping any one of the five dominant episodes" never changed a
   decision in any simulated world (`c3_loeo.json`: R5 = R3 and R6 = R1
   exactly). It stays a reported diagnostic, not a gate.
7. **Correlation-type families need the calibrated functional version too.**
   `spa_functional` on a null family of Spearman gains (five-session changes,
   n = 363, block 20) rejects 9.7% at nominal 5% (R = 300; `mc_functional.json`).
   `spa_functional_calibrated` brings it to 5.3% (raw 8.0% on the same 150
   draws; `mc_functional_cal.json`). Power for one candidate whose true
   correlation is higher by 0.08 is only 41% (raw, nominal 5%): market-layer
   gains of the size v2 found (+0.037) are not detectable on dev with
   family-wise control.

## 1. Evidence tiers

| tier | data | use |
|---|---|---|
| A, confirmatory | sessions after 2026-11-23, never loaded before the pre-registered read | the only out-of-sample evidence for v3 |
| B, selection | dev walk-forward 2023-07..2025-12, evaluate.FOLDS, per-fold parameters (`/tmp/claude-0/v3/shared/fold_models_S1.pkl` for S1) | the only data a choice may depend on |
| C, diagnostic | 2026 (spent); in-sample dev fits | mechanisms, sanity checks, labelled post-hoc numbers; never a choice |

## 2. Declaring candidates (per issue)

1. At most **4 candidates per issue**, each with: an exact, implementable
   spec; the **mechanism** (why it should help, and on which kind of day
   or fold); the incumbent; the scoring metric; the inference method. All of
   this goes in the issue's `candidates.md`, which is written, and its
   SHA-256 noted in a project ledger (`/tmp/claude-0/v3/LEDGER`: issue,
   file hash, time, family), **before** any candidate is scored.
2. A candidate may be motivated by a 2026 diagnosis. If it is, its 2026
   number carries no evidential weight and is labelled "motivating sample".
3. **Families are closed at declaration.** Every declared candidate stays in
   the SPA family even if it is abandoned, fails to converge, or looks bad.
   A new idea after scoring starts a **new family**, which is added to the
   ledger and counts in the cross-issue correction (§4).
4. Scoring metrics are fixed per layer. Stock layer: one-step predictive log
   score under the common censored, zero-inflated likelihood (daily sums,
   scaled per observation). Market layer: Spearman of five-session changes
   with VXN, taken within each fold's test block. Tail forecast: daily Brier
   score. VXN is used for validation only.

## 3. Selection on dev only

Incumbent = v2's component (S1, breadth, q from S1). For each issue's family,
build d (T x K) = incumbent loss minus candidate loss on the 378 stitched test
sessions (positive = candidate better). **A change is adopted only if all of
the following hold:**

* **(a) One-SE.** The chosen candidate beats the incumbent by at least one
  dependence-robust standard error: NW(10) on the daily differentials (as
  `evaluate.run`); for correlation statistics, the paired stationary-bootstrap
  s.e. with block 20 (as `stats.corr_diff_ci`).
* **(b) Family gate, data-snooping-robust.** The calibrated SPA_c p-value for
  the whole declared family, with the incumbent as benchmark, is below the
  issue's Holm threshold from §4 (at most 0.05):
  `spa.spa_calibrated(d, block=10, B=999, R=999)['p_c_calibrated']` for
  mean losses (fixed studentisation, SB-kernel variance);
  `spa.spa_functional_calibrated(stat_fn, n, block=20, B=499, R=499)` for
  correlation gains. The raw p-value and `alpha_nominal_05` are reported, but
  the raw p is never read at face value: for the S2-S5 family a raw p of 0.009
  corresponds to a calibrated 0.05. Block length is 10 for daily losses (the
  Politis-White automatic choice on the dev differentials ranges 1.5-10.7) and
  20 for five-session changes. Results for blocks 5/20/40 are reported.
* **(c) Mechanism check.** The per-fold signs and the LOEO table
  (`spa.episode_robustness`) are reported, and the gain has to fit the stated
  mechanism. A gain confined to one fold or one episode that the mechanism does
  not predict is a veto, decided and written down before 2026 is looked at.
* **(d) Parsimony.** If several candidates pass, take the one with the fewest
  parameters inside the 90% model confidence set (`spa.mcs`, T_max, block
  10): the family-wise version of the one-SE rule.

Otherwise: **no change**. Simulated operating characteristics of (a)+(b) at
the dev covariance for the S2-S5 family:

| rule | false adoption, least-favourable null | false adoption, mixed null | power, S2-like +0.002 | power, S2-like +0.003 |
|---|---|---|---|---|
| (a) one-SE only | 47.5-47.8% | 33.3-33.9% | 94-95% | 99% |
| (a) + raw SPA_c at 5% | 12.9% | 6.7% | 74% | 95% |
| (a) + SPA_c at calibrated raw level 0.0099 (fresh draws) | 4.0-4.5% | 0.7-1.0% | 57-59% | 88-89% |
| (a) + `spa_calibrated` rerun on each sample | 5.0% | 3.5% | n/a | 67% |

(R = 1000, or 200 for the last row; `c3_calib.json`, `c3_loeo.json`,
`c3_prepivot2_*.json`.)

## 4. Across issues

The v3 round runs m issue families in parallel (stock emission, market
aggregation, tail calibration, volatility increment, ...). Apply **Holm at
family-wise 0.10** to the m calibrated family p-values, with each threshold
capped at 0.05. The smallest p must be below 0.10/m, the next below
0.10/(m-1), and so on. With m = 4 the thresholds are 0.025, 0.033, 0.05 and
0.05. This caps the chance that the round adopts any change that is not real
at about 10%.

## 5. Reporting 2026 (spent holdout)

* Computed only **after** the dev decision for the issue is final and in the
  ledger. Computed for **every** declared candidate and the incumbent, with
  dev-only (fold-C or full-dev) parameters, never refit on 2026.
* Each number is labelled "2026, spent holdout, post-hoc diagnostic, not
  out-of-sample evidence" and printed in a separate table from selection
  numbers, next to v2's frozen 2026 numbers.
* A 2026 result may not reverse or confirm a dev decision. If it points to a
  new mechanism, that is a new family (dev evidence only) or v4 material.
* Example from part (b): on 2026, where both are out of sample, v1-as-published
  (S0) minus frozen S1 is **-0.0002 nats/obs (NW s.e. 0.0006, t = -0.33)**.
  This is a diagnostic of S0's dev "advantage", not a selection input
  (`posthoc_2026_s0_s1.json`).

## 6. Freezing v3

1. Selection is finished (§3-4) and every issue's 2026 post-hoc table is
   written up.
2. **Estimation rule (fixed now, before anything is seen):** the selected
   spec is re-estimated by the same code path on 2023-01-03..2026-11-23
   (parameters, quantile maps, climatology, HAR coefficients). This involves
   no choice, so using 2026 here is estimation, not selection. The v2 frozen
   spec is kept unchanged and runs alongside as the prospective comparator.
3. Commit `config/frozen_v3.json` and `docs/PREREGISTRATION_v3.md` (the tests
   in §7 with their numbers) and record the hash, **before** any bar after
   2026-11-23 is downloaded. Extend v2's code lock: the prospective download
   and loader refuse to run until the pre-registered read date, and refuse a
   second read without a logged reason.

## 7. Prospective holdout (sessions from 2026-11-24 on)

**One read, at a fixed sample size, no interim looks.** Interim reads are
ruled out partly because the block-20 percentile CI undercovers at small n
(coverage 0.78-0.88 at 100-150 sessions; `power_h1.json`).

### Minimum sample for the H1-type test

Test: Spearman rho of five-session changes of M_v3 and VXN, percentile
stationary-bootstrap 95% CI (block 20), pass if the lower bound > 0.30.

* Analytic: var(atanh r_s) = VIF x 1.06/(n-3). The variance-inflation
  factor for overlap plus dependence, measured as block-bootstrap variance over
  the iid formula, is **2.15 on 2026 and 2.81 on dev OOS**. For true
  rho = 0.55 and 80% power: n = 196 / 254 sessions (2026 / dev VIF). For 90%
  power: 259 / 337. Without the inflation it would be 96, so ignoring the
  overlap understates the need by a factor of 2-3.
* Simulation (300 replications per n, inner B = 1000; P(lower bound > 0.30),
  with CI coverage in brackets):

| sessions | A Gaussian AR(1), rho = 0.552 | B1 2026 resampled, rho = 0.527 | B2 dev attenuated, rho = 0.533 | B3 dev raw, rho = 0.627 |
|---|---|---|---|---|
| 200 | 0.74 (0.90) | 0.63 (0.90) | 0.68 (0.87) | 0.93 (0.89) |
| 250 | 0.85 (0.91) | 0.74 (0.89) | 0.71 (0.90) | 0.96 (0.89) |
| 300 | 0.88 (0.92) | 0.81 (0.94) | 0.77 (0.90) | 0.97 (0.88) |
| 400 | 0.94 (0.95) | 0.91 (0.92) | 0.89 (0.89) | 0.99 (0.92) |
| 500 | 0.99 (0.94) | 0.97 (0.97) | 0.93 (0.91) | 1.00 (0.94) |

* **Answer:** if rho = 0.55, about **250 sessions give 80% power** and
  about **340 give 90%**. Under the empirical dependence structures, whose
  rho came out at 0.53, 80% needs 300-340 sessions and 90% needs about 400.
* **Pre-register a single read at N = 300 sessions after 2026-11-23**
  (about **2028-02-03** on the NYSE calendar). That gives power 0.77-0.88
  across the realistic DGPs and CI coverage 0.90-0.94. If the team can wait,
  N = 400 (about 2028-06-28) gives 0.89 or more under every DGP. Do not read
  at 252 sessions (2027-11-24) just because it is one year: power there is only
  0.71-0.85.

### Other pre-registered prospective tests (read at the same N)

* **P-H1b** v3 minus v2-frozen (and minus v1), paired Spearman difference.
  Reported; "better" only if the lower bound > 0. Paired s.e. is about 0.020 at
  225 sessions (2026), about 0.017 at 300, so the minimum detectable effect
  at 80% is about 0.05. The dev gain of breadth over the legacy index (0.037)
  is **not** detectable; say so in advance.
* **P-H2** tail calibration: keep v2's criterion (BSS > 0, slope in [0.8,
  1.25]) for comparability, and publish its operating characteristics. With
  the 2026 day-clustered s.e. scaled to 300 sessions (0.118 x sqrt(225/300) =
  0.102): P(pass | slope 1.0) = 0.97, P(pass | 0.9) = 0.84,
  **P(pass | 0.8) = 0.50**, P(pass | 0.7) = 0.16. With the dev s.e. (0.19 at
  378, i.e. about 0.21 at 300) P(pass | 1.0) is only 0.71. A pass is weak
  evidence, and the 2026 pass at 0.801 was a coin flip at the boundary.
  Report the s.e. and the leave-one-day-out range alongside.
* **P-H4/H5** secondary, stated as expected to be underpowered (2026: DM-HLN
  t = 0.86; Hodrick 1B t = 0.70).
* **v2 replication.** v2-frozen H1 and H2 are re-read on the same sample, as
  an independent second test of v2.

## 8. Reusable holdout (Thresholdout, Dwork et al. 2015): considered, not adopted (P2)

Thresholdout answers a query with the training value unless the
holdout-training gap exceeds a noisy threshold T + Lap(4 sigma). In that case it
returns the holdout value + Lap(sigma) and spends one unit of a budget B.
Concretely, for the H1-type statistic at N = 300 (holdout s.e. of rho about
0.07-0.08 from the simulations; training s.e. about 0.05 on dev+2026):

* the gap has s.d. about 0.094, so T has to be about **0.19** (two s.d.) to
  stop honest gaps triggering, with **sigma = T/4 = 0.047**, Laplace s.d.
  0.067, as in the original paper's ratio (T = 0.04, sigma = 0.01 at
  n = 10,000), and **B = 2-3**.
* Every effect this project has to resolve is far below T: candidate gaps of
  0.03-0.04 in rho, and 0.0005-0.003 nats/obs in log score. For log scores the
  dev-vs-prospective gap has s.d. of about 0.0011-0.0018 nats/obs, which puts
  T at about 0.002-0.004. So Thresholdout would almost always return the
  training value and never consult the prospective data. When it did, the noise would
  widen the s.e. by about 30% (0.08 to 0.104), costing about 40% of the sample.
* Its guarantee assumes iid samples. Here the effective sample is a few
  independent stress episodes per year; blocking to 20 sessions leaves about 15
  "samples" in 300 sessions, far too few for the privacy-based bound to mean
  anything.
* The confirmatory design (one pre-registered read of a frozen spec) does not
  need reuse. Adaptive questions belong on dev, not on the prospective sample.

## 9. Why not P3 (2026 as a fourth fold)

The v3 candidate families are being formed after seeing 2026's weak spots:
the H1b reversal, the fragile H2 slope, the over-predicting top bin. Scoring
those candidates on 2026 conditions the selection on the data that suggested
them, so a fourth fold would be in-sample for exactly the failure modes being
fixed. 2026 adds 225 sessions to 378, but with its out-of-sample status gone
that is worth less than the 378 clean ones.

## 10. Part (b) results: the committed dev walk-forward

* **Does any of S2-S5 beat S1?** No. All four mean differentials are
  negative (per obs: S2 -0.00062, S3 -0.00106, S4 -0.00989, S5 -0.00050;
  SB-kernel t = -0.85, -1.10, -3.26, -0.43), so T_SPA = 0 and p_c = p_l = p_u =
  1.00 for every block (5/10/20/40), variance (SB kernel, NW10, bootstrap) and
  studentisation (fixed, bootstrap-t, prepivoted). The 90% MCS on {S1..S5}
  is {S1, S2, S3, S5} (MCS p-values 1.00, 0.78, 0.45, 0.78); S4 is eliminated
  first (p = 0.003). The most parsimonious member is S1, the same choice as the
  committed one-SE rule. Snooping is not an issue for that selection, because
  no alternative was adopted.
* **S1 vs S0 (v1 published; not out of sample).** S1 never beats S0
  (p = 1.00). S0 beats S1 by +0.00547 nats/obs (SB-kernel s.e. 0.00351,
  t = 1.56): raw one-sided p = 0.075 (B = 10,000) / 0.089 (B = 999) with fixed
  studentisation, 0.030 with the bootstrap-t; **calibrated 0.047 (fixed) / 0.020
  (bootstrap-t)**. The differential is right-skewed (skew +6), so the raw
  test is conservative here: the calibrated raw level is 0.092. Taken at face value, S0's
  look-ahead advantage is just significant. It is look-ahead: the lead is 0.0052 /
  0.0111 / 0.0005 nats/obs on 2024H2 / 2025H1 / 2025H2 (t 0.57 / 1.79 / 0.53).
  It is 85% carried by two crash episodes that S1's fold-A and fold-B training
  windows had not yet seen (dropping the five dominant episodes leaves +0.00084,
  t = 0.45). It vanishes once S1 has seen them (fold C) and when both are out of
  sample (2026 post-hoc: -0.0002, t = -0.33). No evidence that v1's published
  parameters are a better model.

## 11. Using the code (for the other v3 issues)

```python
import sys; sys.path.insert(0, "/tmp/claude-0/v3/overfitting-protocol")
import numpy as np, spa
# daily per-obs-scaled log scores on the 378 stitched dev test days
d = np.column_stack([ls_cand[k] - ls_incumbent for k in family])      # (T, K), + = candidate better
gate = spa.spa_calibrated(d, block=10, B=999, R=999, seed=0)           # ~10-60 s
gate["p_c_calibrated"], gate["p_c_raw"], gate["alpha_nominal_05"], gate["t"]
spa.mcs(-np.column_stack([ls_incumbent] + [ls_cand[k] for k in family]), alpha=0.10)
spa.episode_robustness(d[:, j])                                         # report, not a gate
# correlation gains (market layer): stat_fn(idx) -> K gains on the rows idx
spa.spa_functional_calibrated(stat_fn, n, block=20, B=499, R=499)
```

Tests: `/home/user/fear-signal/.venv/bin/python -m pytest -q -p no:cacheprovider test_spa.py`
(18 tests, about 8 s), run from this directory.
