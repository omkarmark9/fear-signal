# H4 (volatility increment): candidates, declared before any candidate is scored

Written after the 2026 *diagnosis* (diag2026.py / diag2026.json) and before
any dev walk-forward forecast (baseline or candidate) was computed. Nothing
below has been scored yet.

## What the 2026 diagnosis says (mechanism, not selection evidence)

* 2026 log 5-day basket vol averaged -3.750 vs -4.063 on dev (+0.313,
  HAC s.e. 0.064; annualised ~37% vs ~27%). The frozen HAR's slopes sum to
  0.57, so it mean-reverts to its dev-implied level (-4.15); its 2026 mean
  forecast was -3.909, giving a bias of +0.158 (HAC s.e. 0.046), 17% of MSE.
* The published "OOS R^2" uses the 2026 sample mean as benchmark, i.e. an
  oracle that knows 2026's level. Against the feasible benchmark (dev mean)
  HAR has OOS R^2 +0.38 and HAR+M +0.41.
* The slopes are not the problem: Mincer-Zarnowitz slope 0.98 (s.e. 0.28);
  HAC Wald test of slope change 2026 vs dev 1.37 on 3 df; 97% of the frozen
  model's excess MSE over a full 2026 refit is removed by fixing only the
  intercept.
* M's increment is not a scale problem: its MSE gain in 2026 (0.0074) equals
  the dev in-sample gain (0.0078) and comes entirely from the variance term
  (bias term 3e-5). 2026 M stays inside the dev range (no extrapolation, no
  floor hits). Partial correlation of M with the HAR residual: 0.23 dev,
  0.25 in 2026. The DM-HLN t falls from 1.63 (n=620, in-sample) to 0.86
  (n=220), about what the sqrt(n) scaling predicts (0.97): low power, not a
  changed relation.

## Fixed design (applies to the baseline and every candidate; fixed now)

* Target y_t = 0.5 ln(mean_{u=t+1..t+5} R_u^2) for the equal-weight basket
  return R (validate.har_design); regressors HAR lags (1, 5, 22) of
  0.5 ln(mean R^2). Loss: squared error of y. Horizon h = 5.
* Dev panel only (data.load_dev_panel). Folds = evaluate.FOLDS. Market
  regressors for fold f come from the shared cache's fold-f model and maps
  (M = fold m_map(logit breadth); BL = logit breadth), computed over the whole
  dev span with fold-f parameters (in-sample on its training window, genuinely
  out-of-sample on its test half).
* Training rows for fold f: eligible days with finite y and X and
  t + 5 <= index(train_end), so no training target uses test-period returns.
* Test rows: the fold's test half, eligible days, finite y and X. The three
  test halves are stitched (2024H2, 2025H1, 2025H2).
* Inference: DM with the HLN correction (stats.diebold_mariano, horizon 5) on
  the stitched loss differential; secondary check: stationary-bootstrap
  (mean block 20, 2000 resamples) 95% interval for the mean differential.
  Per-fold gains are also reported.
* OOS R^2 is reported against two benchmarks: the test-period mean (oracle,
  as published) and the training-period mean of y (feasible; Campbell-Thompson).
  This is a reporting correction and changes no forecast.

## Baseline

B0: per-fold fixed-coefficient OLS of HAR and HAR+M on the training window
(the committed H4 design, applied fold by fold).

## Candidates (3; mechanism first)

C1 recursive: expanding-window re-estimation of HAR and HAR+M every test day
   t on all rows s with s + 5 <= t (only targets already observed). Same
   regressors as B0. Mechanism: the level and the coefficients drift; daily
   re-estimation lets the intercept (implied unconditional level) absorb new
   data. Standard practice in forecast evaluation.

C2 level-free: per-fold fixed coefficients, slopes on the three HAR lags
   constrained to sum to 1 (OLS of y - RV22 on const, RV1 - RV22, RV5 - RV22,
   [+ M]). Mechanism: the 2026 failure was mean reversion to the
   training-period level; a unit-root-in-level HAR has no fixed anchor and is
   robust to location shifts (Clements-Hendry), at the cost of losing genuine
   mean reversion when the level is stable.

C3 logit breadth: per-fold fixed HAR + BL (logit breadth), replacing M.
   Mechanism: M = Phi^-1(F_dev(BL)) is a rank-normal transform. In the dev
   in-sample HAR residual the relation to M is flat over the bottom four
   quintiles and jumps in the top one (convex), so the rank transform
   compresses the informative upper tail; log vol plausibly responds to the
   log-odds of the stressed share. (Motivated by a dev in-sample residual
   plot, not by 2026.)

## Pre-declared decision rule

* Primary comparison for every candidate: its HAR+X forecast against B0's
  HAR+M forecast, d_t = e_B0^2 - e_cand^2 on the stitched dev test days.
* Adopt only if (i) the DM-HLN t on d is > 1 (gain beats B0 by at least one
  dependence-robust s.e.), (ii) the mean gain is positive in at least 2 of 3
  folds, and (iii) the mechanism check holds (C1/C2: the gain comes mostly
  from the bias term; C3: the gain comes from the variance term).
* If several qualify, adopt the one with the largest mean gain; otherwise
  "no change".
* Reported in addition, for each design: the M (or BL) increment over the
  same-design HAR (DM-HLN), i.e. the dev out-of-sample version of H4.
* 2026 numbers for every candidate are post-hoc diagnostics only.
