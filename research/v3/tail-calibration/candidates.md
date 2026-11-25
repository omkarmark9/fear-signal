# Tail-forecast calibration (H2): declared candidates

Written before any candidate was scored on dev, after the 2026 diagnosis
(diag1.py, diag2.py, diag3.py). Selection uses dev walk-forward (2024H2,
2025H1, 2025H2 test halves; per-fold S1 parameters from
/tmp/claude-0/v3/shared/fold_models_S1.pkl) only. 2026 numbers are post-hoc
diagnostics and are never used for selection.

## What the diagnosis found (2026 used only to diagnose)

* The 2026 over-prediction sits in the top bins: q >= 0.075 gave 97 events
  against 128 expected over 169 days. These stock-days come in bursts on the
  days after market-wide shock days (>= 20% of stocks with x >= 3). After 5
  of the 7 shock days in 2026 the next session had an event share of 0.8% or
  less. Examples: 06-08 (63 high-q stocks, 0 events) and 11-06 (98 high-q
  stocks, 0 events).
* On dev OOS the same post-shock days were UNDER-predicted: q 6.0% against
  15.9% observed on the day after a shock, because 2024-08 and 2025-04 had
  multi-day crash runs. In 2026 the predictive panic component's implied
  tail rate was 0.25 (s.e. 0.11) against the model's 0.36. On dev OOS it was
  0.53 (s.e. 0.28).
* The per-fold dev OOS slopes are heterogeneous: 1.31 (0.64), 1.35 (0.30),
  0.65 (0.16). The pooled 0.98 is an average of under-dispersed early folds
  and an over-dispersed latest fold. The latest fold is the one closest to
  the frozen model.
* The forecast level was fine in 2026 (mean q 3.80% vs 3.65% observed). The
  21-day/126-day scale ratio (1.00-1.15 by quarter) was inside the dev
  range, so the slow-sigma mechanism is not the driver. The excess persistence
  in 2026 (q above observed at 3-5 days after a shock) has the opposite sign
  on dev, so the transition matrix is not systematically over-persistent.

## Candidates (at most 4; mechanism first)

**C0. No change.** q = sum_k (alpha_t A)_k (1 - z0) 2 Phi(-3 / s_k) with the
fold's S1 parameters. Mechanism argument: the miscalibration is market-level
sampling variation. Each period has a handful of post-shock episodes, and
their continuation-or-rebound outcome is one common draw. No stock-level
structural fix is identified.

**C1. Walk-forward logistic recalibration (stacking).**
q' = sigmoid(a + b logit q), with (a, b) fitted by maximum likelihood ONLY on
earlier folds' out-of-sample forecasts. Fold A has no earlier OOS data, so it
gets the identity (q' = q). Fold B uses fold A's OOS forecasts, and fold C
uses those of A and B. Mechanism: a global dispersion and level error of the
parametric tail (shared over states), estimated on genuinely out-of-sample
forecasts. The frozen version would use (a, b) fitted on all three dev OOS
folds.

**C2. Training-window state tail probabilities (state-level recalibration).**
Replace the parametric state tail vector theta_k = (1 - z0) 2 Phi(-3/s_k)
with a non-negative least-squares fit of y_t on the one-step predictive law
pred_t = alpha_{t-1} A. The fit uses the fold's own filter and parameters
over that fold's training stock-days only (t <= train_end, with the same
eligibility as scoring), and q' = pred_t . theta. Mechanism: the half-normal
tail and the shared no-dip atom can misstate P(x >= 3 | state), in panic
especially (panic pi ~ 0 under the free-atom model; the half-normal's deep
tail at x >= 6 is over-predicted in high-panic states). Errors in A's
persistence also project onto theta. K = 3 parameters per fold, no tuning.
The frozen version would fit theta on all dev stock-days with the frozen
model.

No fourth candidate. The other mechanisms on the list (slow sigma, an
over-persistent A) were not supported by the diagnosis, and a post-shock
"market rebound" correction would be chosen from 2026's sign, which dev
contradicts.

## Scoring rule (fixed now)

* Metrics on all dev OOS test stock-days with finite q, y and |x|-EWMA
  (48,851 stock-days, 378 days): Brier score, log loss, and logistic
  calibration slope/intercept with day-clustered s.e.
* Paired comparison with C0: the stock-day loss difference is averaged within
  each day, then the mean across days gets a Newey-West (5 lags) s.e. This is
  reported pooled over A+B+C and over B+C (C1 is identical to C0 on A).
* Adopt only if a candidate beats C0 by at least 1 s.e. on BOTH the Brier and
  the log-loss daily gain (pooled A+B+C), does not worsen calibration, and
  its mechanism is understood. Otherwise keep C0 (no change).

---

## ADDENDUM (written after C0-C2 were scored, before C3 was scored)

**Provenance, stated plainly.** C0-C2 have been scored (score.py). No
declared candidate passed. I then ran a between-day/within-day split of the
calibration slope (diag4.py) on dev OOS and on 2026 at the same time, and an
idiosyncratic-versus-market split (diag5.py):

* Within-day slope (each stock's logit q minus the day's mean): dev OOS 0.50
  (0.07). By fold: 0.31 (0.20), 0.74 (0.12), 0.43 (0.11). 2026: 0.73 (0.12).
* Between-day slope (the day's mean logit q): dev OOS 1.43 (0.32). By fold:
  2.53, 1.84, 0.95. 2026: 0.95 (0.36).
* A stock hit at t-1 while < 10% of stocks were hit was over-predicted:
  observed/forecast 0.67 on dev (t = -2.1, 228 days) and 0.74 in 2026
  (t = -1.5, 138 days).

These dev-OOS numbers were seen before C3 was declared, so **C3's dev
evidence is not clean**. Under the protocol (declare before scoring any),
**C3 cannot be adopted in this round, whatever its score.** If it beats C0
by at least 1 s.e. on both the Brier and the log-loss daily gain (pooled
A+B+C) and does not worsen calibration, it is recommended as the
pre-registered candidate for the next prospective test. Otherwise it is
dropped. This is the 4th and last slot.

**C3. Mean-preserving within-day dispersion shrinkage.** For each scored day
t, take the stocks with a forecast (made at the close of t-1),
L_it = logit q_it, and Lbar_t = their mean. Then
q'_it = sigmoid(c_t + Lbar_t + b (L_it - Lbar_t)), where c_t is solved so
that mean_i q'_it = mean_i q_it. The market-level forecast is left exactly as
the model makes it. The single parameter b is the maximum-likelihood
estimate on the fold's TRAINING-window stock-days with the fold's own filter
and parameters (never the test half). b = 1 means no change. Mechanism: one
transition matrix shared by all stocks gives a stock's one-day idiosyncratic
shock (e.g. earnings) the persistence learned mostly from market-wide
episodes, so the stocks that stand out on a day are over-predicted. Shrinking
each stock's deviation from the day's cross-sectional mean addresses that and
leaves the between-day (market-timing) component untouched. Inputs are the
universe's prices only, and the forecast is causal. The frozen version would
fit b on all dev stock-days with the frozen model.
