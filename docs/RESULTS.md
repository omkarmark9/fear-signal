# Results

## 1. Development: walk-forward model comparison (2023–2025 only)

Three expanding folds. Every quantity is refit on the training window, and
only test days are scored: test halves 2024H2, 2025H1, 2025H2, totalling 378
sessions and 48,851 stock-days. Source: `outputs/dev/walkforward.json`,
produced by `fearsignal evaluate-dev`, which is deterministic.

### Stock layer: one-step predictive log score (higher is better)

| candidate | params | log score / obs | vs best (NW s.e.) | Brier skill vs climatology | vs \|x\| EWMA | vs x² EWMA | calibration slope (day-clustered s.e.) |
|---|---|---|---|---|---|---|---|
| S0 v1 published (reference) | 11 | −1.32562 | +0.00547 (0.00366) | +2.7% | +1.2% | +3.3% | 0.98 (0.18) |
| **S1 v1 refit** (selected) | 11 | **−1.33108** | best | +1.8% | +0.35% | +2.4% | 0.98 (0.19) |
| S2 shared atom, censored, K=3 | 12 | −1.33171 | −0.00062 (0.00075) | +1.7% | +0.23% | +2.3% | 0.96 (0.19) |
| S3 per-state atom, censored, K=3 | 14 | −1.33215 | −0.00106 (0.00101) | +1.6% | +0.15% | +2.2% | 0.95 (0.18) |
| S4 per-state atom, censored, K=2 | 7 | −1.34098 | −0.00989 (0.00292) | +1.6% | +0.15% | +2.2% | 0.76 (0.15) |
| S5 per-state atom, censored, K=4 | 23 | −1.33158 | −0.00050 (0.00114) | +1.7% | +0.19% | +2.3% | 0.95 (0.18) |

Reading:

* **v1's emission design survives.** Censoring the clipped values and
  modelling no-dip days are correct in principle and fit better in-sample,
  but neither improves out-of-sample prediction. Every K = 3–4 variant is
  within one standard error of the refit v1 model, and the coded
  one-standard-error rule keeps the one with the fewest parameters.
* **K = 2 is clearly too few** (−0.0099 nats/obs, 3.4 s.e.) and poorly
  calibrated.
* **The free no-dip atom is unstable.** Trained through mid-2024 its fit
  had not converged after 1,500 EM iterations (recorded as
  `converged: false`, with panic π = 0.65 at the cap). The independent review
  continued it for 10,000 more iterations and π moved to about
  (0.18, 0.01, 0.82). The later folds converge with panic π ≈ 0. Either way
  the selection is unchanged.
* **S0 is not out-of-sample.** v1's published parameters were fitted on
  2023–2025, which contains every test fold. Its lead over S1 per fold is
  0.0052, 0.0111 and 0.0005 nats/obs. It is largest on 2025H1, where the
  published parameters had already seen the DeepSeek and tariff crashes
  that S1 was forecasting blind, and it all but vanishes on 2025H2, when S1's
  training window is longest. That fits look-ahead plus small-sample noise,
  not a better model.
* **The tail forecast is calibrated; its skill edge is small.** For the
  event "tomorrow's low is ≥ 3 own-sigmas below today's close" (base rate
  4.0%), q's calibration slope is 0.98. It beats climatology by 1.8% in
  Brier score, but an EWMA of |x| mapped through the same half-normal tail
  by only 0.35%. (The x² EWMA does worse than climatology because one
  censored day contributes 100.) Much of the tail is earnings-day risk
  that past prices cannot anticipate. The forecast's value is that its
  probabilities mean what they say.

### Market layer: 5-session changes vs VXN (stitched out-of-sample)

Changes are taken within each fold's test block, so none straddles two fold
models: 363 changes in all.

| market reading | Spearman (95% CI) | Pearson (95% CI) | level Spearman (95% CI) |
|---|---|---|---|
| **breadth, S1** (selected) | **0.661** (0.55, 0.75) | **0.679** (0.56, 0.76) | 0.527 (0.23, 0.75) |
| v1 expanding z-score, S1 | 0.624 (0.52, 0.71) | 0.604 (0.51, 0.68) | 0.483 (0.18, 0.71) |
| v1 expanding z-score, v1 published | 0.619 (0.51, 0.71) | 0.596 (0.50, 0.67) | 0.509 (0.19, 0.73) |
| market-level Gaussian HMM | 0.217 (0.12, 0.33) | 0.116 (0.03, 0.20) | −0.085 (−0.29, 0.21) |

Breadth beats v1's expanding z-score on every stock model in the candidate
set. On the selected model the paired Spearman gain is **+0.037
(95% CI 0.001 to 0.066)**. That is consistent in sign across all six models
and only just significant. The structural advantages count as much: a
yardstick that never moves, no early-sample instability, and a direct
reading as "the expected share of AI stocks in a fear regime".

## 2. Holdout: the single frozen run on 2026

*Pending: added by the commit that runs the frozen specification.*
