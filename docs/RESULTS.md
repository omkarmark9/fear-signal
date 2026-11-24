# Results

## 1. Development: walk-forward model comparison (2023–2025 only)

Three expanding folds. Every quantity is refit on the training window, and
only test days are scored: test halves 2024H2, 2025H1, 2025H2 (373 sessions,
about 46,000 stock-days). Source: `outputs/dev/walkforward.json`.

### Stock layer: one-step predictive log score (higher is better)

| candidate | emission | log score / obs | vs best (NW s.e.) | Brier skill vs climatology | vs EWMA | calibration slope |
|---|---|---|---|---|---|---|
| S0 v1 published (reference) | v1, params not refit | −1.32562 | +0.00547 (0.00366) | +2.7% | +3.3% | 0.98 |
| **S1 v1 refit** (selected) | v1 | **−1.33108** | best | +1.8% | +2.4% | 0.98 |
| S2 shared atom, censored | K=3 | −1.33171 | −0.00062 (0.00075) | +1.7% | +2.3% | 0.96 |
| S3 per-state atom, censored | K=3 | −1.33216 | −0.00108 (0.00101) | +1.6% | +2.2% | 0.95 |
| S4 per-state atom, censored | K=2 | −1.34096 | −0.00988 (0.00292) | +1.6% | +2.2% | 0.75 |
| S5 per-state atom, censored | K=4 | −1.33159 | −0.00050 (0.00114) | +1.7% | +2.3% | 0.95 |

Reading:

* **v1's emission design survives.** Censoring the clipped values and
  modelling no-dip days are correct in principle and fit better in-sample,
  but neither improves out-of-sample prediction. Every K = 3–4 variant is
  within one standard error of the refit v1 model, so the rule keeps the one
  with the fewest parameters.
* **K = 2 is clearly too few** (−0.0099 nats/obs, about 3.4 s.e.).
* **S0 is not out-of-sample.** v1's published parameters were fitted on
  2023–2025, which contains every test fold. Its lead over S1 per fold is
  0.0052, 0.0111 and 0.0005 nats/obs. It is largest on 2025H1, where the
  published parameters had already seen the DeepSeek and tariff crashes
  that S1 was forecasting blind. It all but vanishes on 2025H2, when S1's
  training window is longest. That pattern fits look-ahead plus small-sample
  noise, not a better model.
* **The tail forecast is calibrated but has modest skill.** For the event
  "tomorrow's low is ≥ 3 own-sigmas below today's close" (base rate 4.0%),
  q has a logistic calibration slope of 0.98 (s.e. 0.04) and beats
  climatology by 1.8% in Brier score. The EWMA volatility baseline (λ = 0.94)
  is *worse* than climatology, so the regime filter's persistence adds real
  forecast information. Much of the tail is earnings-day risk that past
  prices cannot anticipate.

### Market layer: 5-session changes vs VXN (stitched out-of-sample)

| market reading | Spearman (95% CI) | Pearson (95% CI) | level Spearman |
|---|---|---|---|
| **breadth, S1** (selected) | **0.635** (0.52, 0.74) | **0.669** (0.56, 0.76) | 0.527 |
| v1 expanding z-score, S1 | 0.618 (0.52, 0.71) | 0.601 (0.51, 0.68) | 0.483 |
| v1 expanding z-score, v1 published | 0.614 (0.50, 0.71) | 0.593 (0.51, 0.68) | 0.509 |
| market-level Gaussian HMM | 0.212 (0.12, 0.34) | 0.114 (0.03, 0.20) | −0.085 |

Breadth beats v1's expanding z-score on every stock model in the candidate
set. On the selected model the paired Spearman gain is +0.017
(95% CI −0.045 to 0.056). It is consistent in sign but not significant on 373 days.
Its advantages are structural more than statistical: a fixed yardstick, no
early-sample instability, and a direct reading as "the expected share of AI
stocks in a fear regime".

## 2. Holdout: the single frozen run on 2026

*Pending: added by the commit that runs the frozen specification.*
