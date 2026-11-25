# Family F5: shock-type-dependent persistence (stock layer)

Declared before any scoring. This is a new family under the v3 protocol
(research/v3/overfitting-protocol/PROTOCOL.md §2.3) and counts in the
cross-issue Holm correction (m = 5).

## Motivation (dev evidence; 2026 only corroborates)
The tail-calibration diagnosis (research/v3/tail-calibration/) found:
- The tail forecast's *within-day* calibration slope is 0.50 (s.e. 0.07) on dev out-of-sample. It is below 1 in every fold, z = -7.1. The spread of forecasts across stocks on a given day is too wide.
- After stock-specific shocks the model over-forecasts. These are stocks hit at x >= 3 on a day when fewer than 10% of stocks were hit, such as earnings. Observed/forecast is 0.67 on dev (t = -2.1).

## Mechanism
S1 has one transition matrix A for all stocks and all days. Its stress persistence is learned mostly from market-wide episodes, which persist. A one-day stock-specific shock is filtered into the same persistent stressed/panic regime, so the next-day risk is overstated for that stock.

## The single candidate (K = 1)
**S1-PT (persistence by shock type).** S1 unchanged except that the transition from t to t+1 uses A^(m_t), where

    m_t = 1 ("systemic")   if  share of eligible stocks with x_t >= 2  is  >= 0.10
    m_t = 0 ("idiosyncratic") otherwise

m_t is observable at the close of t, so the filter stays causal. EM: the M-step sets A^(m) = row-normalised sum of xi_t over the days with m_t = m. Everything else (emission, init, severity, maps) is as in S1. This adds 6 transition parameters, 17 against S1's 11. The thresholds (x >= 2, 10%) are fixed here and are not searched: x >= 2 is a two-own-sigma dip, and 10% is the cut the diagnosis used.

Predicted signature if the mechanism is right: A^(0) has lower stressed and panic self-persistence than A^(1); the within-day calibration slope moves toward 1; the gain is concentrated on days after idiosyncratic shocks.

## Evaluation (dev only; evaluate.FOLDS, refit per fold, same seeds)
- Primary: one-step predictive log score per observation under the common likelihood, as a daily differential vs S1 on the 378 stitched test sessions.
- Adopt only if all of the following hold:
  - (a) the gain is >= 1 NW(10) s.e.;
  - (b) the calibrated SPA_c p (K = 1, block 10, B = R = 999) is below the Holm threshold for this family;
  - (c) the gain has the predicted signature: A^(0) less persistent, and positive in at least 2 of the 3 folds;
  - (d) no material loss in the market layer: the 5-session-change Spearman of logit breadth with VXN stays within one s.e. of S1's.
- Secondary (reported, not gating): Brier, overall and within-day calibration slopes.
- 2026: post-hoc only, after the decision.
