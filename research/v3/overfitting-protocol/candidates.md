# Overfitting-control protocol for v3: declared candidates and tests

Written 2026-11-24, before any SPA, MCS, Monte Carlo or power computation was
run. The only numbers seen at this point are those already published in
docs/RESULTS.md and outputs/dev/walkforward.json (summary block), and the
2026 holdout numbers in the task brief.

## A. Protocol options (the "candidates" for this issue), max 4

Each is motivated by a mechanism, not by a number.

* **P0 - no change (v2 protocol as is).** Dev walk-forward, pairwise
  one-standard-error rule against the best candidate (Newey-West, 10 lags),
  no family-wise control; nothing further is pre-registered once the 2026
  holdout is spent. Mechanism for keeping it: the 1-SE rule already biases
  toward parsimony. Mechanism against it: with K candidates compared
  pairwise, the chance that at least one beats the incumbent by >= 1 s.e. by
  luck grows with K (each one-sided pairwise test at t > 1 has size about
  16%), and the 2026 holdout cannot be reused to catch it.
* **P1 - dev-only selection with a family gate and a prospective holdout
  (expected recommendation).** (i) At most 4 mechanism-motivated candidates
  per issue declared in writing before scoring; (ii) dev walk-forward,
  per-fold parameters; (iii) a change is adopted only if it beats the
  incumbent by >= 1 dependence-robust s.e. AND Hansen's SPA_c rejects "no
  candidate in the declared family beats the incumbent" at alpha_family =
  0.10 (incumbent = benchmark, family = every candidate declared for the
  issue, including any that were dropped); (iv) 2026 numbers reported only
  as labelled post-hoc diagnostics; (v) v3 frozen and hashed now, tests
  pre-registered on sessions after 2026-11-23, read once at a minimum sample
  set by a power analysis. Mechanism: SPA controls the family-wise
  false-adoption rate under data snooping across the declared family, and the
  consistent recentring (SPA_c) stops clearly inferior candidates (e.g. a K=2
  model) from making the test conservative; the prospective holdout is the
  only data that v3 has not seen.
* **P2 - P1 plus a reusable prospective holdout (Thresholdout, Dwork et al.
  2015).** Mechanism: lets the prospective sample be queried adaptively more
  than once with bounded overfitting, at the price of Laplace noise on every
  answer and a budget of "overfit" detections. To be judged on whether the
  noise needed is small relative to the effects the project must resolve,
  given strongly dependent daily data.
* **P3 - fold 2026 into development (a fourth walk-forward fold, 2026 as a
  test block) and select v3 on four folds.** Mechanism for: more test data,
  more stress episodes. Mechanism against: the v3 candidate set is being
  formed after seeing 2026, so any selection that scores on 2026 is
  conditioned on the same data that motivated the candidates (the
  "garden of forking paths"); its out-of-sample status is lost. Declared so
  that it is considered and rejected explicitly, not silently.

Selection among protocol options is by mechanism and by the simulated
operating characteristics in C3 below (false-adoption rate and power at the
project's own dependence structure), never by 2026 outcomes.

## B. Part (b): tests on the committed dev walk-forward log scores

Data: outputs/dev/walkforward_daily.npz, 'scored' days (expected 378),
logscore__<name> = per-day sum over eligible stocks of the one-step
predictive log density. Loss L_t = -logscore_t. Differentials
d_{k,t} = L_{bench,t} - L_{k,t} = logscore_{k,t} - logscore_{bench,t}
(positive = candidate better), divided by the mean daily number of scored
stock observations so that mean(d) is in nats/obs as in RESULTS.md.

* **T1 (primary).** H0: no member of {S2, S3, S4, S5} has a higher expected
  log score than S1. SPA_c p-value, stationary bootstrap, mean block length
  10 sessions (matches the 10 NW lags of the committed 1-SE rule), Hansen's
  stationary-bootstrap kernel variance, B = 10,000, seed 0. "Some candidate
  beats S1 after data snooping" is claimed only if p < 0.05.
  Sensitivity (reported, not used to decide): block 5, 20, 40; NW(10) and
  bootstrap variance; SPA_l and SPA_u (White's reality check).
* **T2.** S0 (v1 published, NOT out of sample: fitted on 2023-2025) against
  S1, K = 1, both directions: benchmark S1 / candidate S0, and benchmark S0 /
  candidate S1. Also per fold (2024H2, 2025H1, 2025H2). Any S0 advantage is
  an upper bound on look-ahead plus luck, not evidence for S0's parameters.
* **T3 (secondary).** Model confidence set (Hansen, Lunde and Nason 2011),
  T_max statistic, 90% level, on {S1, ..., S5}, same bootstrap settings: the
  family-wise analogue of the 1-SE rule. Most parsimonious member = the
  family-wise 1-SE choice.
* **T4.** Minimum detectable effect: power of SPA_c (alpha 0.05 and 0.10) at
  the empirical dev covariance of the differentials, when one candidate is
  shifted to be better by delta nats/obs, others at their observed means.

## C. Code checks (declared now)

* **C1 size.** T = 380, K in {4, 8, 12}; d = mu + u, u = AR(1) (phi 0.3),
  cross-correlation 0.6 between candidates, Student-t(5) innovations; 1000
  replications, B = 499. Configurations: (a) all mu = 0 (least favourable);
  (b) one mu = 0, rest at -3 s.e. (mixed null: SPA_u expected conservative,
  SPA_c near nominal). Block length 10. Nominal 5% and 10%.
* **C2 power.** As C1 with one candidate at +2 and +3 s.e. (t-scale), rest at
  0 or -3 s.e.
* **C3 decision-rule operating characteristics** at the empirical dev
  covariance (stationary bootstrap of the demeaned observed differentials,
  K = 4): false-adoption rate under H0 and power when one candidate is
  better, for (i) 1-SE rule only, (ii) 1-SE + SPA_c at 0.10, (iii) 1-SE +
  SPA_c at 0.05.

## D. Prospective-holdout power analysis (declared now)

H1-type test: Spearman rho of 5-session changes of M and VXN; percentile
stationary-bootstrap CI, block 20, 95% level, pass if lower bound > 0.30.
Find the number of sessions n at which P(pass) >= 0.80 (and 0.90) when the
true rho is 0.55. Methods: (1) analytic Fisher-z with a variance-inflation
factor for overlap and dependence; (2) simulation under three DGPs: a
parametric Gaussian AR(1) pair with the overlap built in; resampling of the
2026 daily (dM, dVXN) pairs (native rho about 0.55; using 2026 here is
design of a future test, not model selection); resampling of the dev
out-of-sample daily pairs (rho about 0.66, the optimistic case). The minimum
sample is the largest of the three.
