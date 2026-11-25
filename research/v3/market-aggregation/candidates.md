# Market aggregation (H1b): declared candidates

Written 2026-11-24, BEFORE any candidate below was scored on dev walk-forward
or on 2026 by me. Scoring scripts are run only after this file exists.

## What was known when this was written (disclosure)

Already public in the repository before this study (not computed by me):

* `outputs/holdout/holdout_results.json` reports, for 2026, 5-session-change
  Spearman with VXN of: raw breadth B 0.617, v1-on-v2-filter M_legacy 0.575,
  v1-as-published 0.583, frozen v2 M 0.548. So raw B (the probability-scale
  aggregate) was known to have done best in 2026. Candidate C1 is therefore
  NOT blind to 2026; its 2026 number is doubly non-evidential and only dev
  walk-forward can justify it.
* `outputs/dev/dev_reference.json` (in-sample dev, not evidence): raw B 0.654
  vs frozen M 0.607 on 5-session changes.

Descriptive facts I computed before writing this (no VXN involved):

* Composition: zero stocks entered the eligible set during 2026 (all 134
  eligible from 2025Q3; 133 on the last two days because of a masked SNDK
  bar). Ten entered during the dev test blocks. The IPO-composition mechanism
  therefore cannot explain a 2026 failure; no candidate targets it.
* Aggregate saturation: 2026 breadth ranged up to 0.889 (mean 0.413); dev
  breadth reached 0.968 (2024) and 0.953 (2025). 2026 did not push the
  aggregate toward 1 beyond what dev already contained.

## Baseline and reference

* **Baseline (selected on dev):** logit breadth, `BL = logit(mean_i P_i(not calm))`,
  stitched from the cached S1 fold models, exactly as in the committed
  walk-forward (dev paired Spearman gain over legacy +0.037).
* **As-frozen reference (not a candidate):** `M = Phi^-1(F_fold(BL))` with each
  fold's own training `m_map`, i.e. the object v2 actually froze. Reported to
  see whether the dev selection evaluated the thing that was frozen.
* **Legacy reference:** `M_legacy` (expanding z-score of mean Z) on the same folds.

## Candidates (4)

All aggregates are causal, price-only and per-fold: every parameter (scales,
maps, standardisation constants) comes from the fold's training window; for
2026 they come from the frozen v2 point model and its dev period.

**C1 `breadth_lin` — raw breadth on a fixed linear yardstick.**
`M = (B_t - mu_B) / sd_B`, mu/sd of B over the training window.
Mechanism: M's two nonlinear steps (logit, then the dev percentile map
`Phi^-1 F_dev`) re-weight 5-session changes according to the level they
occur at: the percentile map stretches changes in the dense low-fear part of
the dev distribution and its fitted tails, and the logit stretches both ends.
If VXN moves roughly linearly with the expected share of afraid stocks, these
re-weightings add level-dependent noise to changes. A linear yardstick keeps
the fixed, interpretable scale v2 wanted without the re-weighting.

**C2 `sev_mean` — mean severity, fixed linear yardstick.**
`S_t = mean_i p_i,t` (p = posterior mean regime log-scale rescaled to [0,1];
weights 0 / w=0.37 / 1 on calm / stressed / panic), then `(S - mu)/sd` on the
training window. Mechanism: breadth weights stressed and panic equally, so a
move of stocks from stressed to panic (how hard, not how many) is invisible
to it. When much of the cross-section is already outside calm, fear
increments arrive through that margin. Linear, so no level re-weighting.

**C3 `meanZ_fixed` — v1's aggregate on a fixed yardstick.**
`Zbar_t = mean_i Z_i,t` (per-stock Z from the fold's training z_map), then
`(Zbar - mu)/sd` with mu/sd from the training window (frozen dev window in
2026) instead of v1's expanding window. Mechanism: tests whether v1's 2026
edge came from (a) the severity gradient plus per-stock percentile scoring,
which this keeps, or (b) the expanding yardstick adapting to 2026's higher
baseline, which this removes. (For changes, a slowly moving yardstick should
matter little; this candidate makes that testable.)

**C4 `sev_logit_mean` — mean logit severity, fixed linear yardstick.**
`mean_i logit(p_i,t)`, then training-window standardisation. Mechanism:
per-stock de-saturation. A stock's P(not calm) and severity saturate near 0
or 1, so changes of an already-stressed (or already-calm) stock vanish from
B and from mean severity; the per-stock logit keeps them. It sits between C2
(no per-stock transform) and C3 (logit plus per-stock percentile map), so
C2/C4/C3 isolate the per-stock transform.

## Scoring protocol (fixed now)

* Dev: stitched walk-forward test blocks (2024H2, 2025H1, 2025H2), 5-session
  changes within blocks (`evaluate.d5_within_blocks`), VXN dev file.
  Primary statistic: paired difference in Spearman rho(dSignal, dVXN),
  candidate minus baseline, on common changes, stationary bootstrap (mean
  block 20, 2000 resamples, `stats.corr_diff_ci` indices); s.e. = bootstrap
  s.d. of the paired difference. Also Pearson differences, per-fold
  differences, and level Spearman (block 60).
* Adoption rule: candidate - baseline Spearman gain > 1 bootstrap s.e. AND
  the mechanism check is consistent (sign agrees across folds or failure is
  explained), AND no material loss on level correlation. With 4 candidates I
  will also report a Bonferroni-style view (gain/s.e. > 2.24 for 4 one-sided
  tests at 5%); "no change" is preferred unless the evidence is clear.
* 2026: frozen v2 point model, frozen dev calibration; 5-session changes on the
  full calendar (validate._d5); reported as post-hoc diagnostics only.
