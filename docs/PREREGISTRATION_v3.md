# Pre-registration: the v3 prospective read

First committed together with `config/frozen_v3.json` and
`config/frozen_v3_challenger_F5.json` (f8a698a). Two later commits amended
it, both before any market data dated after 2026-11-23 existed in this
repository or its data cache:
- 07f7739 added the operating characteristics and tightened P-H1;
- the commit applying the v3 review added these clarifications.
`fearsignal download-prospective` refuses to run until the v3 spec is
committed, and `data.load_prospective_panel` refuses to return anything until
the full sample below exists.

* **v3 spec:** `config/frozen_v3.json`, SHA-256
  `c3804d1ffe9822ebc1d8c6408f2429cb68794a4cda38bd4c964d9a24965fd14a`.
  - S1, re-estimated on 2023-01-03..2026-11-23: s = (0.960, 1.563, 3.760), converged; 200 of 200 bootstrap replicates converged.
  - Market index M = (logit breadth − μ)/σ, with μ = −0.301 and σ = 0.849.
  - Bands, alerts and conviction are defined on M_pct, the percentile score.
* **Challenger:** `config/frozen_v3_challenger_F5.json`, SHA-256
  `126c32c800066f727f9ea98469b42c1ebd518129e00f63df5cf641290b52c8cb`
  (family F5, persistence by shock type), fitted on the same window.
* **Comparators re-read on the same sample:** v2 frozen (`config/frozen.json`, `7be905b1…`) and v1 as published.
* **Sample:** exactly the first **400 sessions after 2026-11-23** (expected to end near 2028-06-28). There is **one read**, with no interim looks, and a second read needs a logged reason (`outputs/prospective/RUNLOG.md`).
* **Nothing is refit.** VXN is used only for validation. The VXN "elevated" threshold (24.67, the 80th percentile over 2023-07..2026-11) and the tail climatology (3.38%) are frozen.

**Live monitoring is not the prospective test.** `fearsignal live` applies
the frozen v3 spec to recent bars for display. Its bars go to a separate,
git-ignored cache (`data/raw/live.parquet`) with no manifest. It reads no
VXN, computes none of the statistics below, and cannot alter the
hash-locked spec. The prospective read uses its own pull
(`fearsignal download-prospective`).

**Live forecast scoreboard (added 2026-11-25, before any of its forecasts
was scored).** `docs/LIVE.md` scores next-session forecasts one session at a
time. That is an interim look at prospective sessions. It cannot affect the
read: N is fixed at 400, v3 is hash-locked, and any course correction
becomes a separate version judged on sessions after its own freeze. It
computes none of the statistics above.

## Hypotheses

Stationary-bootstrap intervals use mean block 20 (60 for levels), 2000
resamples and the 95% level. Five-session changes are taken on the full
session calendar.

| id | claim | statistic | pass / claim rule |
|---|---|---|---|
| P-H1 | **v3's M measures fear** | Spearman ρ, 5-session changes of M vs VXN; stationary-bootstrap percentile intervals at 97.5% and 95% | pass if the **lower bound of the 97.5% interval** (its 1.25th percentile) > 0.30; the 95% interval is reported |
| P-H1b | v3 vs v2-frozen and vs v1 | paired bootstrap ρ differences | reported; "better" claimed only if the lower bound > 0 (minimum detectable effect ≈ 0.05: the dev-scale gains of 0.02–0.04 are **not** detectable, as stated in advance) |
| P-H2 | **q is calibrated and skilful** | Brier skill vs frozen climatology; logistic slope (day-clustered s.e.); leave-one-day-out range; within/between-day slopes | pass if BSS > 0 and the slope is in [0.8, 1.25] |
| P-H2b | q beats an \|x\| EWMA | Brier skill | claimed if > 0 |
| P-H3 / H3b | reads high when hit; no false panic | as in v2 | ≥ 0.95 / ≤ 0.05 (H3 is near-mechanical and kept for continuity) |
| P-H4 | M adds to HAR for 5-day basket volatility | recursive expanding-window HAR vs HAR+M (rows s ≤ t−5), **Clark–West** t; frozen-coefficient DM-HLN reported as secondary | "adds" claimed if the one-sided p < 0.05; **underpowered** (about 5 years are needed for 80% power at the dev effect size) |
| P-H5 | M predicts basket returns (secondary) | 20-day slope; Hodrick 1B | claimed only if p < 0.05 **and** the sign matches dev |
| P-H6 | alert precision (descriptive) | days with M_pct ≥ 2 and with M_pct_lo ≥ 2 vs VXN ≥ 24.67 | reported against the base rate |
| P-F5 | the F5 challenger forecasts stocks better than v3's S1 | daily one-step log-score gain, NW(10) s.e. | "F5 better" claimed if the one-sided p < 0.05 **and** the challenger's market-layer ρ is not lower than v3's by more than one s.e. (these test dev criteria (a)/(b) and (d) again. Criterion (c), the mechanism signature, is a property of the frozen parameters, not of the prospective sample. It held in only 1 of 3 dev folds, and it holds in the frozen full-window fit: stressed self-transition 0.851 after idiosyncratic days vs 0.976 after systemic days; panic 0.128 vs 0.475. It is reported, not re-tested) |
| replication | v2-frozen P-H1 and P-H2 on the same sample | as above | an independent second test of v2 |

Multiplicity: P-H4 and P-F5 are the two claim-type tests. Holm at
family-wise 0.10 with each threshold capped at 0.05 means each needs a
one-sided p < 0.05. P-H1 and P-H2 are pass/fail rules with the operating
characteristics stated below. In P-F5, the market-layer s.e. is the paired
bootstrap CI width divided by 3.92.

## Operating characteristics, stated before the read

Simulated before any prospective data existed
(`research/v3/operating-characteristics/`). There were three
data-generating processes: a Gaussian AR(1) pair with the dev
autocorrelations, and the 2026 and dev-OOS daily change pairs resampled by
stationary bootstrap and tuned to the target correlation. The intervals use
mean block 20 and 2000 resamples, as in the code.

**P-H1, why the pass rule is the 97.5% interval.** With the originally
drafted rule (95% interval, lower bound > 0.30) and a true ρ of exactly
0.30 at N = 400, the false-pass rate is 4.0% (Gaussian), 2.2% (2026
structure) and 5.0% (dev structure), against a nominal 2.5% (MC s.e.
0.2–0.35 pp, 4000 replications). The block bootstrap's s.d. is 8–9% below
the true sampling s.d. under the Gaussian and dev-structure processes, and
6% above it under the 2026 structure. Interval coverage is 91–96%. The rule was therefore tightened
before any data existed: with the 97.5% interval the boundary false-pass
rate is 2.4% / 1.1% / 3.0%.

| true ρ (N = 400) | P(pass), 97.5% rule: Gaussian / 2026 / dev | 95% rule, for reference |
|---|---|---|
| 0.30 (boundary) | 0.024 / 0.011 / 0.030 | 0.040 / 0.022 / 0.050 |
| 0.45 | 0.49 / 0.54 / 0.39 | 0.59 / 0.63 / 0.49 |
| 0.55 | 0.90 / 0.95 / 0.83 | 0.94 / 0.97 / 0.89 |
| 0.60 | 0.98 / 0.995 / 0.97 | 0.99 / 0.998 / 0.98 |

A pass is evidence that ρ > 0.30 at a one-sided error rate of at most
about 3%. Power is above 0.8 for the ρ ≈ 0.55 seen on dev and in 2026.

**P-H2.** These are upper bounds on P(pass), because Brier skill > 0 is also
required. Under a normal approximation for the slope estimate:

| s.e. assumption at N = 400 | true slope 1.0 | 0.9 | 0.8 | 0.7 |
|---|---|---|---|---|
| 2026 day-clustered s.e. scaled (0.089) | 0.99 | 0.87 | 0.50 | 0.13 |
| dev day-clustered s.e. scaled (0.183) | 0.78 | 0.68 | 0.49 | 0.29 |

A true slope at the 0.8 edge passes half the time, so a pass does not
establish a slope ≥ 0.8. With the dev-sized s.e., even a perfectly
calibrated forecast fails about 22% of the time. The slope, its s.e., the
leave-one-day-out range, and the within-day and between-day slopes are
reported with the verdict.

**P-H1b, P-H4, P-H5 and P-F5** are underpowered by design (minimum
detectable P-H1b difference ≈ 0.05; P-H4 needs about five years; P-H5 sees a
handful of episodes). A null result there is not evidence of no effect.

## Verdict mapping

* P-H1, P-H3 and P-H3b pass: **v3 validated as a fear thermometer** out of sample.
* P-H2 passes in addition: **calibrated tail-risk forecaster**.
* P-H4 claimed in addition: M carries volatility information beyond HAR.
* P-F5 claimed: v4 adopts the F5 stock layer for stock-level outputs, and its effect on the market layer gets its own dev-only evaluation.
* Nothing about returns unless P-H5 is claimed.
