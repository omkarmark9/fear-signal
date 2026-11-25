# Pre-registration: the v3 prospective read

Committed together with `config/frozen_v3.json` and
`config/frozen_v3_challenger_F5.json`. When this was written, no market data
dated after 2026-11-23 existed in this repository or its data cache.
`fearsignal download-prospective` refuses to run until the v3 spec is
committed, and `data.load_prospective_panel` refuses to return anything until
the full sample below exists.

* **v3 spec:** `config/frozen_v3.json`, SHA-256
  `c3804d1ffe9822ebc1d8c6408f2429cb68794a4cda38bd4c964d9a24965fd14a`.
  - S1, re-estimated on 2023-01-03..2026-11-23: s = (0.960, 1.563, 3.760), converged; 200 of 200 bootstrap replicates converged.
  - Market index M = (logit breadth − μ)/σ, with μ = −0.301 and σ = 0.849.
  - Bands, alerts and conviction are defined on M_pct, the percentile score.
* **Challenger:** `config/frozen_v3_challenger_F5.json` (family F5, persistence by shock type), fitted on the same window.
* **Comparators re-read on the same sample:** v2 frozen (`config/frozen.json`, `7be905b1…`) and v1 as published.
* **Sample:** exactly the first **400 sessions after 2026-11-23** (expected to end near 2028-06-28). There is **one read**, with no interim looks, and a second read needs a logged reason (`outputs/prospective/RUNLOG.md`).
* **Nothing is refit.** VXN is used only for validation. The VXN "elevated" threshold (24.67, the 80th percentile over 2023-07..2026-11) and the tail climatology (3.38%) are frozen.

## Hypotheses

Stationary-bootstrap intervals use mean block 20 (60 for levels), 2000
resamples and the 95% level. Five-session changes are taken on the full
session calendar.

| id | claim | statistic | pass / claim rule |
|---|---|---|---|
| P-H1 | **v3's M measures fear** | Spearman ρ, 5-session changes of M vs VXN | pass if the lower 95% bound > 0.30 |
| P-H1b | v3 vs v2-frozen and vs v1 | paired bootstrap ρ differences | reported; "better" claimed only if the lower bound > 0 (minimum detectable effect ≈ 0.05: the dev-scale gains of 0.02–0.04 are **not** detectable, as stated in advance) |
| P-H2 | **q is calibrated and skilful** | Brier skill vs frozen climatology; logistic slope (day-clustered s.e.); leave-one-day-out range; within/between-day slopes | pass if BSS > 0 and the slope is in [0.8, 1.25] |
| P-H2b | q beats an \|x\| EWMA | Brier skill | claimed if > 0 |
| P-H3 / H3b | reads high when hit; no false panic | as in v2 | ≥ 0.95 / ≤ 0.05 (H3 is near-mechanical and kept for continuity) |
| P-H4 | M adds to HAR for 5-day basket volatility | recursive expanding-window HAR vs HAR+M (rows s ≤ t−5), **Clark–West** t; frozen-coefficient DM-HLN reported as secondary | "adds" claimed if the one-sided p < 0.05; **underpowered** (about 5 years are needed for 80% power at the dev effect size) |
| P-H5 | M predicts basket returns (secondary) | 20-day slope; Hodrick 1B | claimed only if p < 0.05 **and** the sign matches dev |
| P-H6 | alert precision (descriptive) | days with M_pct ≥ 2 and with M_pct_lo ≥ 2 vs VXN ≥ 24.67 | reported against the base rate |
| P-F5 | the F5 challenger forecasts stocks better than v3's S1 | daily one-step log-score gain, NW(10) s.e. | "F5 better" claimed if the one-sided p < 0.05 **and** the challenger's market-layer ρ is not lower than v3's by more than one s.e. (the same two conditions that blocked it on dev) |
| replication | v2-frozen P-H1 and P-H2 on the same sample | as above | an independent second test of v2 |

Multiplicity: P-H4 and P-F5 are the two claim-type tests. Holm at
family-wise 0.10 with each threshold capped at 0.05 means each needs a
one-sided p < 0.05. P-H1 and P-H2 are pass/fail rules with the operating
characteristics stated below. In P-F5, the market-layer s.e. is the paired
bootstrap CI width divided by 3.92.

## Operating characteristics, stated before the read

*Filled from the dedicated simulation (see the commit that adds this
section).*

## Verdict mapping

* P-H1, P-H3 and P-H3b pass: **v3 validated as a fear thermometer** out of sample.
* P-H2 passes in addition: **calibrated tail-risk forecaster**.
* P-H4 claimed in addition: M carries volatility information beyond HAR.
* P-F5 claimed: v4 adopts the F5 stock layer for stock-level outputs, and its effect on the market layer gets its own dev-only evaluation.
* Nothing about returns unless P-H5 is claimed.
