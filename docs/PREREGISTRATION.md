# Pre-registration: the 2026 holdout

Written and committed together with `config/frozen.json`, **before** any
2026 bar or 2026 VXN value was downloaded. The repository history shows the
order: this file and the frozen spec are committed first; the holdout
download, the single holdout run and its results come in later commits. The
code enforces the order too: `fearsignal download-holdout`, hashing the
holdout file, and loading it all require a committed, unmodified
`config/frozen.json`.

* Frozen spec: `config/frozen.json`, SHA-256
  `7be905b1633d28dbfef438d1d57cee0578266f674ab98b2556ca833015574ae5`
  (printed by `fearsignal freeze`; `fearsignal holdout` refuses any other).
  Model: S1, v1's emission refit on 2023–2025, s = (0.961, 1.654, 3.949).
  Market layer: breadth. Bootstrap: 200 synchronised 63-session block
  replicates (199 converged to tolerance).
* Development data: 2023-01-03 .. 2025-12-31 (`data/manifest_dev.json`).
* Holdout: every session from 2026-01-02 through the last complete session
  before the run. Filters, scales and the legacy expanding z-score run
  continuously from 2023 into 2026; only 2026 days are scored.
* Nothing is refit on 2026. Parameters, quantile maps (v2 and the v1
  reference), bootstrap replicates, climatology, HAR coefficients and the VXN
  "elevated" threshold are all read from the frozen spec.
* The holdout is run **once**. `pipeline.holdout` refuses a second run unless
  given a reason, and logs every run in `outputs/holdout/RUNLOG.md`. If a bug
  is found afterwards, the fix and the rerun are reported as such, with both
  sets of numbers.
* "v1" below means the starting point exactly as published
  (s = 0.95/1.58/3.82 and its A), with its quantile maps fitted on 2023–2025
  as the v1 description specifies.

## Hypotheses and pass criteria

Stationary-bootstrap intervals use mean block length 20 sessions (60 for
levels), 2000 resamples, and the 95% level. Five-session changes are taken on
the full session calendar.

| id | claim | statistic (2026 days) | pass if |
|---|---|---|---|
| H1 | **M measures fear** (construct validity) | Spearman ρ between 5-session changes in M and in VXN | lower 95% bound > 0.30 |
| H1b | v2 measures it at least as well as v1 | ρ(ΔM_v2, ΔVXN) − ρ(ΔM_v1, ΔVXN), paired bootstrap on common days | reported; "better" claimed only if lower bound > 0 |
| H2 | **q is a calibrated, skilful tail-risk forecast** | Brier skill vs dev climatology; logistic calibration slope (day-clustered s.e.) | BSS > 0 and slope in [0.8, 1.25]. "Significant skill" additionally needs HAC t > 2 on the daily Brier gain |
| H2b | q beats a standard volatility forecast | Brier skill vs an EWMA (λ = 0.94) of \|x\| mapped through the same half-normal tail; x² EWMA also reported | reported; "better" claimed if > 0 |
| H3 | **Z reads high when a stock is visibly hit** (sanity) | share of 3-sigma close-to-close loss days with Z ≥ 1.5 | ≥ 0.95 |
| H3b | Z does not cry wolf | share of Z ≥ 2.5 readings with no x ≥ 2 dip in the last 5 sessions | ≤ 0.05 |
| H4 | M adds to a HAR forecast of 5-day basket volatility | out-of-sample MSE gain of frozen HAR+M over frozen HAR; Diebold–Mariano with the Harvey–Leybourne–Newbold correction | "adds" claimed if gain > 0 and one-sided p < 0.05 |
| H5 | M predicts basket returns (secondary, expected to fail) | 20-day forward basket log return on M: slope for size, Hodrick (1992) 1B test for inference | claimed only if 1B p < 0.05 in 2026 **and** the dev slope has the same sign |
| H6 | alert precision (descriptive) | share of days with M ≥ 2, and with M_lo ≥ 2 (high conviction), on which VXN ≥ its dev 80th percentile | reported against the base rate |

## Known power and design limits, stated before the run

* **H3 is close to mechanical.** A 3-sigma close-to-close loss forces
  x ≥ 3 on the same day, which moves the filter strongly; on dev the share is
  1.000. It is kept because v1 reported it, and H3b is the non-trivial
  companion.
* **H2's significance clause has low power.** In-sample over all of
  2023–2025 the daily Brier gain has t ≈ 1.6, and one year of holdout has
  about 1/√3 of that information. A t > 2 is therefore unlikely even if the
  forecast is genuinely skilful. The pass criterion (positive skill and
  calibration) does not require it.
* **H5 has almost no power.** 2026 contains a handful of independent fear
  episodes. A pass would be surprising; a failure is uninformative about
  small effects.
* **H1b is underpowered.** On dev the paired gain of breadth over the legacy
  z-score was about +0.02–0.04 in Spearman ρ, with intervals spanning zero.

## Verdict mapping

* H1, H3 and H3b pass → **validated fear thermometer**.
* H2 passes in addition → **calibrated tail-risk forecaster** (stock level).
* H4 passes in addition → M carries volatility information beyond past
  realised volatility.
* Only if H5 passes is any claim made about returns.

The same statistics are also computed in-sample on 2023–2025
(`outputs/dev/dev_reference.json`) for reference; those numbers are not
evidence.
