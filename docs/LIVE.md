# Live forecasts and the scoreboard

`fearsignal forecast --target <session>` writes next-session forecasts from
the frozen v3 model, using only data through the latest close. The file is
written once and committed before the target session opens.
`fearsignal score-forecast --target <session>` scores it afterwards and
appends a row to `outputs/live/SCOREBOARD.md`.

## What is forecast

| forecast | where it comes from |
|---|---|
| number of stocks whose low falls ≥ 3 own-σ below the previous close | model mean = Σ q_next. The range comes from the realized counts on past days with a model mean within ±25% (the count is very over-dispersed, because one market shock hits all stocks together) |
| each stock's chance of such a drop | q_next, the model's own probability |
| next-session market fear (M_pct) and its band | past days with M_pct within ±0.30 of today's |
| chance of a return to calm within 5 sessions | the same past days |
| 5-session realized volatility of the equal-weight basket | HAR + M re-estimated recursively; the range comes from recursive out-of-sample residuals |

Every range is calibrated on 2023-07..2026-11-23, the window in which v3's
in-sample outputs exist. Returns are not forecast: the evidence says the
model has no return skill.

## Course-correction rule, fixed before any outcome is seen

* A session is **flagged** if its realized count, or its realized M_pct,
  falls outside the forecast's central 90% range.
* **One session never changes the model.** A single day is roughly one
  independent observation, because every stock trades through the same
  market.
* **Alarm:** 5 or more flagged sessions within any 20. With correct ranges
  that has a probability of about 0.04 (binomial(20, 0.1) ≥ 5). An alarm
  starts a dev-only investigation under the v3 protocol (`docs/V3.md`):
  declared candidates, selection on 2023–2025 walk-forward, and the SPA gate.
* **v3 itself cannot change.** It is hash-locked for its 400-session
  prospective read. Any fix becomes a new version, frozen with its own
  hash and judged on sessions after its own freeze.

## Relation to the pre-registered test

The scoreboard looks at prospective sessions one at a time, which is an
interim look. It cannot bias the pre-registered read:
- the read happens at exactly 400 sessions, whatever the scoreboard shows;
- the v3 spec cannot be altered;
- no test statistic of `docs/PREREGISTRATION_v3.md` is computed here.
The scoreboard reports per-session counts and Brier scores, not the
400-session statistics.
