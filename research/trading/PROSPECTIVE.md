# Model 2: the prospective test (pre-registration)

Written 2026-11-28, a Saturday. No session after 2026-11-27 has traded.
This is model 2's only clean test:
- v1's 2026 holdout was not pristine;
- v2's 215 trials searched dev;
- H_VOL used 2024–26 option prices.

Everything here is fixed before the sample exists. It was revised the
same day, before the code lock, after an independent review
(`research/trading/v2/REVIEW_hvol_prospective.md`, findings PR-1 to PR-11).

## Sample and read

* **Sample.** Exactly the first **400 NYSE sessions after 2026-11-27**:
  2026-11-30 .. 2028-07-03.
  - The sessions come from the exchange calendar (`trading/calendar.py`,
    the NYSE's published 2026–2028 schedule, checked against
    nyse.com/markets/hours-calendars on 2026-11-28), not from the
    vendor's dates.
  - The read refuses unless the price data hold exactly those sessions,
    each with bars for at least 90% of the median name count. The 400
    dates are written into the output.
* **What counts.** An ND1 event counts only if it is signalled at a close
  inside the sample and its 20-session window ends inside it. The last
  such signal is 2028-06-01.
* **One read,** by `fearsignal trading-prospective`. It refuses in each of
  these cases:
  - before all 400 sessions exist;
  - unless `git rev-parse HEAD:src/fearsignal` equals the tree declared in
    `config/trading_prospective.json`. That file is committed right after
    the code.
  - unless every committed input it reads still has its declared SHA-256:
    - `outputs/trading/sizing_calibration.json`;
    - `config/trading_selected.json`;
    - `outputs/trading/holdout/holdout_results.json`;
    - `outputs/trading/v2/selection.json`, `trials.jsonl` and
      `trial_series.parquet`;
    - `config/trading_inputs.json`;
    - `config/frozen_v3.json`.
  - unless the hash-locked raw backtest inputs match
    `config/trading_inputs.json` (`protocol.check_inputs`);
  - unless the key package versions match the declared environment, or a
    reason is logged.
* **How to run it.** Check out the commit that added
  `config/trading_prospective.json` **in place** (`git checkout <commit>`),
  then run the command. A `git worktree` will not work, because
  `data/raw` is git-ignored.
* **Once only.** A second read needs a logged reason whenever any trace
  of an earlier one exists:
  - `outputs/trading/prospective/RUNLOG.md`;
  - the git-ignored marker `data/raw/prospective_read_marker.jsonl`,
    which survives checkouts;
  - any commit on any ref that touches `outputs/trading/prospective/`.

  The start of every read is written to the marker and the run log
  before any outcome is computed.
* **Data.** Frozen v3 applied to the live panel (`fearsignal live` bars)
  and the live trading price cache (`fearsignal trading-data`), both
  refreshed to cover the sample. Nothing is refit.
  - Model 1 is the hash-locked v3 spec.
  - Model 2 uses ND1 exactly as selected in `config/trading_selected.json`.
  - The output records the SHA-256 of the live bars, the price cache and
    the earnings calendar it used.

## Hypotheses (the claims)

| id | claim | statistic | claimed if |
|---|---|---|---|
| **P2-1** | ND1's bearish information persists. A stock that gaps down on its own news on a calm day trails β × the AI basket over the next 20 sessions. | The declared gate series (calendar-time short-info excess, net of 10 bps a side), with t from the stationary-bootstrap long-run variance (block 20). The per-event mean is reported with CR1 s.e. by 5-session episode. | one-sided p < 0.05 **and** a per-event mean > 0 |
| **P2-2** | Model 1 is a better risk forecaster than EWMA. | The daily mean QLIKE differential at h = 20 between frozen v3's σ̃ and EWMA(0.94), over stock-days whose window ends inside the sample. Both are scaled by the **PIT dev level factors** in `outputs/trading/sizing_calibration.json`, which are the live sizing calibration. Its s.e. is the stationary bootstrap (block 20). | one-sided p < 0.05 for a negative differential |

**Multiplicity.** Holm over {P2-1, P2-2} at family-wise 0.10, with each
threshold capped at 0.05. So each claim needs p < 0.05.

**Operating characteristics, stated in advance.**
- **P2-1.**
  - If ND1's true effect equals its dev estimate (gate t 2.54 over 502
    sessions), the expected t over 400 sessions is about 2.3, for a power
    of about 0.73.
  - At half that effect the power is about 0.30.
  - The dev estimate is a selected maximum and an earlier study touched
    its events, so the realistic power is nearer the lower figure.
- **P2-2.**
  - At 2026's out-of-sample effect (−0.060), the expected t is about −1.8,
    for a power of about 0.57.
  - At the dev effect (−0.107), the power is about 0.94 with 2026's
    dispersion.

## Consequences (mechanical; `prospective.decide`)

ND's starting tier is its tier at the end of the sample: the declared
tier after the live monitors, recomputed on the sample's closed events
and on every logged firing.
* **IR_post ≤ 0:** ND goes to **Tier 0** (sizing declaration, section 2).
  IR_post is computed on dev, 2026 and prospective events.
* **P2-1 claimed:** ND moves to **Tier 2**.
  - If the guard suspended ND during the sample, Tier 2 waits for the
    declared review on dev data only.
  - Tier 3 still needs every ticket condition of the declaration.
* **P2-1's per-event mean or gate-series mean is ≤ 0:** ND goes to
  **Tier 0**.
* **Otherwise:** ND keeps its starting tier. A suspension is not undone
  by the read.
* **Risk yardstick.**
  - If P2-2's differential is worse than EWMA's by more than 1 s.e.,
    sizing switches to EWMA, matching the declared validation rule.
  - If P2-2 is claimed, the risk yardstick is confirmed.
* **How it is applied.** Once `outputs/trading/prospective/read.json`
  exists under the committed declaration, `trading-live` applies its ND
  tier automatically (`live.family_status`).

## Pre-registered replications (outside the Holm family)

The holdout_v2 note and the v2 log said the five v2 StepM survivors would
become this test's pre-registered hypotheses. They are tested here with a
fixed, strict rule. A replication carries no tier: it only makes the rule
eligible for a further, separately declared test.

| id | rule | statistic on the sample | replicates if |
|---|---|---|---|
| R-v2 (×5) | Each StepM survivor in `outputs/trading/v2/selection.json`: E7 at H 20, at H 40, and at H 40 with inverse-vol weights; E11 winners at H 5; E11F at H 10. | The per-event mean with CR1 t by episode, plus the placebo percentile (200 draws), as in v2. | one-sided p < 0.01 (Bonferroni over five) **and** a placebo percentile ≥ 0.95 |
| R-HVOL2 | Straddles bought when σ̃ ≥ IV, minus the rest (H_VOL2). | H_VOL2's month-bootstrap difference. | one-sided p < 0.05 |

**R-v2 data.**
- All five rules read the earnings calendar:
  - E7 is defined on reaction sessions;
  - E11 and E11F exclude them.
- The read therefore uses a separate calendar fetched at read time with
  v2's method, `data/raw/earnings_calendar_prospective.json`. Its SHA-256
  is recorded.
- That calendar must have monthly windows from three months before the
  sample through its last month. Otherwise all five are reported as
  **not computable**. They are never computed on a calendar that does not
  cover the sample.
- σ̃ (for E7's inverse-vol weights) comes from frozen v3.

**R-HVOL2** repeats `research/trading/v2/H_VOL.md`'s procedure unchanged,
including amendment 1.
- **Contracts.** The 10 names' standard monthly expiries whose entry
  session falls after 2026-11-27 and whose expiry is inside the sample:
  2027-01..2028-06, or 180 pairs.
- **Prices.** They are collected at read time from Robinhood's
  expired-contract history (`hvol.build_tasks` with those months).
- **Forecast.** σ̃ comes from frozen v3, with the PIT dev κ₂₀.
- **Scope.** It is reported only. Its code is not part of the lock.

## Reported only (no claim, no tier change)

* **SR1 and RF** on the sample (`prospective.v1_off_families`). SR failed
  dev and RF was not confirmed.
* **The live monitors' record** at the end of the sample.

## Live monitors during the sample (they cannot affect the claims)

`fearsignal trading-live` applies the declared risk rules to closed
prospective ND1 events (`trading/prospective.py`). Both can only lower
ND's live tier. Every firing is appended to
`outputs/trading/live/nd_monitor_firings.jsonl` and stands from then on,
even if a later vendor revision would not reproduce it.

* **Sequential guard** (sizing declaration, section 7).
  - After every 25 closed events, ND is suspended (Tier 0) if the running
    per-event mean is below the dev mean (+2.32%) minus 2 s.e., using CR1
    by episode.
  - A suspension stands until a review on dev data only.
* **Demotion, rolling 40-trade** (conviction-tiers declaration).
  - After each closed event from the 40th on, ND drops one tier if the
    mean of the last 40 is below −1 s.e.
  - A further demotion needs 40 more events.
  - A second demotion means Tier 0.

Live reports show these monitors and the running per-event statistics.
They are interim looks for risk control only.
- They do not compute the P2-1 gate statistic or the P2-2 statistic.
- Nothing seen in them can change this declaration.

## Amendments

An amendment is allowed only with a logged reason, and only before the
read. Any change to a rule is judged on sessions after the amendment's
own commit.
