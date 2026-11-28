# Model 2: the prospective test (pre-registration)

Written 2026-11-28, a Saturday. No session after 2026-11-27 has traded.
This is model 2's only clean test:
- v1's 2026 holdout was not pristine;
- v2's 215 trials searched dev;
- H_VOL used 2024–26 option prices.

Everything here is fixed before the sample exists.

## Sample and read

* **Sample.** Exactly the first **400 sessions after 2026-11-27**: from
  2026-11-30 to about early July 2028, depending on the holiday calendar.
  An ND1 event counts only if it is signalled at a close inside the sample
  and its 20-session window ends inside it.
* **One read**, by `fearsignal trading-prospective`.
  - The command refuses until all 400 sessions exist.
  - It refuses unless `git rev-parse HEAD:src/fearsignal` equals the tree
    declared in `config/trading_prospective.json`. That file is committed
    right after the code. The read is therefore run from a checkout of the
    declaration commit, whatever has changed in the code since.
  - A second read needs a logged reason (`outputs/trading/prospective/RUNLOG.md`).
    The start of every read is logged before anything is computed.
* **Data.** The read uses frozen v3 applied to the live panel
  (`fearsignal live` bars), plus the live trading price cache
  (`fearsignal trading-data`), refreshed to cover the sample. Nothing is
  refit.
  - Model 1 is the hash-locked v3 spec.
  - Model 2 uses ND1 exactly as selected in `config/trading_selected.json`.
  - The level factors are the dev values in
    `outputs/trading/sizing_calibration.json`.

## Hypotheses

| id | claim | statistic | claimed if |
|---|---|---|---|
| **P2-1** | ND1's bearish information persists. A stock that gaps down on its own news on a calm day trails β × the AI basket over the next 20 sessions. | The declared gate series (calendar-time short-info excess, net of 10 bps a side), with t from the stationary-bootstrap long-run variance (block 20). The per-event mean is reported with CR1 s.e. by 5-session episode. | one-sided p < 0.05 **and** a per-event mean > 0 |
| **P2-2** | Model 1 is a better risk forecaster than EWMA. | The daily mean QLIKE differential at h = 20 between frozen v3's σ̃ and EWMA(0.94), each scaled by its dev level factor, over stock-days whose window ends inside the sample. Its s.e. is the stationary bootstrap (block 20). | one-sided p < 0.05 for a negative differential |

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
  - At the dev effect (−0.107), the power is about 0.94.

## Consequences (mechanical)

* **P2-1 claimed, and IR_post > 0:** ND moves to **Tier 2**.
  - IR_post is computed on dev, 2026 and prospective events (sizing
    declaration, section 2).
  - Tier 3 still needs every ticket condition of the declaration.
* **The per-event mean or the gate-series mean is ≤ 0:** ND goes to
  **Tier 0**.
* **Otherwise:** ND stays at Tier 1.
* **P2-2 differential worse than EWMA's by more than 1 s.e.:** sizing
  switches to EWMA, matching the declared validation rule.
* **P2-2 claimed:** the risk yardstick is confirmed. Note that H_VOL found
  listed-option IV to be the better forecaster where an option exists.

## Reported only (no claim, no tier change)

* **The five v2 StepM survivors** (`outputs/trading/v2/selection.json`),
  each with its per-event diagnostics on the sample. All five failed their
  2026 read. E7 needs an earnings calendar that covers the sample; a rule
  that cannot be computed is reported as such.
* **The v1 families that stay off:**
  - SR, which failed dev;
  - RF, which was not confirmed.
* **The H_VOL straddle rule** is not supported and is not tested again.

## Live monitors during the sample (they cannot affect the read)

`fearsignal trading-live` applies the declared risk rules to closed
prospective ND1 events (`trading/prospective.py`). Both rules can only
lower ND's live tier:
* **Sequential guard** (sizing declaration, section 7).
  - After every 25 closed events, ND is suspended (Tier 0) if the running
    per-event mean is below the dev mean (+2.32%) minus 2 s.e., using CR1
    by episode.
  - A suspension stands until a review on dev data only.
* **Demotion** (conviction-tiers declaration).
  - At every 40th closed event, the last 40 are checked. A mean below
    −1 s.e. drops one tier.
  - A second demotion means Tier 0.
  - Checking non-overlapping blocks of 40 is this declaration's reading of
    "rolling 40-trade".

Live reports show these monitors and the running per-event statistics.
They are interim looks for risk control only. They do not compute P2-1 or
P2-2, and nothing seen in them can change this declaration.

## Amendments

An amendment is allowed only with a logged reason, and only before the
read. Any change to a rule is judged on sessions after the amendment's
own commit.
