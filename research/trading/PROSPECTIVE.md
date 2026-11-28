# Model 2: the prospective test (pre-registration)

Written 2026-11-28, a Saturday. No session after 2026-11-27 has traded.
This is model 2's only clean test:
- v1's 2026 holdout was not pristine;
- v2's 215 trials searched dev;
- H_VOL used 2024–26 option prices.

Everything here is fixed before the sample exists. It was revised twice
the same day, both times before the code lock:
- after an independent review
  (`research/trading/v2/REVIEW_hvol_prospective.md`, findings PR-1 to PR-11);
- after a multi-agent adversarial verification of those fixes
  (`research/trading/v2/REVIEW_prospective_lock.md`, findings L-1 to L-16).

The claims, their thresholds and their stated power did not change in
either revision.

## Sample and read

* **Sample.** Exactly the first **400 sessions the NYSE holds after
  2026-11-27**. On the exchange's published calendar that is
  2026-11-30 .. 2028-07-03.
  - The sessions come from the exchange calendar (`trading/calendar.py`,
    the NYSE's published 2026–2028 schedule, checked against
    nyse.com/markets/hours-calendars on 2026-11-28), not from the
    vendor's dates.
  - **Unscheduled closures.** The NYSE sometimes closes on a scheduled
    day: a national day of mourning (most recently 2025-01-09) or a
    storm (2012-10-29..30). Such a day is declared at the read with
    `--closed DATE --calendar-reason TEXT`.
    - It is accepted only if neither dataset has a bar for any name that
      day, so a declared closure cannot remove a real session.
    - The sample then runs one session longer: the 400 sessions are the
      ones the exchange actually held.
    - A closure is a correction of fact to the calendar, not a change of
      rule under "Amendments". The closures and their reason are written
      to the marker, the run log and the output.
  - **Both datasets must hold exactly those sessions:** the trading price
    cache and model 1's panel. The read refuses otherwise.
  - **Coverage.** Each session must have usable bars for at least 90% of
    the 134-name universe (121 names): closes and opens in the price
    cache, and eligible names in model 1's panel. The floor is the
    universe, not the sample's own median, so a shortfall spread over the
    whole sample is caught.
  - The 400 dates and per-name coverage are written into the output.
* **What counts.** An ND1 event counts only if it is signalled at a close
  inside the sample and its 20-session window ends inside it. The last
  such signal is 2028-06-01, or one session later per declared closure.
  The output records it as `last_countable_signal`.
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
  - **Environment.** The command also refuses unless Python (major.minor)
    and numpy, pandas, scipy and pyarrow match the declared versions.
    - `--env-reason TEXT` logs a reason to read under other versions.
    - yfinance and a digest of every installed package are recorded
      only. The data refreshes need yfinance; the read does not.
  - while another read is running (an exclusive lock).
* **How to run it.** Check out the commit that added
  `config/trading_prospective.json` **in place** (`git checkout <commit>`),
  then run the command. A `git worktree` will not work, because
  `data/raw` is git-ignored.
* **Once only.** A second read needs a logged reason (`--rerun-reason`)
  whenever any trace of an earlier one exists:
  - `outputs/trading/prospective/RUNLOG.md`;
  - the git-ignored marker `data/raw/prospective_read_marker.jsonl`,
    which survives checkouts;
  - any commit on any ref that touches `outputs/trading/prospective/`.
    If git cannot answer, the read refuses.

  The start of every read is written to the marker and the run log
  before any outcome is computed. Each read is kept as its own
  `read_<time>.json`, and `read.json` is the latest.
* **Data.** Frozen v3 applied to two live datasets, both refreshed to
  cover the sample. Nothing is refit.
  - The datasets are the live panel (`fearsignal live` bars) and the
    live trading price cache (`fearsignal trading-data`).
  - Model 1 is the hash-locked v3 spec.
  - Model 2 uses ND1 exactly as selected in `config/trading_selected.json`.
  - **Purged names.** Every live pull is archived per ticker in
    `data/raw/archive/` (`src/fearsignal/archive.py`). The vendor stops
    serving a name after an acquisition, a delisting or a symbol change.
    - When it does, the name's last good series is restored from the
      archive, so it does not leave the read with its sample events.
    - The same happens if the vendor's series for a ticker is truncated,
      or its daily returns disagree with the archive's (history replaced
      under a reused symbol).
    - The output lists every name taken from the archive and why.
  - **Inputs.** Every live input is hashed before and after it is loaded
    and copied byte for byte into `data/raw/prospective_reads/<time>/`
    before any outcome is computed.
    - The inputs are the two caches, the archive, the earnings calendar
      and the firings log.
    - The hashes are in the marker and the output.
  - **Overlap rule.** When the frozen history is put on a live pull's
    price basis, only overlap bars usable in both pulls enter the
    per-ticker factors (`data.align_adjustment_basis`). A bar one pull
    has broken cannot block the alignment; the frozen pull's SNDK
    2026-11-23 is one such bar. This changes no existing panel.
  - **The $5 price floor** (ND1's name filter) reads the vendor's
    split-adjusted close in the read's data, as on dev and 2026.
    - A split after a signal can therefore move a name across the floor
      for earlier sessions.
    - The read's event set is the one its data define, by the method
      that selected ND1.

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

ND's starting tier is its tier at the end of the sample. That is the
declared tier after the live monitors, recomputed on the sample's closed
events and on every firing logged up to the sample's end. A firing logged
after the sample's end does not count.

* **IR_post ≤ 0:** ND goes to **Tier 0** (sizing declaration, section 2).
  IR_post is computed on dev, 2026 and prospective events.
* **P2-1 claimed:** ND moves to **Tier 2**.
  - **Pending review.** During the sample, the guard may have suspended
    ND, or a second demotion sent it to Tier 0. Either puts ND at Tier 0
    pending a review on dev data only (declaration, section 7 and
    DEMOTION). In that case ND stays at Tier 0, and the output records
    `ND_tier_after_review: 2` for after that review.
  - A single demotion does not block the claim. The declaration says a
    family is promoted again only through this test.
  - Tier 3 still needs every ticket condition of the declaration.
* **P2-1's per-event mean or gate-series mean is ≤ 0:** ND goes to
  **Tier 0**. This includes a sample with no closed ND1 event.
* **Otherwise:** ND keeps its starting tier. A suspension is not undone
  by the read.
* **Risk yardstick.**
  - If P2-2's differential is worse than EWMA's by more than 1 s.e.,
    sizing switches to EWMA, matching the declared validation rule.
  - If P2-2 is claimed, the risk yardstick is confirmed.
* **How it is applied.** Once `outputs/trading/prospective/read.json`
  exists under the committed declaration, `trading-live` applies its ND
  tier automatically (`live.family_status`).
  - From then on the live monitors count afresh: only ND1 events signalled
    after the sample's last countable signal, and only their own firings.
  - The read's tier already includes every in-sample firing, so those
    firings are not applied a second time.

## Pre-registered replications (outside the Holm family)

The holdout_v2 note and the v2 log said the five v2 StepM survivors would
become this test's pre-registered hypotheses. They are tested here with a
fixed, strict rule. A replication carries no tier: it only makes the rule
eligible for a further, separately declared test.

| id | rule | statistic on the sample | replicates if |
|---|---|---|---|
| R-v2 (×5) | Each StepM survivor in `outputs/trading/v2/selection.json`: E7 at H 20, at H 40, and at H 40 with inverse-vol weights; E11 winners at H 5; E11F at H 10. | The per-event mean with CR1 t by episode, plus the placebo percentile (200 draws), as in v2. The one-sided p uses t(G−1), G = episodes: with about 10–20 episodes, the normal reference would overstate significance at the 1% level. | one-sided p < 0.01 (Bonferroni over five) **and** a placebo percentile ≥ 0.95 |
| R-HVOL2 | Straddles bought when σ̃ ≥ IV, minus the rest (H_VOL2). | H_VOL2's month-bootstrap difference. | one-sided p < 0.05 |

**R-v2 data.**
- All five rules read the earnings calendar:
  - E7 is defined on reaction sessions;
  - E11 and E11F exclude them.
- The read therefore uses a separate calendar fetched at read time with
  v2's method, `data/raw/earnings_calendar_prospective.json`.
  - The method is committed in `research/trading/earnings_calendar.py`.
  - It fetches monthly windows for all 134 symbols with Robinhood's
    `get_earnings_calendar`.
  - It writes nothing unless every month is covered by every symbol.
  - It keeps the first of each (symbol, report date).
- The read uses the byte copy taken at load and records its SHA-256.
- **Coverage.** The calendar must cover the sample in its windows and in
  its data (`prospective.calendar_covers`):
  - monthly windows from three months before the sample through its last
    month;
  - at least 5 report rows in each of those months;
  - reports from at least 90% of the universe in each full quarter;
  - a fetch time after the sample's end.

  Otherwise all five are reported as **not computable**. They are never
  computed on a calendar that does not cover the sample, and a malformed
  calendar is reported the same way.
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
  - SR1 is reported by its gate statistic.
  - RF is reported as the declared RF1, by the criteria the 2026 read
    applied: A1, a variance ratio ≤ 0.85; and A2, a mean ≥ −0.5 s.e. at
    10 bps, at 25 bps and with a two-day delay. It is not re-selected on
    the sample.
* **The live monitors' record** at the end of the sample.

## Live monitors during the sample (they cannot affect the claims)

`fearsignal trading-live` applies the declared risk rules to closed
prospective ND1 events (`trading/prospective.py`). Both can only lower
ND's live tier.

**The firings log.** Every firing is appended to
`data/raw/nd_monitor_firings.jsonl` and stands from then on, even if a
later vendor revision would not reproduce it.
- The log is git-ignored, so the read's in-place checkout cannot remove
  it (like the read marker).
- Back up `data/raw` with the rest of the vendor data.
- A line the read cannot parse makes it refuse before any outcome is
  computed.
- Each firing records the event it fired on (signal date and ticker). A
  logged firing is matched to today's events by that identity, so a
  revision that adds or removes an earlier event does not turn one
  demotion into two.
- Demotions closer than 40 events apart count once.

**The standard error.** Both rules use the per-trade s.e. of the mean,
sd/√n.
- The first draft used CR1 by 5-session episode. Over a 40-trade window
  that rests on 3–12 episodes, and it is biased down and noisy.
- Run on ND1's own dev events, that draft demoted ND at event 85, on its
  selection sample.
- The per-trade s.e. keeps the dev clustering, because the operating
  characteristics below resample whole episodes. CR1 by episode is still
  reported, for the running mean only.

* **Sequential guard** (sizing declaration, section 7).
  - After every 25 closed events, ND is suspended (Tier 0) if the running
    per-event mean is below the dev mean (+2.32%) minus 2 s.e.
  - A suspension stands until a review on dev data only.
* **Demotion, rolling 40-trade** (conviction-tiers declaration).
  - After each closed event from the 40th on, ND drops one tier if the
    mean of the last 40 is below −1 s.e.
  - A further demotion needs 40 more events.
  - A second demotion means Tier 0, pending a review on dev data only.

**Operating characteristics of the monitors, stated in advance**
(`research/trading/monitor_oc.py`, `outputs/trading/monitor_oc.json`).
They are measured on dev only: ND1's 145 dev events, resampled by
episode to 110 closed events (about the sample's expected count), with
4,000 draws, starting from Tier 1.

| true per-event effect | guard suspends | at least one demotion | two demotions | Tier 0 by the end | Tier 0 under the first draft (CR1) |
|---|---|---|---|---|---|
| dev effect | 6% | 22% | 1% | 24% | 44% |
| half the dev effect | 18% | 42% | 5% | 46% | 58% |
| zero | 40% | 63% | 15% | 67% | 71% |
| -1% per event | 64% | 80% | 32% | 83% | 82% |
| -2% per event | 86% | 92% | 53% | 94% | 93% |

At the dev effect ND ends the sample at Tier 0 about a quarter of the time. That is the cost of a rolling −1 s.e. rule checked after every event, and it is a risk control's cost, not a claim's: the monitors never touch P2-1 or P2-2. When the edge is gone (zero), they reach Tier 0 two times in three; at −1% per event, five times in six.

Live reports show these monitors and the running per-event statistics.
They are interim looks for risk control only.
- They do not compute the P2-1 gate statistic or the P2-2 statistic.
- Nothing seen in them can change this declaration.

## Amendments

An amendment is allowed only with a logged reason, and only before the
read. Any change to a rule is judged on sessions after the amendment's
own commit. A declared closure is not an amendment (see "Sample and read").
