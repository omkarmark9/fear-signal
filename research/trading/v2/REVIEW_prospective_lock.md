# Verification before the prospective code lock (round 2)

Date: 2026-11-28. Scope: commit 65fd237, the fixes for PR-1..PR-11 and
HV-1..HV-9, before `config/trading_prospective.json` pins the code tree.

**Method.** A multi-agent workflow ran 141 agents.
- Seven finders each took one lens: locks, calendar, statistics,
  monitors, replications, docs and regressions.
- Each BLOCKER or MAJOR finding went to three independent skeptics told
  to refute it, and survived only with at least two "real" votes. Each
  MINOR finding went to one skeptic.
- A completeness critic then looked for what the lenses missed, and its
  findings were verified the same way.
- No agent changed the repository, and none ran a once-only read.

**Outcome.**
- 63 findings were confirmed. They reduce to the 17 defects below
  (L-1..L-17), all fixed or declared before the lock.
- 11 findings were refuted.
- Severity is the skeptics' consensus. The two BLOCKERs were downgraded
  to MAJOR: the code failed safe (it refused, never misread).

## Confirmed and fixed

| id | finding | fix |
|---|---|---|
| L-1 | **An unscheduled NYSE closure in the sample made the locked read refuse forever.** The sample came from a static holiday table, the vendor has no bar for a day of mourning (the repo's own cache lacks 2025-01-09), and only a change to the locked tree could fix it. The chance of at least one such closure in 19 months is roughly 25–35%. | The sample is the first 400 sessions the NYSE actually held. `--closed DATE --calendar-reason TEXT` declares a closure at the read. It is accepted only if neither dataset has a bar for any name that day, and it is logged before any outcome. Tests cover a closure, a closure with bars (refused) and a non-session (refused). |
| L-2 | **The firings log sat in a tracked folder**, so the prescribed in-place checkout would delete it once committed. Logged firings, which are meant to stand, would then vanish from the read. This is the PR-2 failure mode, fixed for the marker but not for this log. | The log moved to `data/raw/nd_monitor_firings.jsonl`, which is git-ignored like the marker. It is hashed and copied at the read. |
| L-3 | **Firings logged after the sample's end entered the read's starting tier**, against "its tier at the end of the sample". Two skeptics refuted this as following the declared rule; three confirmed it. | The read merges only firings logged with `as_of` ≤ the sample's end. |
| L-4 | **After a committed read, `trading-live` applied in-sample firings again.** A claimed P2-1 with one in-sample demotion ran at Tier 1, not Tier 2. | After the read, the live monitors count only events signalled after the sample's last countable signal, with their own firings (the log records each firing's count, `after`). `live.live_monitors` is tested. |
| L-5 | **An empty ND1 trade set crashed** `trading-live` from sample session 22 (about an 80% chance on 2026-12-30), and after the marker, the read. The engine returns a frame with no columns. | Typed empty frames in `closed_nd1_events`, `episode_stats`, `p21` and `ir_post_all`. A sample with no closed event gives Tier 0, as declared. Tests cover the empty paths. |
| L-6 | **The declared command could not log an environment reason.** The CLI had only `--rerun-reason`, and yfinance (needed by the refreshes, never by the read) was a blocking package. | Added `--env-reason`. The check is now Python (major.minor) plus numpy, pandas, scipy and pyarrow. yfinance and a de-duplicated digest of every package are recorded only. |
| L-7 | **The name-count floor was 90% of the sample's own median**, so a name missing for the whole sample was never caught. | The floor is 90% of the 134-name universe, applied to closes and opens. Per-name coverage is recorded. `trading-data` now retries tickers the batch request missed, as `fearsignal live` does. |
| L-8 | **Model 1's panel was never checked**, though it drives every ND1 signal and all of P2-2. | Its dates must equal the sample, and its eligible-name counts must meet the same floor. |
| L-9 | **A broken SNDK bar in the hash-locked holdout pull** (2026-11-23: NaN close beside a stale low) would make `data.live_panel` raise once Yahoo backfills that gap. That stops the read, `trading-live` and `fearsignal live`. | Only overlap rows usable in both pulls enter the alignment factors, by `to_panel`'s rule, and errors name the ticker and date. Every existing panel (dev+holdout, the v3 estimation panel, the live panel) is byte-identical before and after. A simulated backfill raised before and passes now. |
| L-10 | **Names the vendor stops serving drop out of the read with all their sample events**, because each refresh overwrites its cache. Acquisition targets are the likely ones, and a takeover premium is a large loss for a bearish signal, so the loss biases P2-1 toward a claim. | Each live pull is archived per ticker (`archive.py`). A ticker that is missing, truncated or whose returns disagree with the archive (history replaced under a reused symbol) is restored whole from the archive. The read lists every restored name. It had to be in place before the sample's first refresh. |
| L-11 | **Firings were keyed by event index.** A revision that adds an earlier event re-keys a demotion, which was then logged and counted twice, giving Tier 0 from Tier 2. The read and live also merged the log differently. | Each firing records the event it fired on (signal date and ticker) and is matched by that identity. Demotions closer than 40 events apart count once. One merge function serves the read and live. Tests cover a shifted index (the old code counts two) and the spacing rule. |
| L-12 | **`decide()` saw only a recomputed suspension**, so a logged-only guard firing gave "Tier 2, P2-1 claimed" with no pending review. A claim after a suspension was also recorded as Tier 2. | `nd_monitor` returns the merged `pending_review`, which is a suspension or a second demotion (declaration DEMOTION). On a claim it gives Tier 0 with `ND_tier_after_review: 2`. |
| L-13 | **The demotion's s.e. was CR1 by episode**, from 3–12 episodes per 40-trade window, and undeclared. On ND1's own dev events it demoted ND at event 85. At the dev effect it demoted in 39% and suspended in 19% of samples. | Both monitors use the per-trade s.e. of the mean, declared with operating characteristics computed from dev episodes (`research/trading/monitor_oc.py`). At the dev effect, false alarms fall to about 24% (demotion) and 6% (guard). When the edge is gone, detection is the same or better. |
| L-14 | **The read could lose or confuse its own record:** <ul><li>a malformed calendar or firings log crashed it after the marker;</li><li>two concurrent reads both ran as "first";</li><li>a failed `git log` read as "no earlier read";</li><li>a re-read overwrote the first result;</li><li>inputs were hashed only at the end;</li><li>`declaration_record` hashed an unchecked working tree.</li></ul> | <ul><li>A malformed calendar is "not computable" and a corrupt log refuses before the marker.</li><li>An exclusive lock serialises reads.</li><li>A git failure refuses.</li><li>Every read is kept as `read_<time>.json`.</li><li>Inputs are hashed before and after loading, copied, and logged in the marker.</li><li>`declaration_record` requires a clean tree, HEAD's src tree, and committed locked files.</li></ul> |
| L-15 | **The replication rules were weaker than declared:** <ul><li>`calendar_covers` trusted the file's own `windows` list, so an empty calendar passed and the survivors became computed zeros (PR-3's failure mode);</li><li>"v2's method" was not in the repository;</li><li>the episode p used a normal reference with 10–20 clusters;</li><li>RF was re-selected among RF1–RF3 on the sample.</li></ul> | <ul><li>`calendar_covers` checks the data: rows per month, names per quarter, and a fetch after the sample.</li><li>The method is committed in `research/trading/earnings_calendar.py`.</li><li>p comes from t(G−1).</li><li>RF is the declared RF1, by the 2026 read's A1/A2 (`backtest.rf_criteria`, which reproduces the 2026 read exactly).</li></ul> |
| L-16 | **Smaller defects in docs and code:** <ul><li>`docs/TRADING.md` still said no prospective criterion was declared;</li><li>LOG.md still said "IV beats σ̃", and its last entry was stale;</li><li>H_VOL.md miscounted bars (HV-6) and said neither period had exclusions;</li><li>`forecast.py`'s comment said 2023–2026;</li><li>`hvol.main` checked the code state after logging its start;</li><li>live records contained NaN tokens (invalid JSON) under pandas 3;</li><li>the P2-2 test did not call `p22`.</li></ul> | All corrected. The P2-2 test now runs the read's own branch: no refits, frozen v3's A and s, p = Φ(t). |
| L-17 | **The $5 price floor reads split-adjusted closes**, so a split after the sample can move a name across the floor for earlier sessions. | Kept and declared. Dev and 2026 applied the floor the same way, to closes from a later download, so the read uses the method that selected ND1. |

## Refuted (11)

All were judged not to be defects of the declared design:
- the declaration being re-keyable at a later commit with the same tree
  (that is the documented lock model);
- no commit-time check of the declaration (the read's refusals are all
  declared);
- the risk-yardstick rule checking only h = 20 (it is declared that way);
- placebo draws depending on pre-sample RNG use (one Monte Carlo draw of
  the named procedure);
- PR-10's open item (closed by the replication section);
- the monitors' 40-event refractory being undeclared (written before any
  data, the same in code and docs, with no effect on any outcome);
- the R-HVOL2 scope wording;
- H_VOL.md's read order;
- σ̃ "beats" EWMA in the 2026 re-check (the sentence states the point
  estimates; its significance is given beside it).

## Not checked, open

- How Yahoo and yfinance will behave in 2027–28 (schema, rate limits,
  backfills, purges) could only be simulated.
- Numerical reproducibility on another machine was not tested. BLAS and
  summation order could move a p-value near 0.05 in its last digits.
- Model 1's own prospective read (`fearsignal prospective`, sample from
  2026-11-24) uses a single download at read time. It shares the purge
  exposure of L-10 but is a separate, already-running pre-registration.
  The archive is available to it, and any use needs its own logged
  amendment.
- An unscheduled early close would not be in the half-day table, so an
  ND1 signal on such a day would not be excluded as dev's rule excludes
  half days. Unscheduled early closes are rare, and none is listed for
  2023–2026.
