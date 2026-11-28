# Review: H_VOL and the Model-2 prospective test

Independent correctness review, 2026-11-28.
- **Reviewed at:** HEAD `5164c14`. The code is as of `4a1453d`; `d901138` and `5164c14` change docs only.
- **Scope:**
  - H_VOL: `H_VOL.md`, `trading/hvol.py`, `tests/test_hvol.py` and the H_VOL outputs.
  - The prospective test: `PROSPECTIVE.md` and `trading/prospective.py`. Also the monitor hook in `live.main()`, `sizing.qlike_oos`, the `trading-prospective` CLI, `tests/test_prospective.py`, and the declared rules in `declaration.json`.
- **What I changed:** nothing in the repo except this file. I did not call `hvol.main()`.
- **Scratch scripts:** in `/tmp/claude-0/-home-user-fear-signal/746ee66c-3d9a-5a30-89ca-deab40db82a0/scratchpad/review/`, listed at the end.

## Verdict

**H_VOL.**
- The computation is correct. Every headline number reproduces exactly from the raw files with independent code.
- "Not supported" stands.
- The problems are in the write-up, which over-reads what the results show:
  - the post-hoc finding that IV beats σ̃ rests on t 1.5 (dev) and t 0.4 (2026);
  - two "what it means" sentences contradict the results table.

**Prospective.**
- P2-1 and P2-2 compute what the doc declares. P2-1 reproduces the recorded v1 dev and 2026 reads exactly. There is no blocker in the statistics.
- The lock is weaker than the doc says:
  - only the `src/fearsignal` tree is pinned;
  - the read's inputs and environment are never verified;
  - the once-only guard can be bypassed by the checkout procedure the doc itself prescribes.
- The exploratory section will silently report wrong numbers for the earnings-based survivors.
- The live monitors stop working in September 2027.

Fix PR-1 to PR-4 before `config/trading_prospective.json` is written.

## Summary

| ID | Severity | Finding |
|---|---|---|
| PR-1 | MAJOR | The lock pins only the `src/fearsignal` tree. The read silently depends on mutable inputs, git-ignored data and an unpinned environment, and it accepts any commit with the same tree. |
| PR-2 | MAJOR | The once-only guard is a working-tree file. Checking out the declaration commit deletes it once the read's outputs are committed, so a second read needs no reason. |
| PR-3 | MAJOR | `exploratory()`: the earnings calendar ends 2026-11, so all five survivors are miscomputed on the sample without any notice. E7 shows 0 and "fewer than 5 events"; E11 and E11F become different rules. |
| PR-4 | MAJOR | `trading-live`, which runs the declared monitors, fails from the 2027-09-09 close (session 195 of 400). The holiday and half-day tables end 2027-12-31. |
| HV-1 | MAJOR | A post-hoc, non-significant QLIKE comparison is stated as a finding: "IV beats σ̃", "loses to listed-option IV" and "IV is the better risk yardstick for sizing". |
| PR-5 | MINOR | `decide()` ignores a live suspension or demotion. It omits the declared "IR_post ≤ 0 → Tier 0" rule, and nothing applies the read's decision. |
| PR-6 | MINOR | The 400 sessions are counted from the union of vendor dates, with no exchange-calendar check. |
| PR-7 | MINOR | SR and RF are listed as "reported only", but `read()` does not compute them. |
| PR-8 | MINOR | Monitors: "rolling 40-trade" is reinterpreted as non-overlapping blocks. Suspensions are not persisted. "They do not compute P2-1" is not quite true. |
| PR-9 | MINOR | No tests for the `read()` refusals, `p22`, `ir_post_all` or the live hook. |
| PR-10 | MINOR | The holdout_v2 docstring and the LOG still say the survivors become pre-registered hypotheses (already logged as an open item). |
| PR-11 | MINOR | Wording of "its dev level factor" for frozen v3. N_eff counts signal weeks, not entry weeks (both numerically negligible). |
| HV-2 | MINOR | p = 0.053 sits on a Monte Carlo knife edge (converged 0.054; 18% of seeds pass). H_VOL2 is also very sensitive to the κ level. |
| HV-3 | MINOR | "Not something model 1 predicted" contradicts the table. κ is fitted in-sample on dev. |
| HV-4 | MINOR | `excluded_after_collection` mixes periods. |
| HV-5 | MINOR | MU 2025-03-21 is not "no candidate listed": its $100 call traded, and only the put bar is missing. |
| HV-6 | MINOR | Pass 2 could not check the `interpolated` flag on 129 of 238 bars. The close > 0.01 test covered it, but the doc should say so. |
| HV-7 | MINOR | `hvol.main()` does not verify or record the price, PIT, earnings and κ inputs, or the code state. The run log has minute resolution. |
| HV-8 | MINOR | The bootstraps drop non-finite draws (1,999 of 2,000 in the 2026 secondary). |
| HV-9 | MINOR | Design caveat: the same entry price sets both the sort and the return. The bias is small and conservative. |

There is no BLOCKER.

---

## Prospective findings

### PR-1 (MAJOR): the lock pins only the code tree; the read depends on unverified inputs

**Where:**
- `prospective.py:141-173`, in particular 144-149, 153, 159-160 and 164;
- `prospective.py:93-108` (line 99) and `111-117`;
- `PROSPECTIVE.md:17-32`.

**Evidence.**
- **The tree check is weaker than the doc.** The only code check is `HEAD:src/fearsignal == decl["src_tree"]`.
  - Any later commit that leaves `src/fearsignal` unchanged passes it.
  - So "the read is therefore run from a checkout of the declaration commit" (PROSPECTIVE.md:19-22) is not what is enforced.
- **Committed inputs are read without any check:**
  - `outputs/trading/sizing_calibration.json`: κ20 and κ_ewma for P2-2;
  - `config/trading_selected.json` and `outputs/trading/holdout/holdout_results.json`: they set `family_status()`, and so ND's `nd_tier_before` in `decide()`;
  - `outputs/trading/v2/selection.json` and `trials.jsonl`: used by `exploratory()`;
  - `data/raw/earnings_calendar.json`: git-ignored, and in no lock that `read()` checks.
- **Locked inputs that `read()` never verifies:**
  - `ir_post_all()` calls `backtest.load_inputs()`, which reads `pit_*`, `trade_ohlcv.parquet` and `live_frozen_20261127.parquet`.
  - The live panel's history before 2026-11-23 comes from `dev.parquet` and `holdout.parquet`, through `data._frozen_history()`. That history sets σ, and so the ND1 events, until about May 2027.
  - All of these are hash-locked in `config/trading_inputs.json`, but `read()` never calls `protocol.check_inputs()`.
  - Today all eight hashes, including the v2 earnings lock, match their locks; I checked each.
- **`read.json` records no data hashes.** It keeps `code`, but `data` is just `{"source": "yahoo"}`. There is no hash of `live.parquet` or the live price cache, and no list of the 400 session dates.
- **The environment is not pinned.**
  - `pyproject.toml` gives only lower bounds.
  - This venv has pandas 3.1.1, numpy 2.5.5, scipy 1.18.1, pyarrow 26.0.1 and yfinance 1.8.0.
  - A venv rebuilt by 2028 would run the locked tree on different libraries.
- **A worktree will not work.** Every path resolves from the checkout (`frozen.ROOT`, `data.RAW`), so a `git worktree` of the declaration commit has no `data/raw`. Only an in-place `git checkout` works, and the doc does not say so.

**Fix.**
- **In `config/trading_prospective.json`, record:**
  - the declaration commit and the src tree;
  - the SHA-256 of each committed input above;
  - a reference to `config/trading_inputs.json`;
  - the environment: a pip-freeze digest plus the key versions.
- **In `read()`:**
  - verify all of these (refuse on a mismatch, or require a logged reason);
  - call `protocol.check_inputs()` before `ir_post_all`;
  - write into `read.json` the SHA-256 of `live.parquet`, the price cache and the earnings calendar used, and the 400 session dates.
- **In the doc:** spell out the in-place checkout procedure.

### PR-2 (MAJOR): the once-only guard is defeated by the prescribed procedure

**Where:** `prospective.py:150-152`; `PROSPECTIVE.md:23-24`.

**Evidence.**
- **The guard is a working-tree file.** A second read is refused only if `outputs/trading/prospective/read.json` exists in the working tree.
- **The prescribed checkout deletes it.**
  - The read runs from a checkout of the declaration commit, where that file is untracked.
  - Every other read in this repo was committed afterwards (`holdout_results.json`, `hvol_dev.json`). If `read.json` and `RUNLOG.md` are committed the same way, a later `git checkout <declaration commit>` deletes both from the working tree.
  - `read()` then runs again with no reason and starts a fresh `RUNLOG.md`.
- **An interrupted read can also be repeated without a reason.** Its RUNLOG has "read started" but there is no `read.json`.
- For comparison, `protocol.check_holdout_unlock` refuses if either the results file or the run log exists.

**Fix.** Refuse without a `rerun_reason` in any of these cases:
- `RUNLOG.md` exists;
- any ref holds a prospective read (`git log --all --format=%H -- outputs/trading/prospective/` is non-empty).

Alternatively, keep the guard file under the git-ignored `data/raw/`, which survives checkouts.

### PR-3 (MAJOR, reported-only section): `exploratory()` silently miscomputes the earnings-based survivors

**Where:**
- `prospective.py:111-126`;
- `explore.py`: `earnings_sessions` (272-303), E7 (163-174), E11 and E11F (194-206), and `Ctx.refits` (61);
- `PROSPECTIVE.md:73-76`.

**Evidence.**
- **The calendar ends before the sample.** `data/raw/earnings_calendar.json` has monthly windows 2023-11..2026-11; its last report date is 2026-11-24.
  - On the sample, `earnings_sessions()` is therefore all False.
  - Nothing checks coverage.
- **What the sample will see.** I ran the five survivors on the 2026 window with the live state, first as is and then with an all-False earnings mask:

  | trial | 2026, with the calendar | 2026, calendar missing |
  |---|---|---|
  | E7 H 20 | t 0.57 | mean 0.0, t 0.0, "fewer than 5 events" |
  | E7 H 40 | t −0.07 | same zero |
  | E7 H 40, inverse vol | t −0.24 | same zero |
  | E11 winners H 5 | t −0.50 | t −0.29 |
  | E11F H 10 | t −0.38 | t −0.58 |

  - E7 is reported as a computed zero, not as "not computable". PROSPECTIVE.md promises the latter.
  - E11 and E11F lose the `~earnings` part of their no-news filter. A different rule is reported under the same name.
  - All five survivors read the calendar.
- **Refreshing the calendar in place breaks a lock.** Overwriting `data/raw/earnings_calendar.json` would break `config/trading_v2_inputs.json` (`8b1991bb…`).
- **σ̃ comes from a different model.** `Ctx(rf=live, …)` loads the PIT refits. E7's inverse-vol weights therefore combine the 2026-10-01 PIT A and s with frozen v3's probabilities. On 2026 its mean is −2.486e-5, against −2.509e-5 recorded in `holdout_2026.json`.

**Fix.**
- Read the sample's calendar from a separate file that is pinned at read time.
- Require its windows to cover [start − max H, end] for every name. Otherwise report "not computable: the calendar does not cover the sample".
- Give `Ctx` frozen v3's A and s for σ̃, or document the mix.

### PR-4 (MAJOR, live path, not the read): `trading-live` and its monitors fail from 2027-09-09

**Where:**
- `calendar.py:12-33`;
- `forecast.py:54-59`;
- `live.py:138` and `174`.

**Evidence.**
- `calendar.sessions(prices.dates)` adds 80 future sessions. It refuses past 2027-12-31, where both `NYSE_HOLIDAYS` and `HALF_DAYS` end.
- Stepping the last bar through 2027, it first raises at the 2027-09-09 close, session 195 of 400.
- From then on `trading-live` raises before it prints any tickets. The guard and the demotion stop running until the tables are extended.
- **The read is not affected:**
  - the read path never calls `calendar.sessions`;
  - `HALF_DAYS` has no 2028 entries, but the first 2028 early close (2028-07-03, by NYSE convention) is the 400th session itself;
  - the last countable ND1 signal is 2028-06-01, so `half_day` is right for every signal that P2-1 counts.

**Fix.**
- Before the lock, extend `NYSE_HOLIDAYS` and `HALF_DAYS` through 2028, and raise `HALF_DAYS_COVER_THROUGH`.
- Take the dates from the NYSE's published calendar. By the usual rules:
  - full holidays: Jan 17, Feb 21, Apr 14, May 29, Jun 19, Jul 4, Sep 4, Nov 23 and Dec 25;
  - early closes: Jul 3 and Nov 24.

### PR-5 (MINOR): `decide()` against the declaration

**Where:**
- `prospective.py:129-138` and `164-167`;
- `test_prospective.py:53`;
- `declaration.json:101` (section 2) and `102` (DEMOTION).

**Evidence.**
- **(a) A live suspension or demotion is ignored.**
  - "Otherwise: stays at Tier 1" uses the static `family_status()` tier.
  - Suppose the guard suspended ND, or two demotions sent it to Tier 0, during the sample. If P2-1 is then not claimed but both means are positive, the read returns Tier 1.
  - That reinstates the family without a pass. The declaration says "A suspension stands until a review on dev data only" and "promoted again only through the pre-registered prospective test".
- **(b) The "IR_post ≤ 0 → Tier 0" rule is missing.** Section 2 says "If IR_post <= 0, the family is tier 0". `decide()` does not encode it, and the test asserts Tier 1 for a claim with IR_post = −0.01.
  - This cannot happen in practice. IR_post has the sign of the pooled per-event mean. With the dev mean at +2.32% and 2026 at +2.08%, any positive prospective mean forces IR_post > 0.
  - The table and the test should still match the declaration.
- **(c) Nothing applies the decision.** `family_status()` reads only `trading_selected.json` and `holdout_results.json`, so the "mechanical" consequences have to be applied by hand.

**Fix.**
- Pass `monitor(…)`'s `tier_effective` into `decide()` as the starting tier. It is deterministic on the closed in-sample events.
- Encode the IR_post ≤ 0 rule.
- Either wire `read.json` into `family_status()`, or say that the consequences are applied manually.

### PR-6 (MINOR): sessions are counted from vendor dates

**Where:** `prospective.py:53-59`; `market_data.load`.

**Evidence.**
- The sample is the first 400 entries of the union of dates in the yfinance cache. A single spurious row, such as a holiday bar for one ticker, would shift the sample by a session.
- `live.state()` checks only that the last dates of the two sources agree.
- The current cache is clean: every date has at least 123 tickers.

**Fix.**
- Require that each sample date is an exchange session and has bars for most names.
- Write the 400 dates into `read.json`.

### PR-7 (MINOR): two "reported only" families are not computed

`PROSPECTIVE.md:77-79` lists SR and RF as reported on the sample. `read()` (`prospective.py:165-169`) computes only the five v2 survivors.

**Fix:** either compute SR1 (with the same run as `p21`) and RF (with `F.rf_stats`), or drop them from the list.

### PR-8 (MINOR): monitors against the declared text

**Where:** `prospective.py:193-216`; `PROSPECTIVE.md:87-101`; `declaration.json:101-102`.

**Evidence.**
- **(a) Blocks instead of a rolling window.** DEMOTION says "rolling 40-trade", but the code checks non-overlapping blocks at the 40th, 80th, … event.
  - Over the roughly 110 events expected, that is about 2 checks against about 70 for a rolling window, so it weakens a declared risk rule.
  - PROSPECTIVE.md's "reading" changes the declaration. It should be logged as an amendment to `declaration.json`, or the rule should be implemented as rolling.
- **(b) Nothing is persisted.**
  - Suspension and demotions are recomputed from scratch on every run.
  - A vendor revision of past prices could undo a suspension that is meant to "stand".
- **(c) The live report shows part of P2-1.** "They do not compute P2-1" (PROSPECTIVE.md:100-101) is not quite true: the live report shows the running per-event mean and its CR1 t, and that mean is one of P2-1's two claim conditions.
- **Checked on real data.** I treated the 2026 events as prospective, in memory only. Neither rule fires. The guard's thresholds were −4.6% at 25 events and −2.1% at 50.

**Fix.**
- Implement a rolling window, or log the amendment.
- Persist each firing in an append-only file and honour it on later runs.
- Reword (c) as "they do not compute the P2-1 gate statistic".

### PR-9 (MINOR): missing tests

**Missing:**
- the `read()` refusals: a tree mismatch, a second read and an interrupted read;
- `p22`: for example, with the PIT refits it should reproduce `sizing_qlike_2026.json` at h 20;
- `ir_post_all`: the window boundaries and the absence of duplicates;
- the live hook: it should only lower the tier.

**Also:** `test_p21_reproduces_the_v1_holdout_read` is skipped silently when the inputs are absent. Inside `read()`, missing inputs should be a hard failure.

### PR-10 (MINOR): stale wording

- `holdout_v2.py:4-6` and `LOG.md:135` say the survivors "become the prospective test's pre-registered hypotheses".
- `PROSPECTIVE.md` lists them as reported only.
- This is already an open item in `LOG.md:246-250`.

### PR-11 (MINOR, numerically negligible): κ wording and N_eff

**κ wording** (`PROSPECTIVE.md:31-32, 39`): "each scaled by its dev level factor" is ambiguous for frozen v3.
- The factor used is the PIT-refit calibration. I reproduced `sizing_calibration.json` exactly with `sizing.calibrate`, to 1e-12 for every h.
- On dev, frozen v3's σ̃ is about 5.6% above PIT's. Its implied κ20 is 0.905, against 0.949 for PIT on the same stock-days.
- On 2026 the two levels agree (median ratio 1.000), and P2-2 on 2026 with frozen v3 gives −0.0623 (t −1.38), against −0.0602 (t −1.35) for PIT.
- So the choice is harmless for the sample. Still, say "the PIT dev κ" explicitly.

**N_eff** (`sizing.py:135-136`): the code counts ISO weeks of signal dates, while the declaration counts weeks "with at least one entry".
- On dev plus 2026, N_eff is 95 against 96.
- IR_post is 0.02848 against 0.02872.

---

## H_VOL findings

### HV-1 (MAJOR, statement): "IV beats σ̃" over-reads a post-hoc, non-significant diagnostic

**Where:**
- `H_VOL.md:163-169` and `181-196`;
- `PROSPECTIVE.md:68-69`;
- `docs/TRADING.md:231-233, 257-259, 366-368`;
- `README.md:65-66`;
- `LOG.md:205`, and the title of commit 5a96922.

**Evidence.**
- **The test.** I recomputed the table from `hvol_pairs_*.csv`; no code for it is committed. QLIKE of σ̃ minus raw IV:
  - dev: +0.0966 (CR1 by month, t 1.47);
  - 2026: +0.0447 (t 0.44).

  Neither is significant, and the comparison was not pre-registered.
- **The "IV × 1.127" row is in-sample on dev.** Its factor is √(ΣRV²/ΣIV²) = 1.1268, fitted on the dev pairs themselves.
- **The combination did better in 2026.** The geometric mean of σ̃ and IV beat raw IV in 2026 (−0.033). That is consistent with the positive, non-significant c.
- **The docs nonetheless state:**
  - "its IV is the better risk yardstick for sizing";
  - "it loses to listed-option IV";
  - "implied vol beats sigma-tilde";
  - "adds nothing beyond listed implied volatility", although c is +0.12 on dev and +0.18 in 2026.

**Fix.**
- Reword: "Point estimates favour listed IV (QLIKE differential t 1.5 on dev, 0.4 in 2026; post hoc, not significant). σ̃ adds no significant information beyond IV (c > 0, not significant)."
- Drop the sizing implication, or mark it as a hypothesis for a future pre-registered test.
- Commit the diagnostic's code.

### HV-2 (MINOR): p = 0.053 sits on a Monte Carlo knife edge, and H_VOL2 depends strongly on κ

**Where:** `hvol.py:236-253`; `H_VOL.md:151` and `158-160`.

**Monte Carlo error.**
- With B = 400,000, the converged bootstrap p is 0.0539.
- Over 400 seeds at B = 2,000, the p has a mean of 0.0547 and an s.d. of 0.005. 18% of seeds give p < 0.05.
- Seed 0 and B were fixed in code at `61fa3ab`, before the read, so the decision stands.
- The doc should state the Monte Carlo precision and the converged value.

**κ sensitivity (post hoc, for context only).** κ shifts the threshold, which matters a great deal:

| κ20 | dev H_VOL2 p | 2026 difference |
|---|---|---|
| 0.852 (2026's implied level) | 0.014 | +1.3 points |
| 0.90 | 0.079 | +9.8 points |
| 0.952 (declared) | 0.053 | +27.6 points |
| 1.00 | 0.150 | +21.6 points |

The "consistent direction" sentence (`H_VOL.md:170-172`) is fragile in size.

### HV-3 (MINOR): two interpretive sentences contradict the table

**Where:** `H_VOL.md:173-179`; `docs/TRADING.md:228-230`.

**Evidence.**
- **Model 1 did signal cheap options.** Against "not something model 1 predicted":
  - σ̃ ≥ IV held in 62% of dev pairs and 79% of 2026 pairs. The mean log(σ̃/IV) was +0.038 and +0.178.
  - The σ̃ ≥ IV group carried the gains: +17.0% (p 0.028) and +33.6% (p 0.017). The rest earned +2.5% and +6.0%.
  - What failed is the sorting test, not the level signal.
- **The dev level is fitted in-sample.**
  - κ20 was fitted on dev realized volatility over all 134 names.
  - So σ̃'s near-zero bias on dev (mean log RV/σ̃ = −0.006) holds by construction, and the dev threshold uses an in-sample level.
  - In 2026, σ̃ ran about 10% above realized volatility.

**Fix:** reword both sentences.

### HV-4 (MINOR): `excluded_after_collection` mixes periods

**Where:** `hvol.py:327` and `339-340`.

`pairs()` runs over all bars, and its exclusion counter is not split by period, so the dev output would include 2026 exclusions. It is `{}` today, so nothing is affected.

### HV-5 (MINOR): the MU 2025-03-21 label is inaccurate

**Where:** `H_VOL.md:137-138`; status `not_listed_at_entry`.

**Evidence.**
- The doc says "no candidate was listed at entry".
- The pass-2 record shows the $100 call had a real entry bar (close 6.43). Only the put had no bar (`not_found`).
- The exclusion is correct under amendment 1, which needs both legs.

**Fix:** say "the $100 put had no entry bar".

### HV-6 (MINOR): pass 2 could not test the `interpolated` flag on most bars

**Evidence.**
- 129 of the 238 bars checked in pass 2 carried no `interpolated` field (recorded as null). For those bars, the "not interpolated" half of the amended rule could not be applied.
- The close > 0.01 test covered it:
  - all 16 flagged-interpolated pairs in either pass (15 in pass 1, 1 among the pass-2 checks) had closes of exactly 0.01;
  - so close > 0.01 caught every known placeholder;
  - pass-2 pairs' median absolute parity residual (0.81%) matches pass 1's (0.88%).

**Fix:** state this in the doc.

### HV-7 (MINOR): provenance of the H_VOL read

**Where:** `hvol.py:142-154`.

**Evidence.**
- **Only the option files are checked.** `main()` verifies and records the hashes of the three option files only.
- **Not verified or recorded:**
  - the price, PIT, fear-panel and earnings inputs;
  - κ;
  - the code state.
- **The run log cannot prove the order.** It has minute resolution, and both reads share the lock commit's minute (00:56; `61fa3ab` is at 00:56:09).
- **The inputs do match today.** Every one of those files matches its lock; I checked all eight hashes.

**Fix:** add `protocol.input_manifest()`, the calendar hash, κ and `protocol.code_state()` to the outputs.

### HV-8 (MINOR): the bootstraps drop non-finite draws

**Where:** `hvol.py:245-248` and `262-266`.

A resample with no σ̃ < IV pair gives a non-finite draw, which is dropped; the 2026 secondary used 1,999 of 2,000. This conditions the bootstrap on both groups being present. It affects only reported results.

**Fix:** count such draws as not exceeding, or say that they are dropped.

### HV-9 (MINOR): design caveat

The same entry close sets both the sort (IV) and the return's denominator (C + P). Stale prints therefore bias H_VOL2 and H_VOL3 upward; Goyal & Saretto skip a day for this reason.
- The median absolute parity residual is 1.0% on dev and 0.6% in 2026, so the bias is small.
- It favours finding an effect, so "not supported" is conservative.

---

## Verified as correct

### H_VOL

**Inputs and sample.**
- The raw files' SHA-256 match `config/hvol_inputs.json`.
- The committed task list equals `build_tasks(close)`.
- There are 34 expiries. Two are holiday-moved: 2025-04-17 and 2026-06-18.
- Entry gaps are 28 days, or 29 after a holiday.
- None of the 10 names split in 2024-26. The largest daily gap between raw and adjusted log returns is 0.0064, from dividends.

**Pass merge.**
- In pass 1, every `tried` list is a prefix of the candidate list.
- In pass 2, all 135 chosen strikes are the first qualifying candidate. I checked each against the pass-1 non-listings, the pass-1 placeholder and the pass-2 checks.
- The 136 placeholders are exactly the 136 pass-2 keys.
- No placeholder survives among the 339 usable pairs.
- `is_placeholder` handles DataFrame rows correctly when the flags are NaN.

**IV, σ̃ and realized volatility.**
- **IV:**
  - S is the raw close, and it matches `spot_close`.
  - T is calendar days / 365, r is 4% and q is 0. IV is the average of the call and put IVs.
  - My own Black-Scholes and Brent inversion matches to 2.5e-8.
- **σ̃:**
  - My own implementation, built from `pit_signals.parquet` and `pit_refits.json`, reproduces `st_tot` to 6e-17.
  - h counts the sessions after entry through expiry (17 to 20).
  - The cross-section has 122 to 134 names.
- **No look-ahead in σ̃.**
  - σ at t excludes r_t.
  - The regime probabilities are filtered through t.
  - Each refit is trained through the session before its refit date.
  - `reindex(close.index)` is a no-op, because both frames come from the same price file.
  - κ20 reproduces exactly from `sizing.calibrate`.
- **Realized volatility** uses returns from entry+1 through expiry. It matches my version from `trade_ohlcv` adjusted closes to 7.7e-6 relative.

**Pair-level quantities.**
- S_T, the net straddle formula and the period split all match exactly.
- The `hi` and earnings-in-life flags agree pair for pair.
- The cost is a constant, so it cancels in H_VOL2.
- The earnings window (entry+1..expiry) uses reaction sessions, as declared.

**Statistics.**
- The CR1 factor is G/(G−1) × (n−1)/(n−k), and H_VOL1's p comes from t(G−1).
- The multinomial month bootstrap and the null-centred p = (1 + #{d* − d ≥ d})/(1 + B) are right.
- My independent code reproduces every headline number:

  | | dev | 2026 |
  |---|---|---|
  | H_VOL1 c | 0.1212 (s.e. 0.1248, t 0.971, p 0.171) | 0.1786 (t 0.823, p 0.215) |
  | H_VOL1 b on log IV | 0.873 (t 7.22) | 0.860 (t 3.94) |
  | H_VOL2 difference | +0.1451, p 0.0535, CI [−0.0218, 0.3225] | +0.2764, p 0.1089 |
  | H_VOL3 mean | 0.1696, p 0.0280 | 0.3361, p 0.0170 |
  | secondary H_VOL2 | 0.2267, p 0.0505 | 0.4081, p 0.028 |

  The describe block matches too.
- **The write-up's numbers.** Every number in the Results table and the post-hoc table matches, up to rounding. The dev σ̃ differential is +0.0966, printed as +0.096. The sample counts match: 339 pairs, 204 from pass 1 and 135 from pass 2.

### Prospective

**Sample window.** The first 400 sessions after 2026-11-27 run from 2026-11-30 to 2028-07-03, which matches "about early July 2028". The last countable signal is 2028-06-01.

**P2-1** is the v1 gate series. It reproduces both recorded reads exactly:

| | t | daily mean | events | per-event mean |
|---|---|---|---|---|
| 2026 | 0.60939 | 0.00063822 | 54 | 0.020756 |
| dev | 2.54259 | 0.0017492 | 145 | 0.023242 |

The live (frozen v3) state gives the same 54 events and the same mean on 2026.

**P2-2:**
- `qlike_oos` with the refits reproduces `sizing_qlike_2026.json`.
- The new `A_live`/`s_live` branch is backward compatible.
- On 2026 with frozen v3, the statistic is −0.0623 (t −1.38).

**Multiplicity.** Holm with m = 2, family-wise 0.10 and a 0.05 cap is the same as requiring each p < 0.05.

**Power figures.** All reproduce:

| statement | computation | power |
|---|---|---|
| P2-1 at the dev effect | t = 2.54 × √(400/502) = 2.27 | 0.73 |
| P2-1 at half the dev effect | t = 1.13 | 0.31 |
| P2-2 at 2026's effect | t = −1.35 × √(380/208) = −1.82 | 0.57 |
| P2-2 at the dev effect, with 2026's dispersion | t = −3.24 | 0.94 |

With dev's own dispersion, the last power would be about 1.0.

**IR_post** (`ir_post_all`):
- Nothing is counted twice and no windows overlap. The signal ranges are 2024-01-04..2025-11-26, 2026-01-06..2026-10-27, and from 2026-11-30.
- There are no duplicate (date, ticker) events and no shared ISO weeks.
- Dev plus 2026 alone gives IR_post 0.0285 with N_eff 95.

**`decide()`** implements the doc's three branches.

**`exploratory()`.**
- `X.Ctx(rf=…, prices=…)` works as called.
- On 2026 it reproduces the v2 read for 4 of the 5 survivors exactly (E7 with inverse-vol weights differs, see PR-3).
- Its output is JSON-serialisable.

**Monitors:**
- The guard runs every 25 events and suspends if the running mean is below the dev mean minus 2 CR1 s.e.
- Demotion uses blocks of 40, with mean < −1 s.e.; a second demotion means Tier 0.
- Both behave as PROSPECTIVE.md describes.
- The live hook only lowers the tier: `tier_effective` ≤ `tier` by construction, and the hook runs only when the tier is above 0.

**Refusal logic.**
- `read()` refuses in the right order: the declaration file, a dirty tree, a tree mismatch, an earlier `read.json`, then fewer than 400 sessions.
- The dirty check ignores `__pycache__`, `.pytest_cache` and `*.egg-info`, so build artifacts do not trip it.

**Tests.** The full suite passes: 150 tests.

---

## Reproduction (scratch scripts)

All are in `/tmp/claude-0/-home-user-fear-signal/746ee66c-3d9a-5a30-89ca-deab40db82a0/scratchpad/review/`.

| script | what it does |
|---|---|
| `indep.py` | Independent H_VOL recomputation from the raw files, with no `fearsignal` imports. |
| `repro_pkg.py`, `cmp_outputs.py`, `cmp_pairs.py` | Package reproduction (without `main()`), and comparison with the outputs and with the independent pairs. |
| `inspect_raw.py`, `inspect_raw2.py`, `check_strike_rule.py`, `tasks_check.py` | Pass-merge and strike-rule audit; task list and expiries. |
| `mc_p.py`, `kappa_sens.py`, `posthoc.py` | Monte Carlo error of p, κ sensitivity, and the post-hoc QLIKE table. |
| `p21_check.py`, `p22_check.py`, `kappa_check.py` | P2-1 against the v1 reads; P2-2 and the frozen-v3 κ levels; κ reproduction. |
| `explo_check.py`, `monitor_check.py`, `neff_dates.py` | Exploratory run and the missing-calendar simulation; monitors on 2026; N_eff and the date index. |
