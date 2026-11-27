# Pre-backtest audit of the trading model

The audit ran after the declaration was committed (4341c9b) and before any
declared rule was backtested on real data. It used four independent
auditors, one per lens.

They worked under one hard rule: compute no return that follows a signal
on real data. They could read everything, build real signals and count
events, perturb real inputs, and run the whole pipeline on synthetic data.
Every finding was verified and fixed before the placebo, the dev run and
the holdout read. The fixes are listed in `candidates.md` under
"Amendments before the dev run".

## What was verified correct

**Spec conformance**
- Every declared event count reproduces exactly, dev and 2026.
- The event sets are identical whether the no-re-entry walk runs over the full history or inside each window.
- An independent recomputation from the declaration text matches `signals.rule_features` exactly: systemic and share_hit2, rB, beta against explicit OLS (2,787 cells, maximum difference 6e-15), u, v and g_seg with the segment fallback.
- Every selection mask and contrast (S2, news-excluded, N3, systemic) equals one hand-built from the declaration.
- `suppress_reentry` matches the declared no-re-entry rule.
- On a synthetic world with missing prices injected, the implementation matches brute-force versions of the declared definitions to about 1e-16, for every variant at 10 bps, 25 bps and delay 2.
- RF1 reproduces the declared power note exactly: 141 of 502 dev sessions reduced, 38 changes, 12 episodes.
- The gates match the declaration: block 20, B = R = 999, both Holm thresholds 0.05 with m = 2, and all four gates required.

**Look-ahead and causality**
- **Perturbation on real inputs, at seven cut dates:** changing raw data after the cut moves no rule feature and no entry at or before it.
- **Point-in-time refits:** rebuilt on histories truncated at four dates, they reproduce the stored signals bit for bit (42 of 42 comparisons).
- **Brute-force recomputation:** beta (3,491 cells, maximum difference 6e-15), σ, the segment medians and dv_med20 all match.
- **Dev cut-off:** with every 2026 price and rule feature poisoned, the dev selection is unchanged in full. That covers the statistics, the SPA, the 1-SE choice, the signatures, robustness, Holm, the daily series and RF.
- **Same-stock baseline:** it uses no 2026 price.
- **HAR:** it fits only rows s ≤ t−5.
- **Live inputs:** M_pct in every backtest is point-in-time; only the live report reads frozen v3.

## Findings and their disposition

Four findings were major. AC-1 concerned the calendar-time portfolio's turnover-day weighting, and SP-1, SP-2 and SP-3 were protocol locks. The rest were minor. Only AC-1, SC-4 (non-universe rows in the H1-2024 basket) and SC-2 (RF's fit) change a dev statistic.

| id | lens | finding | disposition |
|---|---|---|---|
| SC-1, LA-1 | spec, look-ahead | 2026-07-02 is listed as a half-day but was a full NYSE session (volume 1.1× normal, against 0.3–0.7× on true half-days); 2026-12-24 is missing | fixed: the list is corrected for both models |
| SC-2 | spec | 10 fabricated zero basket returns in RF's HAR fit | fixed |
| SC-3 | spec | RF2's state never switches on at a half-day close | fixed (latent) |
| SC-4, LA-3 | spec, look-ahead | pre-listing rows (TLN OTC, OKLO's SPAC shell) in the benchmark basket | fixed |
| SC-5 | spec | input hashes not enforced | fixed: `config/trading_inputs.json` |
| SC-6 | spec | some declared reported diagnostics missing | fixed |
| LA-2 | look-ahead | no early closes after 2026-11-27 | fixed: through 2027, with a coverage guard |
| LA-4 | look-ahead | the live report could misalign its two caches | fixed: it refuses to run |
| AC-1 | accounting | the calendar-time series under-invested on every turnover day (entrants counted in the overnight leg, exiters in the intraday leg), shrinking means by 3–11% depending on H and diluting costs | fixed: the series is computed per leg; probe tests added; placebo rerun |
| SP-1 | statistics | the holdout read was not tied to the dev selection's code | fixed: a clean, identical `src/fearsignal` tree is required; the start is logged before any outcome |
| SP-2 | statistics | the declaration check covered 3 files | fixed: clean tree required, code state recorded |
| SP-3 | statistics | `trading-live` could show 2026 outcomes before the read | fixed: hidden until a matching read |
| SP-4 | statistics | 'upper bound' overstated the placebo (B = R = 199 against 999) | wording corrected; R-adjusted sizes reported |
| SP-5 | statistics | locking `live.parquet` would block the read after a refresh | fixed: frozen snapshot |
| SP-6 | statistics | live tiers could reuse a read under another selection | fixed |

**Verification of findings.** The workflow's skeptic phase (three refuters per finding) was stopped after the four
audits. With two agents allowed at a time, it would have taken several hours. Each finding was instead reproduced
and checked by the implementer, using the auditors' scripts and evidence. Each fix is covered by a regression test
where one applies: SC-2, SC-3, SC-4, AC-1 (three probes), SP-1 and SP-3/SP-6.
