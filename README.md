# fear-signal

A causal, price-only **fear index for AI stocks**, built only on data from
2023-01-03 onward. Per stock and day it measures how far below the previous
close the stock traded at its worst moment, in units of the stock's own
normal daily move. It filters that through a three-regime hidden Markov
model (calm / stressed / panic) and reports:

| output | meaning |
|---|---|
| `Z`, `F` | stock fear on a standard-normal scale, and 0–100 (`F = 100·Φ(Z)`); bands calm < 1 ≤ uneasy < 1.5 ≤ stressed < 2 ≤ acute fear < 2.5 ≤ panic |
| `Z_lo`, `Z_hi` | 90% band from parameter uncertainty (block bootstrap) |
| `q_next` | calibrated probability that tomorrow's low is ≥ 3 own-sigmas below today's close |
| `I` | the stock's fear relative to the other AI stocks that day (robust z) |
| `breadth` | expected share of AI stocks currently in a stressed or panic regime |
| `M` | **v3** market fear: logit breadth on a fixed linear yardstick, `(logit breadth − μ)/σ` (v2: the percentile score below) |
| `M_pct`, `F_M` | breadth's percentile among calibration days, as a normal score and as 0–100; **bands and alerts use this** |
| `M_lo`/`M_hi`, `M_pct_lo`/`M_pct_hi` | 90% parameter bands; **high-conviction fear** means `M_pct_lo ≥ 2` |
| `M_legacy` | the v1 market index (expanding z-score of mean Z), kept for comparison |

Inputs are adjusted close, close and low, and nothing else. VXN is used only
to validate.

## Status

* **v3 (current), frozen 2026-11-23, awaiting its prospective test.** v3
  changes the market index's scale to the object v2's dev selection actually
  chose, and changes how the volatility test is run. Everything else is v2.
  The market-scale change was selected on 2023–2025 alone and passed a
  data-snooping-robust gate after 2026 had been spent. The volatility-test
  change is an evaluation convention with no improvement claimed
  (`docs/V3.md`). Its single pre-registered
  read is the first 400 sessions after 2026-11-23, around mid-2028
  (`docs/PREREGISTRATION_v3.md`). The code refuses to read that sample
  earlier.
* **v2, frozen 2025-12-31, tested once on 2026.** Its verdict is below.

## Verdict (2026 holdout, frozen spec, run once)

| pre-registered test | 2026 result | |
|---|---|---|
| H1 M tracks VXN (5-session changes) | Spearman 0.548 (95% CI 0.39–0.67); levels 0.81 | **pass** |
| H3 / H3b reads high when hit, no false panic | 559/559 three-sigma loss days at Z ≥ 1.5; 0/139 panics without a dip | **pass** |
| H2 tail forecast calibrated, skilful | slope 0.801 (s.e. 0.118; floor 0.8, fragile); Brier skill +0.7% vs climatology (t 1.44) | **pass, at the boundary** |
| H2b beats an \|x\| EWMA | +1.7% Brier skill | **claimed** |
| H1b v2 market index beats v1's | −0.035 (CI −0.077 to +0.001) | not claimed |
| H4 adds to a HAR volatility forecast | DM-HLN t 0.86 | fail |
| H5 predicts returns | Hodrick 1B t 0.70 | fail |

**A validated fear thermometer and a calibrated next-day tail-risk
probability. Not a return predictor.** v1's stock model was independently
replicated and beat every alternative tested. The new market aggregation
(breadth) did not beat v1's on the primary 2026 test; the two are
statistically indistinguishable. The value v2 adds is a calibrated forecast,
stated uncertainty, a fixed yardstick, and claims that can be checked. Full
write-up: `docs/RESULTS.md`.

Latest reading (2026-11-23):
- v2 frozen: **M = 1.80** (90% band 1.65–1.94), *stressed*; expected share of AI stocks in a stressed or panic regime 74%.
- v3: **M = 2.17** (band 2.00–2.23) on its linear scale; percentile score 1.95, *stressed*; breadth 82%. 2026-11-20 was a high-conviction *acute fear* day (M_pct_lo ≥ 2).

The two versions differ on 11-23 for two reasons. Re-estimation with 2026
data, in which the stressed regime is stickier (self-transition 0.93 against
0.89), moves the percentile score from 1.80 to 1.95. The change of scale, from
percentile to linear, moves the headline from 1.95 to 2.17.

![Market fear and VXN](docs/figures/market_fear.png)

## How it was built

1. **Hard rules in code.** Nothing before 2023-01-03 may enter a computation
   (`config.assert_era`). 2026 is a locked holdout: development code cannot
   load it, and the holdout loader requires the SHA-256 of a committed,
   unmodified `config/frozen.json`.
2. **Pre-declared comparison on 2023–2025.** Six stock-layer candidates and
   three market-layer readings were scored by expanding-window walk-forward,
   using rules committed before the run. The v1 emission design survived. The
   two corrections I proposed (censored clipping, modelled no-dip days) were
   rejected out of sample. Breadth replaced v1's expanding z-score.
   → `docs/RESULTS.md` §1
3. **Freeze, pre-register, run once.** Parameters, quantile maps, 200 bootstrap
   replicates, climatology, HAR coefficients and test thresholds were frozen.
   The hypotheses and pass criteria were written down and committed before
   any 2026 data was downloaded. → `docs/PREREGISTRATION.md`
4. **Independent correctness review** of the code by a separate agent before
   the freeze, and of the results after (`docs/REVIEW.md`).
5. **Course correction without overfitting (v3).** Four agents diagnosed the
   2026 weak spots, and an adversarial critic tried to refute every fix.
   Changes were chosen only on 2023–2025 walk-forward, through a calibrated
   Hansen-SPA family gate with Holm. One signal change passed. One promising
   stock-layer idea (F5) passed the statistical gate but failed its mechanism
   and market-layer conditions; it is frozen as a prospective challenger
   instead. → `docs/V3.md`, raw provenance in `research/v3/`

The math, with derivations (reflection principle for the half-normal,
severity as posterior mean log-volatility, exact EM for censored and
zero-inflated emissions, why breadth rather than a pooled likelihood):
`docs/METHODOLOGY.md`. Data handling and universe: `docs/DATA.md`.

## Usage

```bash
python -m venv .venv && .venv/bin/pip install -e ".[data,plot,dev]"
.venv/bin/python -m pytest                      # engine, math and protocol tests

fearsignal download-dev                          # 2023-01-03 .. 2025-12-31
fearsignal evaluate-dev                          # walk-forward comparison
fearsignal freeze --stock-spec S1_v1_refit       # writes config/frozen.json
fearsignal dev-reference                         # in-sample reference metrics
fearsignal download-holdout --end-exclusive YYYY-MM-DD
fearsignal holdout --sha256 <sha256 of config/frozen.json>
fearsignal latest                                # latest market and stock readings

# v3
fearsignal posthoc-2026                          # labelled diagnostic of the v3 market scale
fearsignal freeze-v3                             # re-estimate on 2023-01-03..2026-11-23
fearsignal v3-reference                          # in-sample v3 metrics (reference only)
fearsignal download-prospective --end-exclusive YYYY-MM-DD   # only after v3 is committed
fearsignal prospective --sha256 <sha256 of config/frozen_v3.json>  # once, at 400 sessions
```

Outputs: `outputs/holdout/market_fear.csv` (daily M, band, breadth,
legacy M), `outputs/holdout/stock_fear.csv.gz` (per stock-day x, Z, band, F,
I, q_next, state probabilities), `outputs/holdout/holdout_results.json`.

## Layout

```
src/fearsignal/
  config.py     era and holdout guards, constants
  universe.py   134 AI value-chain stocks, 11 segments, listing overrides
  data.py       download, manifest hashes, panels, holdout loader
  features.py   robust scale and drawdown-to-low observation
  hmm.py        scaled filter/smoother, exact EM, half-normal and Gaussian emissions,
                regime-indexed transitions
  scoring.py    severity, logit, frozen quantile map, Z/F/bands, robust I
  market.py     breadth, legacy M, market summary
  model.py      candidate specs, fitting, full signal, tail forecast
  bootstrap.py  synchronised moving-block bootstrap in time
  stats.py      Newey-West, HAC OLS, stationary bootstrap, calibration, Clark-West, Hodrick 1B
  spa.py        Hansen SPA (calibrated), model confidence set, episode robustness
  evaluate.py   dev walk-forward harness
  validate.py   pre-registered tests
  pipeline.py   freeze / apply / holdout
  figures.py    docs figures
  report.py     renders the holdout section of RESULTS.md from JSON
tests/          engine vs brute force, EM properties, causality, protocol, test sizes
research/v3/    raw provenance of the v3 course-correction round
```

## Known limitations

* The universe is chosen with 2026 hindsight (survivorship, AI-winner
  selection). That matters little for a contemporaneous thermometer, but it
  biases forward-return tests.
* The sample is short. 2023-H2 to 2026 holds only a handful of independent
  market stress episodes, which is the binding limit on every confidence
  interval here.
* Raw vendor data is not redistributed. `data/manifest_*.json` records
  hashes so a fresh download can be checked.
