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
| `M`, `F_M` | market fear: breadth on a frozen 2023–2025 percentile scale (standard-normal), and 0–100 |
| `M_lo`, `M_hi` | 90% parameter band; **high-conviction fear** means `M_lo ≥ 2` |
| `M_legacy` | the v1 market index (expanding z-score of mean Z), kept for comparison |

Inputs are adjusted close, close and low, and nothing else. VXN is used only
to validate.

## Verdict (2026 holdout, frozen spec, run once)

| pre-registered test | 2026 result | |
|---|---|---|
| H1 M tracks VXN (5-session changes) | Spearman 0.548 (95% CI 0.39–0.67); levels 0.81 | **pass** |
| H3 / H3b reads high when hit, no false panic | 559/559 three-sigma loss days at Z ≥ 1.5; 0/139 panics without a dip | **pass** |
| H2 tail forecast calibrated, skilful | slope 0.801 (floor 0.8); Brier skill +0.7% vs climatology (t 1.44) | **pass, barely** |
| H2b beats an \|x\| EWMA | +1.7% Brier skill | **pass** |
| H1b v2 market index beats v1's | −0.035 (CI −0.077 to +0.001) | fail |
| H4 adds to a HAR volatility forecast | DM-HLN t 0.86 | fail |
| H5 predicts returns | Hodrick 1B t 0.70 | fail |

**A validated fear thermometer and a calibrated next-day tail-risk
probability. Not a return predictor.** v1's stock model was independently
replicated and beat every alternative tested. The new market aggregation
(breadth) did not beat v1's on the primary 2026 test; the two are
statistically indistinguishable. The value v2 adds is a calibrated forecast,
stated uncertainty, a fixed yardstick, and claims that can be checked. Full
write-up: `docs/RESULTS.md`.

Latest reading (2026-11-23): **M = 1.80** (90% band 1.65–1.94), *stressed*;
74% of AI stocks are in a stressed or panic regime.

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
   the freeze, and of the results after.

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
  hmm.py        scaled filter/smoother, exact EM, half-normal and Gaussian emissions
  scoring.py    severity, logit, frozen quantile map, Z/F/bands, robust I
  market.py     breadth, legacy M, market summary
  model.py      candidate specs, fitting, full signal, tail forecast
  bootstrap.py  synchronised moving-block bootstrap in time
  stats.py      Newey-West, HAC OLS, stationary bootstrap, calibration
  evaluate.py   dev walk-forward harness
  validate.py   pre-registered tests
  pipeline.py   freeze / apply / holdout
  figures.py    docs figures
tests/          engine vs brute force, EM properties, causality, protocol
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
