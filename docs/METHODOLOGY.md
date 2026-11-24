# Methodology

This document states the model, derives each piece, and says which parts of
the v1 starting point were kept, which were changed, which proposed changes
the data rejected, and why. Numbers quoted
come from `outputs/dev/walkforward.json` (development period only) and
`config/frozen.json`.

## 1. The observation: how hard was the stock hit today?

For stock *i* on session *t* (previous valid session *t-1*), with adjusted
close *C* and adjusted low *L = low · C / close*:

```
r_t     = ln(C_t / C_{t-1})
sigma_t = 1.4826 · median(|r_{t-126}|, …, |r_{t-1}|)        today excluded
x_t     = min( max(ln(C_{t-1} / L_t), 0) / sigma_t , 10 )
```

*Kept from v1 unchanged.* Two facts justify it.

**Robust scale.** 1.4826 = 1/Φ⁻¹(3/4) makes `median|r|` consistent for σ
under normality. The median tolerates up to half the window being stress
days, so the "normal daily move" is not inflated by the very episodes being
measured. The median m = 0.6745σ of |r| has asymptotic variance
1/(4n f(m)²) with f(m) = 2φ(0.6745)/σ = 0.6356/σ, so
sd(σ̂)/σ = 1.4826 / (2 · 0.6356 · √n) ≈ 1.17/√n: about 10% at n = 126.

**Half-normal emission (reflection principle).** If log price is a driftless
Brownian motion with daily variance σ², the running minimum over one day
satisfies P(−min B ≥ m) = 2 P(B₁ ≤ −m): the drawdown-to-low is *exactly*
|N(0, σ²)|. So a calm regime should produce x ≈ half-normal with scale ≈ 1,
and stress shows up as a larger scale. The fitted calm scale (0.96) agrees
with that benchmark; `tests/test_features.py` checks the reflection claim by
simulation.

Every observation is typed: **MISSING** (no bar, fewer than 126 in-era
returns, or scale below a 0.3% stale-data floor), **ZERO** (x ≤ 0.001: the
low never went below the previous close), **EXACT**, or **CENSORED**
(d ≥ 10: only "at least ten sigma" is retained).

## 2. The regime model

Hidden state S ∈ {calm, stressed, panic} follows a Markov chain with
transition matrix A, one parameter set for all stocks. The frozen model is
v1's: given S = k, an EXACT x has density (2/s_k) φ(x/s_k), a clipped x is
evaluated at the clip value, and a no-dip day carries no evidence (e = 1).
It is **refit** by Baum–Welch on this universe and period (134 stocks,
2023-07-07 .. 2025-12-31), giving s = (0.961, 1.654, 3.948) against v1's
published (0.95, 1.58, 3.82) — an independent replication of the starting
point.

### Two corrections that were proposed, tested, and rejected

The code implements a general emission so the following could be tested:

| observed | likelihood given k |
|---|---|
| ZERO | π_k + (1 − π_k)(2Φ(ε/s_k) − 1) |
| EXACT x | (1 − π_k) · (2/s_k) φ(x/s_k) |
| CENSORED | (1 − π_k) · 2(1 − Φ(10/s_k)) |
| MISSING | 1 |

* **Censoring.** A clipped value means "fell ten sigma or more", whose
  likelihood is a survival probability, not the density at 10. Censoring is
  the textbook-correct treatment, and it raises the fitted panic scale
  (≈ 4.7–5.0 vs ≈ 3.7–4.3).
* **Modelled no-dip days.** v1's "no evidence" rule is the restriction
  π_calm = π_stressed = π_panic (a state-independent atom cannot move the
  filter). Freeing π_k nests it. In-sample the likelihood-ratio statistic is
  large (2·ΔlogL ≈ 29 on 2 df), and the fitted panic π is ≈ 0.

Both lose out of sample. On the pre-declared walk-forward (§7) the refit v1
model has the best one-step predictive log score; the censored shared-atom
model is 0.0006 nats/obs worse (s.e. 0.0008) and the free-atom model 0.0011
worse (s.e. 0.0010). The free atom is also unstable. Trained through mid-2024
its likelihood is multimodal and slow to converge: π_panic sits at 0.65
after 1,500 iterations and is still moving. The later folds converge to
π_panic ≈ 0. With the differences inside one
standard error, the selection rule keeps the model with fewer parameters,
which is v1's. The in-sample likelihood-ratio test was misleading because
it treats 79,000 stock-days as independent evidence, when they share a handful
of market episodes.

### Estimation

Baum–Welch on all stocks pooled, each stock its own sequence from its first
to its last eligible day. The E-step is the scaled forward–backward
recursion. The M-step is closed form for every variant, including the
censored and zero-inflated parts, using truncated-normal second moments:

```
E[X² | X ≥ c] = s²(1 + a λ(a)),             a = c/s, λ = φ/(1 − Φ)
E[X² | X ≤ ε] = s²(1 − 2aφ(a)/(2Φ(a) − 1)), a = ε/s
r = π_k / (π_k + (1 − π_k)(2Φ(ε/s_k) − 1))    (P(atom | ZERO day, k))
π_k   = Σ_ZERO γ r / Σ_all γ
s_k²  = Σ γ E[X² | obs] (1 − r on ZERO days) / Σ γ (1 − r on ZERO days)
```

so no iteration can lower the likelihood. Tests verify the
filter and smoother against exhaustive path enumeration, EM monotonicity,
parameter recovery on simulated panels, and that the EM fixed point equals a
direct numerical maximum-likelihood fit.

## 3. The filter is causal

```
pred_t  = α_{t−1} A          (stationary law on a stock's first eligible day)
α_t     = pred_t ⊙ e(x_t) / c_t,     c_t = p(x_t | x_<t)
```

α_t uses nothing after t. `tests/test_hmm.py::test_filter_is_causal` rewrites
every observation after a cut-off and checks that all earlier filtered values
are bit-identical.

## 4. Severity, Z and F

**Severity is the posterior mean log-volatility.** v1 defines
p = w·P(stressed) + P(panic) with w = ln(s₁/s₀)/ln(s₂/s₀) = 0.367. That is
exactly

```
p = (E[ln s_S | x_≤t] − ln s₀) / (ln s₂ − ln s₀),
```

the filtered expectation of the regime's log-scale, rescaled so calm = 0 and
panic = 1. The weight w is simply the stressed regime's position on the
log-volatility axis. This identity justifies the formula and extends it to
any number of regimes.

L = logit(p) is mapped to a standard-normal score through a frozen,
monotone quantile map fitted on 2023–2025 stock-days,
Z = Φ⁻¹(F_dev(L)), linearly extrapolated beyond the calibration range and
clipped to ±6; F = 100·Φ(Z). Bands are v1's: calm < 1, uneasy 1–1.5, stressed
1.5–2, acute fear 2–2.5, panic ≥ 2.5.

## 5. Market fear

**What v1 did.** M_t = z-score of the day's mean Z against all earlier days.
Three defects: (i) the yardstick changes every day, so the same market state
reads differently in 2026 than in 2024; (ii) the first weeks of the sample
are standardised against a handful of days; (iii) mean Z mixes "how calm" the
calm stocks are (deeply negative Z varies a lot) with how many are afraid.

**Why not pool stocks in one likelihood.** A market-regime HMM that treats
the N stocks as independent given the market state multiplies N likelihoods
that share one common shock; its posterior is roughly N times
over-confident and flips to 0/1 on any broad down day.

**v2: breadth.** The expected share of eligible stocks that are not calm,

```
B_t = (1/N_t) Σ_i P_i(S_t ≠ calm | x_i,≤t),
```

is an average of posteriors, not a pooled likelihood, so the dependence
problem does not arise; each stock keeps its own scale and its own filtered
history. M = Φ⁻¹(F_dev(logit B)), with the map frozen on 2023–2025 days, so
M is the dev-period percentile of today's breadth on a normal scale and its
yardstick never moves. F_M = 100·Φ(M).

A market-level Gaussian HMM on a trimmed cross-sectional summary
(y_t = ln of the 10%-trimmed mean of x) was a declared candidate. It tracks
VXN poorly (5-session-change Spearman 0.21) because one day's cross-section
discards each stock's own scale and filtered history; it was rejected.

## 6. Forecasts and uncertainty

**Tail forecast.** The filter yields a genuine forecast:

```
q_t = P(x_{t+1} ≥ 3 | x_≤t) = Σ_k (α_t A)_k (1 − π_k) 2Φ(−3/s_k),
```

the probability that tomorrow's low is at least three own-sigmas below
today's close. It is scored with the Brier score and logistic calibration
(slope 1, intercept 0 when calibrated; standard errors clustered by day,
because stock-days on one date share a market shock). The baselines are
climatology and EWMA (λ = 0.94) scale forecasts of |x| and of x², mapped
through the same zero-inflated half-normal tail.

**Inference for overlapping horizons.** Forward-return and forward-volatility
tests use overlapping windows. In samples of a few hundred days, Newey–West
t-tests on such regressions over-reject badly: the simulated size of a
nominal 5% test of 20-day returns at n = 225 is 22%. Return predictability
is therefore tested with Hodrick's (1992) 1B reverse regression, which
regresses the one-day return on the rolled-up regressor
Σ_{j<h} M_{t−j}. Its errors do not overlap, and its simulated size is 5–6%.
The volatility-forecast gain is tested with Diebold–Mariano using the
Harvey–Leybourne–Newbold correction (simulated size 4.7%). Both size checks
are in `tests/test_stats_size.py`.

**Parameter uncertainty.** What limits precision is the handful of stress
episodes, which all stocks share. Replicates come from a synchronised
moving-block bootstrap in time (63-session blocks, the full cross-section
kept together), refit by EM. Each replicate is pushed through the filter
with its own dev-fitted quantile maps, giving 90% bands for Z and M.

**Conviction.** A *high-conviction* market fear reading is one whose lower
90% band clears the threshold (M_lo ≥ 2): it survives the estimation
uncertainty of the model, not just the point estimate.

## 7. Model selection (development period only)

Candidates and selection rules were committed before the harness was run
(`src/fearsignal/model.py::CANDIDATES`, `src/fearsignal/evaluate.py`):
three expanding-window folds with test halves 2024H2, 2025H1 and 2025H2,
refitting everything on each training window. The full results are in
`docs/RESULTS.md` §1 and `outputs/dev/walkforward.json`. Outcome:

* **Stock layer:** S1 (v1's emission, refit). v1's published parameters
  (S0) score higher still, but they were estimated by their author on
  2023–2025, so they have seen every test fold. S0 is a reference, not a
  candidate.
* **Market layer:** breadth from S1. Its 5-session-change correlation with
  VXN (Spearman 0.661, Pearson 0.679) beats v1's expanding z-score on the
  same filter (0.624, 0.604). The paired Spearman gain is +0.037, with 95% CI
  [0.001, 0.066], over 363 within-fold changes.

## 8. Limitations

* The universe is drawn up with 2026 knowledge (survivorship, AI-winner
  selection). For a contemporaneous thermometer this changes which stocks are
  averaged, not when the index reads high; it biases any forward-return test
  upward, which is one reason return predictability is a secondary claim.
* One parameter set for all stocks; the per-stock scale σ carries the
  heterogeneity.
* Only adjusted close, close and low enter the signal. Highs, opens, volume
  and options are excluded by design.
