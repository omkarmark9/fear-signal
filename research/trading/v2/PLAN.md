# Trading model v2: research program

Written 2026-11-28, before any v2 trial was run. v1 (`research/trading/`)
adopted one weak bearish rule (ND, Tier 1) and nothing tradable as a long.
v2 searches much more widely. Because it searches widely, it counts every
search.

## Ground rules

1. **Samples.**
   - **Exploration:** dev only, 2024-01-02..2025-12-31, with every cohort
     exiting by 2025-12-31.
   - **Holdout:** 2026-01-02..2026-11-27, read once, for the frozen v2 set
     only.
   - **The clean test** is prospective: sessions after 2026-11-27, with a
     pre-registered size and pass rule.
2. **Every trial is logged** in an append-only registry
   (`outputs/trading/v2/trials.jsonl`, with the daily series in
   `trial_series.parquet`). That covers the ten v1 variants, all
   pre-declared grid points, and every adaptive follow-up, including the
   failures. The selection statistics are computed over the whole
   registry, so the multiple-testing cost of searching is paid in full.
3. **Known before v2 starts (contamination disclosure).**
   - v1's dev and 2026 results: SR1 failed dev and rose in 2026; ND1 was
     marginal on dev and uninformative in 2026; RF1 passed dev on one
     episode and failed 2026.
   - The exploratory event study's results.
   - The results review's diagnostics (`research/trading/review.md`).

   2026 cannot select anything in v2, but it is not a clean confirmation
   for families resembling SR, ND or RF. Only the prospective test is
   clean.
4. **Checkpoints.** `LOG.md` in this folder records:
   - what has run;
   - every result;
   - the exact commit;
   - the next step.

   Every milestone is committed and pushed, so the program can resume
   after a context reset or a usage limit.

## The permutation space (Round 1, declared before running)

Execution is unchanged from v1:
- signals at the close of t, executed at the open of t+1;
- 10 bps per side, with a 25 bps stress test;
- engine tradability as before;
- point-in-time model-1 outputs.

### P: risk-based portfolio construction (long only, fully invested)

These use what model 1 forecasts best: volatility. Its regime-adjusted
yardstick σ̃ beat EWMA on QLIKE by 6–7 s.e. on dev. The literature behind
them: volatility timing beats 1/N net of costs (Kirby & Ostdiek 2012;
Fleming, Kirby & Ostdiek 2001), and low-risk weighting improves Sharpe
(Frazzini & Pedersen 2014).

- The portfolio rebalances every 5 sessions, with costs on turnover. The
  benchmark is the equal-weight basket.

| trial | weights |
|---|---|
| P1 (3 trials) | w ∝ σ̃_i(20)^−k, k ∈ {1, 2, 4} |
| P2 (2 trials) | w ∝ σ_EWMA^−k, k ∈ {1, 2}; does the fear yardstick beat a generic one? |
| P3 (2 trials) | equal weight, excluding names whose q_next is in the top {10%, 20%} |
| P4 (1 trial) | long-only minimum variance: σ̃ with the equicorrelation ρ_t, negative weights set to 0 |
| P5 (1 trial) | P1 with k = 1, combined with the P3 10% exclusion |

- **Statistic:** the vol-matched differential d_t = r_s − (σ_s/σ_b) r_b
  (mean > 0 ⇔ Sharpe_s > Sharpe_b).
- Also reported: the Sharpe difference with a bootstrap CI, turnover, and
  the return gap.

### T: market-exposure overlays on the equal-weight basket (RF generalisations)

- **T1 (6 trials):** the staircase. It switches on at M_pct ≥ {1.0, 1.5,
  2.0}, cuts exposure to {0, 0.5}, and steps back +0.25 per calm session.
- **T2 (1 trial):** a vol target on the fear model's basket volatility,
  σ̃ aggregated with ρ_t: E = clip(median/σ̂, 0.25, 1).
- **Statistic:** the utility differential d_t (γ = 5) against the static
  mix at Ē, as declared for RF.

### E: event families (cohorts, beta benchmark)

The primary series is the **1/H constant-capital series**. The
calendar-time series was shown to over-weight lone positions.

| family | rule | grid | trials |
|---|---|---|---|
| E1, news drift (bearish information, ND generalised) | day ∈ {calm, any}; x ≥ {2, 3}; g_seg ≤ {−1.5, −2.5}; v ≤ −2; cap 5 | H ∈ {5, 10, 20, 40} | 32 |
| E2, good-news drift (long, ND's mirror) | calm day; g_seg ≥ {+1.5, +2.5}; r ≥ σ; relvol ≥ {1, 2}; cap 5 | H ∈ {5, 10, 20, 40} | 16 |
| E3, systemic reversal (long, SR generalised) | delay ∈ {1, 2}; v ≤ {−1.5, −2.5}; M_pct filter ∈ {none, ≥ 1}; news filter on; cap 10 | H ∈ {3, 5, 10} | 24 |
| E4, earnings reaction drift (needs earnings data) | on a report session (pm reports react at the next open, am reports the same day), the reaction is the beta- and segment-adjusted move from the last close to the session's close, in own σ: \|v\| ≥ {1.5, 3}; good reactions long, bad reactions as bearish information | H ∈ {5, 20, 40} | 12 |
| E5, cross-sectional fear factor | each day, rank eligible names by {Z, I}; hold the {lowest, highest} 10 | H ∈ {5, 20} | 8 |

Round 1 is 9 + 7 + 92 = **108 trials**, plus the **10 v1 trials**, for
118 in total. Rounds 2+ are adaptive refinements of whatever looks
promising, capped at 100 more trials. Each is logged before its result is
viewed.

## Statistics computed for every trial (`trading/multitest.py`)

**Per trial**
- the mean and its stationary-bootstrap t;
- the annualised Sharpe of the performance series;
- for event families:
  - the per-event mean with t clustered by episode (signal days more than
    5 sessions apart);
  - the same-stock-adjusted mean;
  - the random-name placebo percentile (same days and counts, 200 draws);
  - costs ×2.5;
  - delay 2.

**Across the whole registry**
- **Romano–Wolf StepM** (Romano & Wolf 2005), stationary bootstrap,
  block 20: which individual trials beat their benchmark with family-wise
  error ≤ 10%.
- **Hansen SPA_c** over all trials: is the best trial better than chance
  at all?
- **Probability of backtest overfitting** by combinatorially symmetric
  cross-validation (Bailey, Borwein, López de Prado & Zhu 2017). Dev is
  cut into S = 16 contiguous blocks, and all C(16, 8) = 12,870 in/out
  splits are used. PBO is the share of splits in which the in-sample best
  trial ranks below the out-of-sample median.
- **Deflated Sharpe ratio** (Bailey & López de Prado 2014) of each
  candidate, using N = the effective number of independent trials (from
  the eigenvalues of the trial correlation matrix) and the series' skew
  and kurtosis.

## Selection rule (mechanical, applied to the full registry at the end)

A trial is a **v2 candidate** if all of these hold on dev:

1. StepM rejects it at FWER 10% across all trials.
2. Its deflated Sharpe ratio is ≥ 0.95.
3. Its mean stays > 0 at 25 bps per side and with a one-day delay.
4. For event families:
   - the per-event episode-clustered t is ≥ 2;
   - the placebo percentile is ≥ 95%;
   - the same-stock-adjusted mean is > 0.
5. The registry-wide PBO is < 0.5. Otherwise picking the in-sample best
   is no better than a coin, and no candidate is adopted.

Within each family (P, T, E1–E5), the candidate with the highest dev
Sharpe is chosen. The frozen v2 set is written to
`config/trading_v2.json` and committed. Then **2026 is read once**:
- **agrees:** mean > 0;
- **strong:** t ≥ 1.

The tiers follow v1's declared scheme. Finally a prospective test is
pre-registered: a fixed number of sessions after 2026-11-27, with a fixed
pass rule.

If nothing passes, v2 says so, and the live model stays as v1.
