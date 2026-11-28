"""Multiple-testing tools for a search over many trading rules.

Every v2 trial is tested on the same 2024-2025 data, so a trial that looks
good may only be the luckiest of many. These tools price the search, given
the full registry of trials. Each column of M is one trial's daily
performance series, where positive means better than its benchmark.

* ``stepm``          Romano & Wolf (2005) StepM. Which individual trials beat
                     their benchmark, with family-wise error <= alpha across
                     all trials. Studentised, with the stationary bootstrap.
* ``pbo_cscv``       probability of backtest overfitting by combinatorially
                     symmetric cross-validation (Bailey, Borwein, Lopez de
                     Prado & Zhu 2017). Over all C(S, S/2) splits of S
                     contiguous blocks, it is the share in which the
                     in-sample best trial ranks below the median out of
                     sample.
* ``deflated_sharpe`` Bailey & Lopez de Prado (2014). The probability that a
                     trial's true Sharpe exceeds the maximum expected from
                     N_eff independent null trials, adjusted for skew and
                     kurtosis.
* ``effective_trials`` the participation ratio of the trial correlation
                     matrix, (sum lambda)^2 / sum lambda^2.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.stats import norm

from .. import spa

EULER = 0.5772156649015329


def _se(M: np.ndarray, block: float) -> np.ndarray:
    n = M.shape[0]
    return np.sqrt(spa.lrv(M, "sb", block) / n)


def stepm(M: np.ndarray, alpha: float = 0.10, block: float = 20.0, B: int = 2000, seed: int = 0) -> dict:
    """Romano-Wolf StepM on mean > 0, studentised by the fixed long-run s.e.

    The bootstrap distribution of max_k (mean*_k - mean_k) / se_k over the
    still-active set gives the critical value. Every trial above it is
    rejected, rejected trials leave the active set, and this repeats until no
    new rejection occurs.
    """
    M = np.asarray(M, float)
    n, K = M.shape
    mean = M.mean(axis=0)
    se = _se(M, block)
    t = mean / se
    rng = np.random.default_rng(seed)
    boot = spa.bootstrap_means(M, block, B, rng)            # (B, K)
    z = (boot - mean) / se                                   # centred, studentised
    active = np.ones(K, bool)
    rejected = np.zeros(K, bool)
    steps = []
    while active.any():
        c = float(np.quantile(z[:, active].max(axis=1), 1 - alpha))
        new = active & (t > c)
        steps.append({"critical_t": c, "n_rejected": int(new.sum())})
        if not new.any():
            break
        rejected |= new
        active &= ~new
    return {"t": t, "mean": mean, "se": se, "rejected": rejected, "steps": steps,
            "critical_t_first": steps[0]["critical_t"] if steps else None, "alpha": alpha, "B": B}


def _block_stats(M: np.ndarray, S: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = np.linspace(0, M.shape[0], S + 1).astype(int)
    s1 = np.array([M[a:b].sum(axis=0) for a, b in zip(edges[:-1], edges[1:])])        # (S, K)
    s2 = np.array([(M[a:b] ** 2).sum(axis=0) for a, b in zip(edges[:-1], edges[1:])])
    cnt = np.diff(edges).astype(float)
    return s1, s2, cnt


def pbo_cscv(M: np.ndarray, S: int = 16) -> dict:
    """PBO by CSCV with Sharpe as the performance metric."""
    M = np.asarray(M, float)
    n, K = M.shape
    if K < 2:
        raise ValueError("PBO needs at least two trials")
    s1, s2, cnt = _block_stats(M, S)
    idx = np.arange(S)
    lam, is_best_oos_sr, is_best_is_sr, below0 = [], [], [], []
    for c in combinations(idx, S // 2):
        ins = np.zeros(S, bool)
        ins[list(c)] = True
        out = ~ins

        def sharpe(mask):
            N = cnt[mask].sum()
            m = s1[mask].sum(axis=0) / N
            v = s2[mask].sum(axis=0) / N - m ** 2
            return m / np.sqrt(np.maximum(v, 1e-300))
        sr_in, sr_out = sharpe(ins), sharpe(out)
        best = int(np.argmax(sr_in))
        # relative rank of the in-sample best among all trials out of sample (1 = worst)
        rank = 1 + np.sum(sr_out < sr_out[best]) + 0.5 * (np.sum(sr_out == sr_out[best]) - 1)
        w = rank / (K + 1)
        lam.append(np.log(w / (1 - w)))
        is_best_oos_sr.append(sr_out[best])
        is_best_is_sr.append(sr_in[best])
        below0.append(sr_out[best] < 0)
    lam = np.array(lam)
    oos, ins_ = np.array(is_best_oos_sr), np.array(is_best_is_sr)
    slope = float(np.polyfit(ins_, oos, 1)[0]) if np.std(ins_) > 0 else float("nan")
    return {"pbo": float(np.mean(lam <= 0)), "n_splits": int(len(lam)), "S": S, "K": K,
            "median_logit": float(np.median(lam)),
            "p_selected_oos_sharpe_below_0": float(np.mean(below0)),
            "degradation_slope": slope,
            "mean_selected_oos_sharpe_daily": float(oos.mean())}


def effective_trials(M: np.ndarray) -> float:
    """Participation ratio of the eigenvalues of the trial correlation matrix."""
    M = np.asarray(M, float)
    keep = M.std(axis=0) > 0
    C = np.corrcoef(M[:, keep], rowvar=False)
    ev = np.clip(np.linalg.eigvalsh(np.atleast_2d(C)), 0, None)
    return float(ev.sum() ** 2 / (ev ** 2).sum())


def expected_max_sharpe(sr_std: float, N: float) -> float:
    """E[max of N iid null Sharpe ratios] ~ sd x ((1-g) z(1-1/N) + g z(1-1/(N e)))."""
    if N <= 1:
        return 0.0
    return float(sr_std * ((1 - EULER) * norm.ppf(1 - 1 / N) + EULER * norm.ppf(1 - 1 / (N * np.e))))


def deflated_sharpe(x: np.ndarray, sr_trials: np.ndarray, N: float) -> dict:
    """DSR of series x (per-period Sharpe), against the max of N null trials.

    sr_trials holds the per-period Sharpe ratios of all trials; their
    cross-sectional variance estimates Var[SR] under the search.
    """
    x = np.asarray(x, float)
    n = len(x)
    sd = x.std(ddof=1)
    sr = x.mean() / sd if sd > 0 else 0.0
    g3 = float(((x - x.mean()) ** 3).mean() / sd ** 3) if sd > 0 else 0.0
    g4 = float(((x - x.mean()) ** 4).mean() / sd ** 4) if sd > 0 else 3.0
    sr0 = expected_max_sharpe(float(np.std(sr_trials, ddof=1)), N)
    denom = np.sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2, 1e-12))
    z = (sr - sr0) * np.sqrt(n - 1) / denom
    return {"sr_daily": float(sr), "sr_annual": float(sr * np.sqrt(252)), "sr0_daily": sr0,
            "skew": g3, "kurtosis": g4, "N": float(N), "dsr": float(norm.cdf(z)),
            "psr_vs_0": float(norm.cdf(sr * np.sqrt(n - 1) / denom))}
