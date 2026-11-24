"""Model specifications, fitting, and the full signal computation."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from . import hmm, market, scoring
from .features import CENSORED, EXACT, ZERO, Obs

# The v1 starting point, exactly as published (rows renormalised: they were
# printed to three decimals and sum to 0.999-1.001).
V1_S = np.array([0.95, 1.58, 3.82])
V1_A = np.array([[0.967, 0.011, 0.022], [0.033, 0.916, 0.052], [0.263, 0.319, 0.417]])
V1_A = V1_A / V1_A.sum(axis=1, keepdims=True)

TAIL_THRESHOLD = 3.0   # the forecast event: tomorrow's low >= 3 own-sigmas below today's close


@dataclass(frozen=True)
class StockSpec:
    name: str
    K: int
    zero_mode: str          # "none" | "shared" | "state"
    censor: bool
    fixed: bool = False     # True: use the published v1 parameters, no fitting


CANDIDATES = (
    StockSpec("S0_v1_published", 3, "none", False, fixed=True),
    StockSpec("S1_v1_refit", 3, "none", False),
    StockSpec("S2_shared_K3", 3, "shared", True),
    StockSpec("S3_state_K3", 3, "state", True),
    StockSpec("S4_state_K2", 2, "state", True),
    StockSpec("S5_state_K4", 4, "state", True),
)


# Generic EM starting points. They are round numbers chosen from theory, not
# estimates: the calm scale is 1 by the reflection principle, and stress and
# panic are "somewhat" and "much" wider. v1's published values are
# deliberately not used, because they were estimated on 2023-2025 and would
# leak full-sample information into the walk-forward folds.
_START_S = {2: (1.0, 2.2), 3: (1.0, 1.6, 4.0), 4: (0.9, 1.35, 2.3, 4.6)}


def _base_scales(K: int) -> np.ndarray:
    return np.array(_START_S[K], float)


def _base_A(K: int) -> np.ndarray:
    """Persistent ordinary regimes, a short-lived top regime."""
    A = np.full((K, K), 0.06 / (K - 1)) + np.eye(K) * 0.94
    A[-1] = np.full(K, 0.6 / (K - 1))
    A[-1, -1] = 0.4
    return A / A.sum(axis=1, keepdims=True)


def zero_rate(obs: Obs) -> float:
    k = obs.kind[obs.eligible]
    return float((k == ZERO).mean())


def starting_points(spec: StockSpec, obs: Obs, n_random: int = 3, seed: int = 0) -> list[hmm.HMM]:
    rng = np.random.default_rng(seed)
    z = zero_rate(obs)
    out = []
    for r in range(n_random + 1):
        s = _base_scales(spec.K) * (np.exp(rng.normal(0, 0.15, spec.K)) if r else 1.0)
        A = _base_A(spec.K)
        if r:
            A = np.array([rng.dirichlet(200 * row + 0.5) for row in A])
        pi = None
        if spec.zero_mode != "none":
            jitter = rng.normal(0, 0.03, spec.K) if r else np.zeros(spec.K)
            pi = np.clip(np.full(spec.K, z) + jitter, 0.01, 0.9)
            if spec.zero_mode == "shared":
                pi = np.full(spec.K, pi.mean())
        out.append(hmm.HMM(hmm.HalfNormalEmission(np.sort(s), pi, spec.zero_mode, spec.censor), A))
    return out


def fit_stock_model(spec: StockSpec, obs: Obs, n_random: int = 3, seed: int = 0,
                    max_iter: int = 1500, tol: float = 1e-9) -> hmm.HMM:
    if spec.fixed:
        m = hmm.HMM(hmm.HalfNormalEmission(V1_S.copy(), None, "none", False), V1_A.copy())
        m.n_obs = int(obs.eligible.sum())
        return m
    return hmm.fit(starting_points(spec, obs, n_random, seed), obs, max_iter=max_iter, tol=tol)


def as_scoring_model(model: hmm.HMM, train_zero_rate: float) -> hmm.HMM:
    """Express any candidate in the common censored, zero-inflated likelihood.

    A "none" model (v1) does not model no-dip days; its natural completion is
    a state-independent atom equal to the training no-dip rate, which leaves its
    filter unchanged (see hmm.HalfNormalEmission). Its clip value is scored by
    the survival function, as for every other candidate.
    """
    e = model.emission
    if e.zero_mode == "none":
        e = replace(e, zero_mode="shared", pi=np.full(e.K, train_zero_rate), censor=True)
    elif not e.censor:
        e = replace(e, censor=True)
    return replace(model, emission=e)


def tail_forecast(model: hmm.HMM, fr: hmm.FilterResult, train_zero_rate: float,
                  threshold: float = TAIL_THRESHOLD) -> np.ndarray:
    """q_t = P(x_t >= threshold | x_<t): the forecast made at the close of t-1.

    Uses the one-step predictive state law (alpha_{t-1} A) and each state's
    tail probability (1 - pi_k) 2 Phi(-threshold / s_k).
    """
    e = as_scoring_model(model, train_zero_rate).emission
    return fr.pred @ e.tail_prob(threshold)


def tail_event(obs: Obs, threshold: float = TAIL_THRESHOLD) -> np.ndarray:
    y = ((obs.kind == EXACT) & (obs.x >= threshold)) | (obs.kind == CENSORED)
    return np.where(obs.eligible, y.astype(float), np.nan)


@dataclass
class Signal:
    alpha: np.ndarray        # filtered state probabilities (T,N,K), NaN when ineligible
    severity: np.ndarray     # (T,N)
    L: np.ndarray            # logit severity
    Z: np.ndarray            # standard-normal stock fear
    F: np.ndarray            # 0-100
    I: np.ndarray            # robust cross-sectional z of Z
    q_next: np.ndarray       # P(x_{t+1} >= 3 | x_<=t), forecast made at t
    breadth: np.ndarray      # expected share of stocks not calm
    breadth_L: np.ndarray    # logit breadth
    M: np.ndarray            # market fear, standard-normal scale (v2)
    M_legacy: np.ndarray     # v1 expanding z-score of mean Z


def compute_signal(model: hmm.HMM, obs: Obs, z_map: scoring.QuantileMap | None,
                   m_map: scoring.QuantileMap | None, train_zero_rate: float) -> Signal:
    fr = model.filter(obs)
    el = obs.eligible
    alpha = np.where(el[:, :, None], fr.alpha, np.nan)
    sev = scoring.severity(alpha, np.log(model.emission.s))
    L = np.where(el, scoring.logit(sev), np.nan)
    Z = z_map(L) if z_map is not None else np.full_like(L, np.nan)
    with np.errstate(all="ignore"):
        I = scoring.robust_cross_section_z(Z)
    e = as_scoring_model(model, train_zero_rate).emission
    q_next = np.where(el, (np.nan_to_num(alpha) @ model.A) @ e.tail_prob(TAIL_THRESHOLD), np.nan)
    B = market.breadth(fr.alpha, el)
    BL = scoring.logit(B)
    M = m_map(BL) if m_map is not None else np.full_like(BL, np.nan)
    return Signal(alpha, sev, L, Z, scoring.fear_score(Z), I, q_next, B, BL, M, market.legacy_M(Z))


def fit_maps(model: hmm.HMM, obs: Obs) -> tuple[scoring.QuantileMap, scoring.QuantileMap]:
    """Quantile maps for stock Z (pooled stock-days) and market M (days)."""
    s = compute_signal(model, obs, None, None, zero_rate(obs))
    z_map = scoring.QuantileMap.fit(s.L[obs.eligible])
    bl = s.breadth_L[np.isfinite(s.breadth_L)]
    m_map = scoring.QuantileMap.fit(bl, dz=0.02)
    return z_map, m_map
