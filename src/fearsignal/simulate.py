"""Simulation from the observation model, used by tests and power checks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import X_CLIP, ZERO_EPS
from .features import CENSORED, EXACT, MISSING, ZERO, Obs


def simulate_states(A: np.ndarray, init: np.ndarray, T: int, N: int, rng) -> np.ndarray:
    K = A.shape[0]
    S = np.empty((T, N), dtype=int)
    S[0] = rng.choice(K, size=N, p=init)
    cum = np.cumsum(A, axis=1)
    for t in range(1, T):
        u = rng.random(N)
        S[t] = (u[:, None] > cum[S[t - 1]]).sum(axis=1)
    return S


def simulate_halfnormal(s, A, pi, T: int, N: int, rng, init=None,
                        eps: float = ZERO_EPS, clip: float = X_CLIP):
    """Draw (Obs, states) from the zero-inflated half-normal HMM."""
    from .hmm import stationary

    s = np.asarray(s, float)
    A = np.asarray(A, float)
    init = stationary(A) if init is None else np.asarray(init)
    S = simulate_states(A, init, T, N, rng)
    x = np.abs(rng.standard_normal((T, N))) * s[S]
    if pi is not None:
        atom = rng.random((T, N)) < np.asarray(pi)[S]
        x[atom] = 0.0
    kind = np.full((T, N), EXACT, dtype=np.int8)
    kind[x <= eps] = ZERO
    kind[x >= clip] = CENSORED
    x = np.minimum(x, clip)
    dates = pd.bdate_range("2023-01-03", periods=T)
    return Obs(x, kind, dates, [f"S{i}" for i in range(N)]), S


def drop_random(obs: Obs, frac: float, rng) -> Obs:
    """Mark a random fraction of observations MISSING."""
    kind = obs.kind.copy()
    kind[rng.random(kind.shape) < frac] = MISSING
    x = np.where(kind == MISSING, np.nan, obs.x)
    return Obs(x, kind, obs.dates, obs.tickers)
