r"""Market-level fear.

Three readings of the whole AI complex, all causal:

``M_legacy`` (v1)
    z-score of the day's mean stock Z against all earlier days (expanding mean
    and standard deviation, at least ``min_history`` earlier days).

``breadth``
    B_t = (1/N_t) sum_i P_i(S_t != calm | x_<=t): the expected share of
    eligible stocks that are in a stressed or panic regime. Bounded and
    directly interpretable, but it averages many dependent posteriors.

``market HMM`` (v2)
    The cross-section is first collapsed to one robust number per day,
    y_t = ln(max(TM_10%(x_1t, ..., x_Nt), 0.02)), where TM_10% is the 10%
    two-sided trimmed mean of the stocks' drawdown-to-low readings (a single
    stock's earnings collapse is trimmed away; a broad sell-off is not). A
    three-state Gaussian HMM on y_t is fitted by EM and filtered forward. Using
    one scalar per day avoids the main defect of pooling stocks as if they
    were independent given the market regime: on a market sell-off all N
    stocks share the same shock, so an N-fold product of likelihoods would be
    N times over-confident. The market regime has its own persistence, so it
    is a time-series model of the common component rather than an average.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from .features import EXACT, MISSING, Obs

TRIM = 0.10
Y_FLOOR = 0.02
MIN_STOCKS = 20


def legacy_M(Z: np.ndarray, min_history: int = 20) -> np.ndarray:
    """v1 market fear: today's mean Z against the mean/sd of all earlier days."""
    m = _nanmean_rows(Z)
    out = np.full(len(m), np.nan)
    hist = []
    for t, v in enumerate(m):
        if np.isfinite(v) and len(hist) >= min_history:
            h = np.asarray(hist)
            out[t] = (v - h.mean()) / h.std(ddof=1)
        if np.isfinite(v):
            hist.append(v)
    return out


def breadth(alpha: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    stress = 1.0 - alpha[:, :, 0]
    stress = np.where(eligible, stress, np.nan)
    return _nanmean_rows(stress)


def market_observation(obs: Obs) -> Obs:
    """Collapse the cross-section to y_t = ln(max(trimmed mean of x, 0.02))."""
    x = np.where(obs.eligible, obs.x, np.nan)
    n = np.isfinite(x).sum(axis=1)
    tm = np.full(len(x), np.nan)
    for t in np.flatnonzero(n >= MIN_STOCKS):
        tm[t] = stats.trim_mean(x[t][np.isfinite(x[t])], TRIM)
    y = np.log(np.maximum(tm, Y_FLOOR))
    kind = np.where(np.isfinite(y), EXACT, MISSING).astype(np.int8)[:, None]
    return Obs(y[:, None], kind, obs.dates, ["MARKET"])


def _nanmean_rows(a: np.ndarray) -> np.ndarray:
    n = np.isfinite(a).sum(axis=1)
    s = np.nansum(a, axis=1)
    return np.where(n > 0, s / np.maximum(n, 1), np.nan)
