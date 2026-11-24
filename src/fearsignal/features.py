r"""Observation construction: how far below yesterday's close did the stock trade?

For stock i on session t (previous valid session t-1)::

    r_t     = ln(C^adj_t / C^adj_{t-1})                         close-to-close log return
    sigma_t = 1.4826 * median(|r_{t-126}|, ..., |r_{t-1}|)       robust scale, today excluded
    d_t     = max(ln(C^adj_{t-1} / L^adj_t), 0) / sigma_t        drawdown-to-low in own sigmas
    x_t     = min(d_t, 10)

Why a half-normal fits x (reflection principle). If log price is a driftless
Brownian motion with daily variance sigma^2, the running minimum over one day,
M = -min_{u<=1} B_u, satisfies P(M > m) = 2 P(B_1 < -m), i.e. M is exactly
half-normal with scale sigma. So in a calm regime x should be half-normal with
scale close to 1, and stress shows up as a larger scale. The fitted calm scale
(about 0.95) agrees with that benchmark.

1.4826 = 1/Phi^{-1}(3/4) makes the median of |r| consistent for sigma under
normality; the median tolerates up to half the window being stress days, so
"normal daily move" is not inflated by the very episodes being measured.
Asymptotic relative s.e. of the estimate is about 1.17/sqrt(126) = 10%.

Observation kinds used by the likelihood:

    MISSING  no bar, scale not yet defined (fewer than 126 in-era returns),
             or scale below the stale-data floor
    ZERO     x <= 0.001: the low never went (materially) below the previous close
    EXACT    0.001 < d < 10
    CENSORED d >= 10: only "at least 10 sigma" is retained
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import MAD_TO_SIGMA, SIGMA_FLOOR, SIGMA_WINDOW, X_CLIP, ZERO_EPS
from .data import Panel

MISSING, ZERO, EXACT, CENSORED = 0, 1, 2, 3


@dataclass
class Obs:
    """HMM observations on a (T x N) grid."""

    x: np.ndarray       # clipped drawdown-to-low in sigmas; NaN where MISSING
    kind: np.ndarray    # int8 observation kind
    dates: pd.DatetimeIndex
    tickers: list[str]

    @property
    def eligible(self) -> np.ndarray:
        return self.kind != MISSING

    def slice_time(self, end) -> "Obs":
        m = self.dates <= pd.Timestamp(end)
        return Obs(self.x[m], self.kind[m], self.dates[m], self.tickers)


@dataclass
class Features:
    obs: Obs
    sigma: np.ndarray   # robust daily scale (T x N)
    r: np.ndarray       # close-to-close log return (T x N)
    d: np.ndarray       # unclipped drawdown-to-low in sigmas


def compute(panel: Panel, window: int = SIGMA_WINDOW) -> Features:
    T, N = panel.adj_close.shape
    adj_low = panel.adj_low
    r = np.full((T, N), np.nan)
    dd = np.full((T, N), np.nan)       # ln(prev adj close / adj low), unscaled
    sigma = np.full((T, N), np.nan)
    for j in range(N):
        valid = np.flatnonzero(np.isfinite(panel.adj_close[:, j]) & np.isfinite(adj_low[:, j]))
        if len(valid) < 2:
            continue
        c = panel.adj_close[valid, j]
        prev, cur = c[:-1], c[1:]
        rows = valid[1:]
        r[rows, j] = np.log(cur / prev)
        dd[rows, j] = np.log(prev / adj_low[rows, j])
        # rolling median over the previous `window` returns (today excluded)
        absr = pd.Series(np.abs(np.log(cur / prev)))
        med = absr.rolling(window, min_periods=window).median().shift(1).to_numpy()
        sigma[rows, j] = MAD_TO_SIGMA * med
    with np.errstate(invalid="ignore", divide="ignore"):
        d = np.maximum(dd, 0.0) / sigma
    ok = np.isfinite(d) & np.isfinite(sigma) & (sigma >= SIGMA_FLOOR)
    kind = np.full((T, N), MISSING, dtype=np.int8)
    kind[ok & (d <= ZERO_EPS)] = ZERO
    kind[ok & (d > ZERO_EPS) & (d < X_CLIP)] = EXACT
    kind[ok & (d >= X_CLIP)] = CENSORED
    x = np.where(ok, np.minimum(d, X_CLIP), np.nan)
    return Features(Obs(x, kind, panel.dates, list(panel.tickers)), sigma, r, d)
