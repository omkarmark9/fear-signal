"""Risk forecasts from the fear model, for sizing and for judging option prices.

Regime-adjusted forward volatility. In regime k the day's low falls a
half-normal distance below the previous close with scale s_k own-sigmas. By
the reflection principle that scale is the day's return standard deviation in
own-sigma units. With filtered probabilities p_t and transition matrix A,
the probability of regime k at t+j is (p_t A^j)_k. So the H-session variance
is

    sigma_hat(H)^2 = sigma_t^2 * sum_{j=1..H} (p_t A^j) . s^2

and the annualised average forward volatility is
sigma_t * sqrt(252 * sigma_hat(H)^2 / (H sigma_t^2)).
"""

from __future__ import annotations

import numpy as np

ANN = 252


def forward_var_units(p: np.ndarray, A: np.ndarray, s: np.ndarray, H: int) -> np.ndarray:
    """sum_{j=1..H} (p A^j) . s^2 for each row of p (N x K), in own-sigma^2 units."""
    p = np.atleast_2d(np.asarray(p, float))
    s2 = np.asarray(s, float) ** 2
    out = np.zeros(p.shape[0])
    q = p.copy()
    for _ in range(H):
        q = q @ A
        out += q @ s2
    return out


def forward_vol(sigma: np.ndarray, p: np.ndarray, A: np.ndarray, s: np.ndarray, H: int) -> dict:
    """H-session return s.d. and the annualised average forward volatility."""
    u = forward_var_units(p, A, s, H)
    sig = np.asarray(sigma, float)
    hvol = sig * np.sqrt(u)
    ann = sig * np.sqrt(ANN * u / H)
    return {"h_vol": hvol, "ann_vol": ann}
