r"""From filtered state probabilities to severity, Z, F and bands.

Severity is the posterior mean of the regime's log-scale, rescaled to [0, 1]::

    p_t = (E[ln s_S | x_<=t] - ln s_0) / (ln s_{K-1} - ln s_0)
        = sum_k P(S_t = k | x_<=t) * ln(s_k/s_0) / ln(s_{K-1}/s_0)

For K = 3 this is exactly the v1 formula p = w P(stressed) + P(panic) with
w = ln(s1/s0)/ln(s2/s0): v1's weight is the stressed regime's position on the
log-volatility axis. The same construction applied to the market HMM uses the
emission means (already on a log scale) in place of ln s_k.

L = logit(p) is mapped to a standard-normal score through a frozen,
monotone quantile map fitted on the calibration window:
Z = Phi^{-1}(F_cal(L)), linearly extrapolated beyond the calibration range
using the slope of the outer 0.5 standard deviations, and clipped to +-6.
F = 100 Phi(Z).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtri, ndtr

from .config import BANDS, Z_CLIP

P_FLOOR = 1e-12


def severity(alpha: np.ndarray, location: np.ndarray) -> np.ndarray:
    """Posterior mean of the regime location, rescaled so calm = 0 and worst = 1."""
    loc = np.asarray(location, float)
    w = (loc - loc[0]) / (loc[-1] - loc[0])
    return alpha @ w


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, P_FLOOR, 1 - P_FLOOR)
    return np.log(p) - np.log1p(-p)


@dataclass
class QuantileMap:
    """Monotone map L -> Z = Phi^{-1}(F_cal(L)) with linear tails."""

    knots_L: np.ndarray
    knots_z: np.ndarray
    slope_lo: float
    slope_hi: float

    @classmethod
    def fit(cls, values: np.ndarray, dz: float = 0.01, tail: float = 0.5) -> "QuantileMap":
        v = np.sort(np.asarray(values, float)[np.isfinite(values)])
        n = len(v)
        zmax = float(ndtri(1 - 0.5 / n))
        z = np.arange(-np.floor(zmax / dz) * dz, zmax + 1e-12, dz)
        L = np.quantile(v, ndtr(z))
        # strictly increasing knots (ties broken by a negligible ramp)
        L = np.maximum.accumulate(L) + np.arange(len(L)) * 1e-12
        lo = z <= z[0] + tail
        hi = z >= z[-1] - tail
        slope_lo = float(np.polyfit(L[lo], z[lo], 1)[0]) if np.ptp(L[lo]) > 0 else 0.0
        slope_hi = float(np.polyfit(L[hi], z[hi], 1)[0]) if np.ptp(L[hi]) > 0 else 0.0
        return cls(L, z, max(slope_lo, 0.0), max(slope_hi, 0.0))

    def __call__(self, L: np.ndarray) -> np.ndarray:
        L = np.asarray(L, float)
        z = np.interp(L, self.knots_L, self.knots_z)
        z = np.where(L < self.knots_L[0], self.knots_z[0] + self.slope_lo * (L - self.knots_L[0]), z)
        z = np.where(L > self.knots_L[-1], self.knots_z[-1] + self.slope_hi * (L - self.knots_L[-1]), z)
        return np.where(np.isfinite(L), np.clip(z, -Z_CLIP, Z_CLIP), np.nan)

    def to_dict(self, decimals: int = 6) -> dict:
        return {
            "knots_L": np.round(self.knots_L, decimals).tolist(),
            "z0": float(self.knots_z[0]),
            "dz": float(np.round(self.knots_z[1] - self.knots_z[0], 10)),
            "slope_lo": self.slope_lo,
            "slope_hi": self.slope_hi,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "QuantileMap":
        L = np.asarray(d["knots_L"], float)
        L = np.maximum.accumulate(L) + np.arange(len(L)) * 1e-12
        z = d["z0"] + d["dz"] * np.arange(len(L))
        return cls(L, z, d["slope_lo"], d["slope_hi"])


def fear_score(z: np.ndarray) -> np.ndarray:
    return 100.0 * ndtr(z)


def band(z: np.ndarray) -> np.ndarray:
    out = np.full(np.shape(z), "", dtype=object)
    for lo, hi, name in BANDS:
        out[(z >= lo) & (z < hi)] = name
    return out


def robust_cross_section_z(Z: np.ndarray, floor: float = 0.25) -> np.ndarray:
    """Stock-versus-others score: (Z - median) / (1.4826 MAD), per day.

    The MAD is floored so that a day on which almost every stock sits at the
    calm floor does not turn a modest reading into an extreme outlier.
    """
    med = np.nanmedian(Z, axis=1, keepdims=True)
    mad = 1.4826 * np.nanmedian(np.abs(Z - med), axis=1, keepdims=True)
    return (Z - med) / np.maximum(mad, floor)
