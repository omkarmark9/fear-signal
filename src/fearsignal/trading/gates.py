"""Selection gates for the trading model.

The rules are declared in research/trading/candidates.md before any backtest:

* dev 2024-01-02 .. 2025-12-31. A dev cohort must also EXIT by 2025-12-31,
  so no 2026 price enters a dev statistic;
* each variant is scored on its daily excess return (net of costs) over the
  benchmark;
* standard errors come from the stationary-bootstrap long-run variance with
  mean block 20 (Politis and Romano 1994);
* each family gets a calibrated Hansen SPA_c test across its variants
  (spa.spa_calibrated, actual level about 6-7% at a nominal 5% on this
  project's data);
* Holm step-down across families at family-wise 0.10, each threshold capped
  at 0.05;
* within a family, the 1-SE rule: take the first variant in declared order
  whose mean excess is within one standard error of the best variant's.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import spa

DEV = (pd.Timestamp("2024-01-02"), pd.Timestamp("2025-12-31"))
HOLDOUT = (pd.Timestamp("2026-01-02"), pd.Timestamp("2026-11-27"))
BLOCK = 20.0
FWER = 0.10
CAP = 0.05


def last_signal_date(dates: pd.DatetimeIndex, H: int, delay: int, end) -> pd.Timestamp:
    """Latest signal session whose cohort exits (open of t+delay+H) on or before ``end``."""
    i_end = int(np.searchsorted(dates, pd.Timestamp(end), side="right")) - 1
    return dates[i_end - delay - H]


def window(x: pd.Series, start, end) -> pd.Series:
    return x[(x.index >= pd.Timestamp(start)) & (x.index <= pd.Timestamp(end))]


def variant_stats(x: pd.Series, start, end, block: float = BLOCK) -> dict:
    """Mean daily excess, its dependence-robust s.e. and t, over [start, end]."""
    s = window(x, start, end).astype(float)
    n = len(s)
    m = float(s.mean())
    se = float(np.sqrt(spa.lrv(s.to_numpy(), "sb", block)[0] / n))
    return {"n_days": n, "mean_daily": m, "se_daily": se, "t": m / se if se > 0 else 0.0,
            "ann_excess": m * 252, "ann_se": se * 252,
            "days_invested": int((s != 0).sum()),
            "pw_block": float(spa.pw_block_length(s.to_numpy())) if n > 30 and s.std() > 0 else None}


def family_spa(D: pd.DataFrame, block: float = BLOCK, B: int = 999, R: int = 999, seed: int = 0) -> dict:
    """Calibrated SPA_c across a family's variants (columns of D, daily excess)."""
    r = spa.spa_calibrated(D.to_numpy(float), block=block, B=B, R=R, seed=seed, names=list(D.columns))
    keep = ("T", "p_c_raw", "p_c_calibrated", "p_c_calibrated_by_world", "alpha_nominal_05", "n", "K")
    out = {k: r[k] for k in keep}
    out["t"] = dict(zip(D.columns, np.asarray(r["t"]).round(3).tolist()))
    return out


def holm(pvals: dict, fwer: float = FWER, cap: float = CAP) -> dict:
    """Holm step-down: the i-th smallest p must be <= min(cap, fwer / (m - i + 1))."""
    m = len(pvals)
    order = sorted(pvals, key=lambda k: pvals[k])
    out, alive = {}, True
    for i, k in enumerate(order):
        thr = min(cap, fwer / (m - i))
        ok = alive and pvals[k] <= thr
        alive = ok
        out[k] = {"p": float(pvals[k]), "threshold": thr, "rank": i + 1, "pass": bool(ok)}
    return out


def one_se_choice(stats: list[dict]) -> int:
    """Index of the first variant (declared order) within one s.e. of the best mean."""
    means = np.array([s["mean_daily"] for s in stats])
    b = int(np.argmax(means))
    floor = means[b] - stats[b]["se_daily"]
    return int(np.flatnonzero(means >= floor)[0])


def ols_cluster(y: np.ndarray, X: np.ndarray, groups: np.ndarray) -> dict:
    """OLS with cluster-robust (CR1) standard errors; clusters = signal days.

    V = (X'X)^-1 [sum_g X_g' e_g e_g' X_g] (X'X)^-1 x G/(G-1) x (n-1)/(n-k).
    """
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    _, gi = np.unique(groups, return_inverse=True)
    G = gi.max() + 1
    S = np.zeros((G, k))
    np.add.at(S, gi, X * e[:, None])
    meat = S.T @ S
    V = XtX_inv @ meat @ XtX_inv * (G / (G - 1)) * ((n - 1) / (n - k))
    se = np.sqrt(np.diag(V))
    return {"b": b, "se": se, "t": b / se, "n": n, "clusters": int(G)}
