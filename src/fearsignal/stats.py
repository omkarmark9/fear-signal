"""Inference tools for dependent data.

Daily signals are serially dependent (and 5-day changes overlap), so naive
i.i.d. standard errors overstate precision. Everything reported uses one of:

* Newey-West (Bartlett kernel) HAC standard errors for means and OLS slopes;
* the stationary bootstrap of Politis and Romano (1994) for correlations and
  other smooth statistics, with mean block length ``block``.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def newey_west_se_mean(x: np.ndarray, lags: int) -> float:
    """HAC standard error of the sample mean."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    u = x - x.mean()
    v = u @ u / n
    for l in range(1, min(lags, n - 1) + 1):
        w = 1.0 - l / (lags + 1.0)
        v += 2.0 * w * (u[l:] @ u[:-l]) / n
    return float(np.sqrt(max(v, 0.0) / n))


def ols_hac(y: np.ndarray, X: np.ndarray, lags: int) -> dict:
    """OLS with Newey-West covariance. X should include a constant column."""
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    y, X = y[ok], X[ok]
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    e = y - X @ beta
    g = X * e[:, None]
    S = g.T @ g / n
    for l in range(1, lags + 1):
        w = 1.0 - l / (lags + 1.0)
        G = g[l:].T @ g[:-l] / n
        S += w * (G + G.T)
    cov = n * XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(cov))
    t = beta / se
    p = 2 * stats.t.sf(np.abs(t), df=n - k)
    r2 = 1 - (e @ e) / ((y - y.mean()) @ (y - y.mean()))
    return {"beta": beta, "se": se, "t": t, "p": p, "n": n, "r2": float(r2)}


def hodrick_1b(r: np.ndarray, x: np.ndarray, horizon: int, lags: int = 0) -> dict:
    """Hodrick (1992) "1B" test of long-horizon predictability.

    The long-horizon regression of sum_{j=1}^{h} r_{t+j} on x_t has
    overlapping errors; HAC tests built on it over-reject badly in samples of a
    few hundred days. Under the null of no predictability the slope is zero
    iff the *reverse* regression of the one-period return r_{t+1} on the
    rolled-up regressor X_t = sum_{j=0}^{h-1} x_{t-j} has zero slope, and that
    regression has no overlap in its errors. Under the null r_{t+1} is a
    martingale difference, so heteroskedasticity-robust (lags = 0) errors
    suffice; simulated size at n = 225 is about 5-6% against 22% for a
    Newey-West(h) test on the overlapping regression (tests/test_stats_size.py).
    """
    r = np.asarray(r, float)
    x = np.asarray(x, float)
    X = pd_rolling_sum(x, horizon)
    y = np.full(len(r), np.nan)
    y[:-1] = r[1:]
    D = np.column_stack([np.ones(len(r)), X])
    res = ols_hac(y, D, lags=lags)
    return {"t": float(res["t"][1]), "p": float(res["p"][1]), "n": res["n"]}


def pd_rolling_sum(x: np.ndarray, h: int) -> np.ndarray:
    """sum_{j=0}^{h-1} x_{t-j}; NaN unless all h values are finite."""
    import pandas as pd

    return pd.Series(np.asarray(x, float)).rolling(h, min_periods=h).sum().to_numpy()


def diebold_mariano(d: np.ndarray, horizon: int) -> dict:
    """DM test of E[d] = 0 for h-step-ahead loss differentials, with the
    Harvey-Leybourne-Newbold (1997) small-sample correction.

    Under the null, d_t is at most MA(h-1), so the long-run variance uses
    autocovariances up to lag h-1 with a rectangular kernel; the statistic is
    scaled by sqrt((n + 1 - 2h + h(h-1)/n) / n) and referred to t(n-1).
    Falls back to the Bartlett kernel if the rectangular estimate is not
    positive.
    """
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    n = len(d)
    u = d - d.mean()
    gam = [u @ u / n] + [u[k:] @ u[:-k] / n for k in range(1, horizon)]
    v = gam[0] + 2 * sum(gam[1:])
    if v <= 0:
        v = gam[0] + 2 * sum((1 - k / horizon) * g for k, g in enumerate(gam[1:], start=1))
    dm = d.mean() / np.sqrt(v / n)
    adj = np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    t = float(dm * adj)
    return {"mean": float(d.mean()), "t": t, "p_one_sided": float(stats.t.sf(t, df=n - 1)), "n": n}


def stationary_bootstrap_indices(n: int, block: float, rng) -> np.ndarray:
    """One stationary-bootstrap resample of range(n) (geometric block lengths)."""
    idx = np.empty(n, dtype=int)
    p = 1.0 / block
    i = rng.integers(n)
    for t in range(n):
        if t > 0 and rng.random() < p:
            i = rng.integers(n)
        idx[t] = i
        i = (i + 1) % n
    return idx


def corr_ci(a: np.ndarray, b: np.ndarray, method: str = "pearson", block: float = 20.0,
            n_boot: int = 2000, level: float = 0.95, seed: int = 0) -> dict:
    """Correlation with a stationary-bootstrap percentile interval."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    f = (lambda u, v: stats.pearsonr(u, v)[0]) if method == "pearson" else (
        lambda u, v: stats.spearmanr(u, v)[0])
    est = float(f(a, b))
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for j in range(n_boot):
        ix = stationary_bootstrap_indices(len(a), block, rng)
        boots[j] = f(a[ix], b[ix])
    lo, hi = np.quantile(boots, [(1 - level) / 2, (1 + level) / 2])
    return {"r": est, "lo": float(lo), "hi": float(hi), "n": int(len(a))}


def corr_diff_ci(a: np.ndarray, b: np.ndarray, v: np.ndarray, method: str = "spearman",
                 block: float = 20.0, n_boot: int = 2000, level: float = 0.95, seed: int = 0) -> dict:
    """corr(a, v) - corr(b, v) with a paired stationary-bootstrap interval."""
    a, b, v = (np.asarray(z, float) for z in (a, b, v))
    ok = np.isfinite(a) & np.isfinite(b) & np.isfinite(v)
    a, b, v = a[ok], b[ok], v[ok]
    f = (lambda u, w: stats.spearmanr(u, w)[0]) if method == "spearman" else (
        lambda u, w: stats.pearsonr(u, w)[0])
    est = float(f(a, v) - f(b, v))
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for j in range(n_boot):
        ix = stationary_bootstrap_indices(len(a), block, rng)
        boots[j] = f(a[ix], v[ix]) - f(b[ix], v[ix])
    lo, hi = np.quantile(boots, [(1 - level) / 2, (1 + level) / 2])
    return {"diff": est, "lo": float(lo), "hi": float(hi), "n": int(len(a))}


def logistic_calibration(q: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None,
                         iters: int = 50) -> dict:
    """Fit y ~ Bernoulli(sigmoid(a + b logit q)). Calibrated forecasts give a=0, b=1.

    With ``groups`` (e.g. the day of each stock-day), standard errors are
    cluster-robust: stock-days on the same date share one market shock and
    are not independent evidence.
    """
    q = np.clip(np.asarray(q, float), 1e-9, 1 - 1e-9)
    y = np.asarray(y, float)
    X = np.column_stack([np.ones_like(q), np.log(q / (1 - q))])
    beta = np.array([0.0, 1.0])
    for _ in range(iters):
        eta = X @ beta
        mu = 1 / (1 + np.exp(-eta))
        W = mu * (1 - mu)
        H = X.T @ (X * W[:, None])
        g = X.T @ (y - mu)
        step = np.linalg.solve(H, g)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10:
            break
    Hinv = np.linalg.inv(H)
    cov = Hinv
    if groups is not None:
        mu = 1 / (1 + np.exp(-(X @ beta)))
        score = X * (y - mu)[:, None]
        _, gidx = np.unique(np.asarray(groups), return_inverse=True)
        G = np.zeros((gidx.max() + 1, 2))
        np.add.at(G, gidx, score)
        cov = Hinv @ (G.T @ G) @ Hinv
    return {"intercept": float(beta[0]), "slope": float(beta[1]),
            "se_intercept": float(np.sqrt(cov[0, 0])), "se_slope": float(np.sqrt(cov[1, 1]))}


def reliability_table(q: np.ndarray, y: np.ndarray, bins=(0, .005, .01, .02, .05, .1, .2, .5, 1)) -> list[dict]:
    q = np.asarray(q, float)
    y = np.asarray(y, float)
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (q >= lo) & (q < hi)
        if m.sum():
            out.append({"bin": f"[{lo:g},{hi:g})", "n": int(m.sum()),
                        "mean_forecast": float(q[m].mean()), "observed": float(y[m].mean())})
    return out
