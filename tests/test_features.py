import numpy as np
import pandas as pd
from scipy import stats

from fearsignal import features
from fearsignal.config import MAD_TO_SIGMA, SIGMA_WINDOW
from fearsignal.data import Panel
from fearsignal.features import CENSORED, EXACT, MISSING, ZERO


def _panel(adj, close, low):
    T = len(adj)
    return Panel(pd.bdate_range("2023-01-03", periods=T), ["A"],
                 np.asarray(adj, float)[:, None], np.asarray(close, float)[:, None],
                 np.asarray(low, float)[:, None])


def test_sigma_uses_previous_window_only():
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.02, 300)
    adj = 100 * np.exp(np.concatenate([[0], np.cumsum(r)]))
    f = features.compute(_panel(adj, adj, adj * 0.99))
    t = 200  # row t holds return r[t-1]; the window is returns r[t-1-126 .. t-2]
    expect = MAD_TO_SIGMA * np.median(np.abs(r[t - 1 - SIGMA_WINDOW: t - 1]))
    assert np.isclose(f.sigma[t, 0], expect)
    # the first eligible row has exactly SIGMA_WINDOW earlier returns
    first = np.flatnonzero(np.isfinite(f.sigma[:, 0]))[0]
    assert first == SIGMA_WINDOW + 1
    assert f.obs.kind[first - 1, 0] == MISSING


def test_x_definition_and_kinds():
    rng = np.random.default_rng(1)
    r = rng.normal(0, 0.02, 200)
    adj = 100 * np.exp(np.concatenate([[0], np.cumsum(r)]))
    low = adj * 0.995
    low[150] = adj[149] * np.exp(-0.5)     # a crash far beyond 10 sigma -> censored
    low[160] = adj[159] * 1.01             # never below the previous close -> zero
    f = features.compute(_panel(adj, adj, low))
    t = 170
    expect = max(np.log(adj[t - 1] / low[t]), 0) / f.sigma[t, 0]
    assert np.isclose(f.obs.x[t, 0], min(expect, 10))
    assert f.obs.kind[150, 0] == CENSORED and f.obs.x[150, 0] == 10
    assert f.obs.kind[160, 0] == ZERO and f.obs.x[160, 0] == 0
    assert f.obs.kind[t, 0] in (EXACT, ZERO)


def test_dividend_adjustment_removes_ex_date_artifact():
    """A 5% dividend: the raw low drops 5% below the raw previous close, but the
    adjusted low does not, so no fake drawdown is recorded."""
    T = 140
    close = np.full(T, 100.0) * np.exp(np.random.default_rng(2).normal(0, 0.01, T).cumsum())
    close[135:] *= 0.95                     # ex-date at t=135: raw price falls by the dividend
    factor = np.where(np.arange(T) < 135, 0.95, 1.0)
    adj = close * factor                    # vendor-style back-adjustment
    low = close * 0.999
    f = features.compute(_panel(adj, close, low))
    raw = features.compute(_panel(close, close, low))   # ignoring the adjustment
    gap = np.log(1 / 0.95) / f.sigma[135, 0]            # the dividend, in sigmas
    assert raw.obs.x[135, 0] - f.obs.x[135, 0] > 0.9 * min(gap, 10 - f.obs.x[135, 0])
    assert gap > 3


def test_reflection_principle_half_normal():
    """Driftless Brownian path: max drawdown below the start is |N(0, sigma^2)|."""
    rng = np.random.default_rng(3)
    n_paths, steps = 20000, 2000
    inc = rng.standard_normal((n_paths, steps)) / np.sqrt(steps)
    m = -np.minimum(np.cumsum(inc, axis=1).min(axis=1), 0)
    # discretisation shortens the minimum by ~0.5826/sqrt(steps) (Broadie-Glasserman-Kou)
    m = m + 0.5826 / np.sqrt(steps)
    ks = stats.kstest(m, stats.halfnorm.cdf)
    assert ks.pvalue > 0.01
