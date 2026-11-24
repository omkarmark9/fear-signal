"""Size of the long-horizon tests under the null, at holdout-like sample sizes."""

import numpy as np
import pytest

from fearsignal import evaluate, stats, validate


def _ar1(n, phi, rng):
    x = np.zeros(n)
    e = rng.normal(size=n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


def test_hodrick_1b_has_near_nominal_size_where_overlapping_hac_does_not():
    rng = np.random.default_rng(11)
    reps, n, h = 400, 225, 20
    rej_1b = rej_nw = 0
    for _ in range(reps):
        x, r = _ar1(n, 0.95, rng), rng.normal(0, 0.02, n)
        rej_1b += stats.hodrick_1b(r, x, h)["p"] < 0.05
        fwd = validate._fwd_sum(r, h)
        ok = np.isfinite(fwd)
        res = stats.ols_hac(fwd[ok], np.column_stack([np.ones(ok.sum()), x[ok]]), lags=h)
        rej_nw += res["p"][1] < 0.05
    assert rej_1b / reps < 0.09            # nominal 0.05 (MC s.e. ~0.011)
    assert rej_nw / reps > 0.12            # the naive test over-rejects


def test_diebold_mariano_hln_size():
    rng = np.random.default_rng(12)
    reps, rej = 400, 0
    for _ in range(reps):
        e = rng.normal(size=230)
        d = np.convolve(e, np.ones(5) / 5, mode="valid")[:220]   # MA(4) under the null
        rej += stats.diebold_mariano(d, 5)["p_one_sided"] < 0.05
    assert rej / reps < 0.09


def test_d5_within_blocks_never_straddles_folds():
    s = np.arange(30, dtype=float)
    v = np.arange(30, dtype=float) * 2
    block = np.array([-1] * 5 + [0] * 10 + [1] * 15)
    d = evaluate.d5_within_blocks(s, v, block)
    idx = d.index.to_numpy()
    assert np.all(block[idx] == block[idx - 5]) and np.all(block[idx] >= 0)
    assert len(d) == (10 - 5) + (15 - 5)


def test_clustered_calibration_se_is_wider_under_common_shocks():
    rng = np.random.default_rng(13)
    days, per = 400, 50
    shock = np.repeat(rng.normal(0, 1.0, days), per)
    q = np.full(days * per, 0.04)
    p_true = 1 / (1 + np.exp(-(np.log(0.04 / 0.96) + shock)))
    y = (rng.random(days * per) < p_true).astype(float)
    q = np.clip(p_true * np.exp(rng.normal(0, 0.3, len(q))), 1e-4, 0.9)
    g = np.repeat(np.arange(days), per)
    iid = stats.logistic_calibration(q, y)
    cl = stats.logistic_calibration(q, y, groups=g)
    assert cl["se_slope"] > iid["se_slope"]
