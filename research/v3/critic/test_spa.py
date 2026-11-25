"""Unit tests for spa.py. Run from the scratch dir:
    /home/user/fear-signal/.venv/bin/python -m pytest -q test_spa.py
"""
import numpy as np
import pytest
from scipy import stats as sst

import spa


def test_indices_shape_range_and_block_length():
    rng = np.random.default_rng(1)
    idx = spa.sb_indices(500, 10.0, 400, rng)
    assert idx.shape == (400, 500)
    assert idx.min() >= 0 and idx.max() < 500
    # block starts are where idx[t] != idx[t-1] + 1 (mod n); mean block length ~ 10
    cont = (idx[:, 1:] == (idx[:, :-1] + 1) % 500)
    mean_len = 1.0 / (1.0 - cont.mean())
    assert abs(mean_len - 10.0) < 0.4
    # uniform marginal
    counts = np.bincount(idx.ravel(), minlength=500)
    assert counts.std() / counts.mean() < 0.15


def test_iid_block_one_is_iid_bootstrap():
    rng = np.random.default_rng(2)
    idx = spa.sb_indices(300, 1.0, 50, rng)
    cont = (idx[:, 1:] == (idx[:, :-1] + 1) % 300).mean()
    assert cont < 0.01  # only chance continuations (prob 1/n)


def test_counts_sum_to_n():
    rng = np.random.default_rng(3)
    c = spa.sb_counts(250, 7.0, 30, rng)
    assert np.all(c.sum(axis=1) == 250)


def test_sb_kernel_equals_bootstrap_variance():
    """Politis-Romano: the 'sb' kernel is the exact bootstrap variance of sqrt(n) mean."""
    rng = np.random.default_rng(4)
    n = 200
    e = rng.standard_normal((n, 3))
    x = np.empty_like(e)
    x[0] = e[0]
    for t in range(1, n):
        x[t] = 0.5 * x[t - 1] + e[t]
    analytic = spa.lrv(x, "sb", block=8.0)
    boot = spa.bootstrap_means(x, 8.0, 200_000, np.random.default_rng(5))
    empirical = n * boot.var(axis=0)
    assert np.allclose(empirical, analytic, rtol=0.02)


def test_nw_matches_repo_formula():
    import sys
    sys.path.insert(0, "/home/user/fear-signal/src")
    from fearsignal import stats as fs
    rng = np.random.default_rng(6)
    x = rng.standard_normal(380).cumsum() * 0.05 + rng.standard_normal(380)
    se_repo = fs.newey_west_se_mean(x, 10)
    se_here = np.sqrt(spa.lrv(x, "nw", nw_lags=10)[0] / 380)
    assert abs(se_repo - se_here) < 1e-12


def test_statistic_by_hand():
    rng = np.random.default_rng(7)
    d = rng.standard_normal((300, 3)) + np.array([0.0, 0.2, -0.5])
    r = spa.spa(d, block=1.0, B=500, variance="nw", nw_lags=0)
    se = d.std(axis=0, ddof=0) / np.sqrt(300)
    t = d.mean(axis=0) / se
    assert np.isclose(r["T"], max(0, t.max()))
    assert np.allclose(r["t"], t)


def test_pvalue_ordering_and_all_negative():
    rng = np.random.default_rng(8)
    d = rng.standard_normal((380, 5)) * 0.5 + np.array([0.02, -0.01, -0.2, -0.3, 0.0])
    r = spa.spa(d, B=2000)
    assert r["p_l"] <= r["p_c"] + 1e-12 <= r["p_u"] + 2e-12
    neg = -np.abs(rng.standard_normal((380, 4))) - 0.1
    r2 = spa.spa(neg, B=500)
    assert r2["T"] == 0.0 and r2["p_c"] == 1.0 and r2["p_u"] == 1.0


def test_single_candidate_is_bootstrap_t_test():
    """K = 1: SPA_u p-value ~ one-sided normal p-value of the HAC t-statistic."""
    rng = np.random.default_rng(9)
    d = rng.standard_normal(2000) + 0.05
    r = spa.spa(d, block=1.0, B=20_000, variance="nw", nw_lags=0, seed=1)
    p_norm = 1 - sst.norm.cdf(r["t"][0])
    assert abs(r["p_u"] - p_norm) < 0.01


def test_consistent_recentring_ignores_very_bad_candidates():
    """A hopeless candidate must not make SPA_c conservative: p_c with it ~= p without it."""
    rng = np.random.default_rng(10)
    good = rng.standard_normal((380, 1)) + 0.12
    bad = rng.standard_normal((380, 3)) - 1.0
    r_with = spa.spa(np.hstack([good, bad]), block=1.0, B=10_000, seed=3)
    r_without = spa.spa(good, block=1.0, B=10_000, seed=3)
    assert abs(r_with["p_c"] - r_without["p_c"]) < 0.01
    assert r_with["n_very_bad"] == 3


def test_functional_matches_mean_version():
    rng = np.random.default_rng(11)
    d = rng.standard_normal((300, 2)) + np.array([0.15, 0.0])
    rf = spa.spa_functional(lambda ix: d[ix].mean(axis=0), 300, block=5.0, B=3000, seed=0)
    rm = spa.spa(d, block=5.0, B=3000, variance="boot", seed=0)
    assert abs(rf["p_c"] - rm["p_c"]) < 0.03
    assert np.allclose(rf["dbar"], rm["dbar"])


def test_mcs_drops_clearly_worse_model():
    rng = np.random.default_rng(12)
    n = 400
    base = rng.standard_normal((n, 1))
    L = base + 0.3 * rng.standard_normal((n, 4)) + np.array([0, 0.01, 0.0, 0.3])
    r = spa.mcs(L, alpha=0.10, block=1.0, B=4000, names=list("abcd"))
    assert "d" not in r["included"]
    assert r["elimination_order"][0] == "d"
    assert all(m in r["included"] for m in "ac")


def test_lrv_batch_matches_lrv():
    rng = np.random.default_rng(13)
    x = rng.standard_normal((3, 150, 2)).cumsum(axis=1) * 0.1 + rng.standard_normal((3, 150, 2))
    for meth in ("sb", "nw"):
        b = spa.lrv_batch(x, meth, block=7.0)
        for r in range(3):
            assert np.allclose(b[r], spa.lrv(x[r], meth, block=7.0))


def test_replicate_studentisation_ordering_and_single_candidate():
    rng = np.random.default_rng(14)
    d = rng.standard_normal((380, 5)) * 0.5 + np.array([0.03, -0.01, -0.2, -0.3, 0.0])
    r = spa.spa(d, B=2000, studentize="replicate")
    assert r["p_l"] <= r["p_c"] + 1e-12 <= r["p_u"] + 2e-12
    # K = 1, iid symmetric data: bootstrap-t p ~ normal p
    x = rng.standard_normal(1500) + 0.06
    r1 = spa.spa(x, block=1.0, B=20_000, studentize="replicate", variance="nw", nw_lags=0, seed=2)
    assert abs(r1["p_u"] - (1 - sst.norm.cdf(r1["t"][0]))) < 0.012


def test_replicate_corrects_skewness():
    """Left-skewed differentials (rare large losses): fixed studentisation gives a
    smaller (too liberal) upper-tail p-value than the bootstrap-t."""
    rng = np.random.default_rng(15)
    n = 380
    x = np.where(rng.random(n) < 0.02, -8.0, 0.0) + rng.standard_normal(n) * 0.3
    x = x - x.mean() + 0.02
    pf = spa.spa(x, block=1.0, B=20_000, variance="nw", nw_lags=0, seed=1)["p_c"]
    pr = spa.spa(x, block=1.0, B=20_000, variance="nw", nw_lags=0, seed=1, studentize="replicate")["p_c"]
    assert pr > pf


def test_pw_block_length():
    rng = np.random.default_rng(16)
    e = rng.standard_normal(3000)
    x = np.empty_like(e)
    x[0] = e[0]
    for t in range(1, len(e)):
        x[t] = 0.6 * x[t - 1] + e[t]
    assert spa.pw_block_length(e) < 3
    assert spa.pw_block_length(x) > 8


def test_calibrated_versions_sanity():
    rng = np.random.default_rng(17)
    good = rng.standard_normal((300, 2)) + np.array([0.35, 0.0])
    r = spa.spa_calibrated(good, block=1.0, B=199, R=99, seed=1)
    assert r["p_c_raw"] < 0.01 and r["p_c_calibrated"] < 0.05
    bad = rng.standard_normal((300, 2)) - 0.3
    r = spa.spa_calibrated(bad, block=1.0, B=199, R=49, seed=1)
    assert r["p_c_raw"] == 1.0 and r["p_c_calibrated"] == 1.0
    x = rng.standard_normal((200, 2)) + np.array([0.4, 0.0])
    rf = spa.spa_functional_calibrated(lambda ix: x[ix].mean(axis=0), 200, block=1.0, B=99, R=49, seed=2)
    assert rf["p_c_raw"] < 0.02 and rf["p_c_calibrated"] < 0.1


def test_calibrated_null_pvalue_roughly_uniform():
    """Under an iid null the prepivoted p-value should not be wildly off uniform."""
    rng = np.random.default_rng(18)
    ps = [spa.spa_calibrated(rng.standard_normal((150, 2)), block=1.0, B=99, R=49,
                             seed=int(rng.integers(1e9)))["p_c_calibrated"] for _ in range(60)]
    assert 0.0 <= np.mean(np.array(ps) < 0.2) <= 0.4


def test_episode_robustness_flags_single_day_gain():
    d = np.zeros(300) + np.random.default_rng(19).standard_normal(300) * 0.01
    d[100] = 5.0      # the whole "gain" is one day
    r = spa.episode_robustness(d)
    assert r["share_ss_top"] > 0.9
    assert r["drop_one"][0]["mean"] < 0.01
