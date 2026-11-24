import numpy as np
from scipy import stats as sst

from fearsignal import hmm, market, model as mdl, scoring, stats
from fearsignal.features import EXACT, MISSING, Obs
from fearsignal.simulate import simulate_halfnormal


def test_severity_equals_v1_weighting():
    rng = np.random.default_rng(0)
    a = rng.dirichlet([1, 1, 1], size=50)
    s = mdl.V1_S
    w = np.log(s[1] / s[0]) / np.log(s[2] / s[0])
    np.testing.assert_allclose(scoring.severity(a, np.log(s)), w * a[:, 1] + a[:, 2], atol=1e-14)
    # Published w = 0.367; the rounded published scales give 0.3656. The gap is
    # within the rounding of s (each s_k is known only to +-0.005).
    assert abs(w - 0.367) < 2e-3


def test_quantile_map_standardises_calibration_sample():
    rng = np.random.default_rng(1)
    v = rng.gamma(2.0, 1.5, 50000) - 6
    qm = scoring.QuantileMap.fit(v)
    z = qm(v)
    assert abs(z.mean()) < 0.01 and abs(z.std() - 1) < 0.01
    assert sst.kstest(z, "norm").statistic < 0.01
    grid = np.linspace(-20, 20, 2001)
    assert np.all(np.diff(qm(grid)) >= 0)                 # monotone, incl. extrapolation
    assert qm(np.array([1e6]))[0] == 6
    assert qm(np.array([-1e6]))[0] == qm.knots_z[0]      # lower tail held flat
    qm2 = scoring.QuantileMap.from_dict(qm.to_dict())
    np.testing.assert_allclose(qm2(grid), qm(grid), atol=1e-5)


def test_band_labels():
    z = np.array([-1, 0.99, 1.0, 1.6, 2.2, 2.5, 7])
    assert list(scoring.band(z)) == ["calm", "calm", "uneasy", "stressed", "acute fear", "panic", "panic"]


def test_legacy_M_matches_definition():
    rng = np.random.default_rng(2)
    Z = rng.normal(size=(60, 5))
    Z[10, :] = np.nan
    M = market.legacy_M(Z, min_history=20)
    m = market._nanmean_rows(Z)
    hist = [v for v in m[:40] if np.isfinite(v)]
    expect = (m[40] - np.mean(hist)) / np.std(hist, ddof=1)
    assert np.isclose(M[40], expect)
    assert np.isnan(M[10]) and np.all(np.isnan(M[:20]))


def test_breadth_is_mean_non_calm_probability():
    rng = np.random.default_rng(3)
    a = rng.dirichlet([2, 1, 1], size=(30, 8))
    el = rng.random((30, 8)) > 0.2
    B = market.breadth(a, el)
    for t in range(30):
        if el[t].any():
            assert np.isclose(B[t], (1 - a[t, el[t], 0]).mean())


def test_market_observation_trims_single_stock_crash():
    x = np.full((1, 40), 0.8)
    obs = Obs(x.copy(), np.full((1, 40), EXACT, np.int8), None, [str(i) for i in range(40)])
    base = market.market_observation(obs).x[0, 0]
    x2 = x.copy()
    x2[0, 0] = 10.0
    crash = market.market_observation(Obs(x2, obs.kind, None, obs.tickers)).x[0, 0]
    assert np.isclose(base, crash) and np.isclose(base, np.log(0.8))


def test_tail_forecast_matches_direct_computation():
    rng = np.random.default_rng(4)
    obs, _ = simulate_halfnormal(mdl.V1_S, mdl.V1_A, [0.2, 0.2, 0.1], T=50, N=3, rng=rng)
    m = hmm.HMM(hmm.HalfNormalEmission(mdl.V1_S.copy(), np.array([0.2, 0.2, 0.1]), "state", True), mdl.V1_A)
    fr = m.filter(obs)
    q = mdl.tail_forecast(m, fr, 0.2)
    t, i = 30, 1
    direct = sum(fr.pred[t, i, k] * 0.8 * 2 * sst.norm.sf(3 / mdl.V1_S[k]) for k in range(2)) \
        + fr.pred[t, i, 2] * 0.9 * 2 * sst.norm.sf(3 / mdl.V1_S[2])
    assert np.isclose(q[t, i], direct)


def test_newey_west_zero_lag_is_iid_se_and_ols_matches_lstsq():
    rng = np.random.default_rng(5)
    x = rng.normal(size=500)
    assert np.isclose(stats.newey_west_se_mean(x, 0), x.std(ddof=0) / np.sqrt(500))
    X = np.column_stack([np.ones(500), rng.normal(size=500)])
    y = X @ np.array([0.3, -1.2]) + rng.normal(size=500)
    res = stats.ols_hac(y, X, lags=5)
    np.testing.assert_allclose(res["beta"], np.linalg.lstsq(X, y, rcond=None)[0])


def test_logistic_calibration_recovers_identity():
    rng = np.random.default_rng(6)
    q = rng.beta(0.5, 10, 200000)
    y = (rng.random(200000) < q).astype(float)
    c = stats.logistic_calibration(q, y)
    assert abs(c["slope"] - 1) < 4 * c["se_slope"] and abs(c["intercept"]) < 4 * c["se_intercept"]


def test_stationary_bootstrap_indices_cover_range():
    idx = stats.stationary_bootstrap_indices(1000, 20, np.random.default_rng(7))
    assert idx.min() >= 0 and idx.max() < 1000 and len(idx) == 1000
    runs = np.mean(np.diff(idx) == 1)
    assert 0.9 < runs < 0.99        # mean block length ~20 -> ~95% continuations


def test_starting_points_have_vector_parameters_for_every_candidate():
    rng = np.random.default_rng(8)
    obs, _ = simulate_halfnormal(mdl.V1_S, mdl.V1_A, [0.2, 0.2, 0.1], T=40, N=3, rng=rng)
    for spec in mdl.CANDIDATES:
        for m in mdl.starting_points(spec, obs, n_random=2):
            e = m.emission
            assert e.s.shape == (spec.K,) and m.A.shape == (spec.K, spec.K)
            assert (e.pi is None) == (spec.zero_mode == "none")
            if e.pi is not None:
                assert e.pi.shape == (spec.K,)
            m.emission.log_emissions(obs)   # must not raise
