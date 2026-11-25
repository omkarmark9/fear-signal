"""v3 additions: linear market scale, recursive HAR + Clark-West, prospective lock."""

import numpy as np
import pandas as pd
import pytest

from fearsignal import config, hmm, model as mdl, scoring, stats, validate
from fearsignal.simulate import simulate_halfnormal


def _signal(m_linear=None):
    rng = np.random.default_rng(0)
    obs, _ = simulate_halfnormal(mdl.V1_S, mdl.V1_A, None, T=300, N=30, rng=rng)
    m = hmm.HMM(hmm.HalfNormalEmission(mdl.V1_S.copy(), None, "none", False), mdl.V1_A.copy())
    base = mdl.compute_signal(m, obs, None, None, 0.2)
    m_map = scoring.QuantileMap.fit(base.breadth_L[np.isfinite(base.breadth_L)], dz=0.02)
    return mdl.compute_signal(m, obs, None, m_map, 0.2, m_linear), m_map


def test_linear_market_scale_is_affine_in_logit_breadth_and_keeps_percentile_score():
    v2, m_map = _signal()
    v3, _ = _signal((-0.8, 0.85))
    np.testing.assert_allclose(v3.M, (v3.breadth_L + 0.8) / 0.85)
    np.testing.assert_array_equal(v3.M_pct, v2.M)          # v2's headline is v3's M_pct
    np.testing.assert_array_equal(v2.M, v2.M_pct)
    # rank-preserving: same order of days, so the same alert days
    ok = np.isfinite(v3.M)
    assert np.all(np.diff(v3.M_pct[ok][np.argsort(v3.M[ok])]) >= -1e-12)


def test_clark_west_size_under_nested_null():
    rng = np.random.default_rng(1)
    reps, n, rej = 300, 220, 0
    for _ in range(reps):
        x = np.cumsum(rng.normal(size=n + 200)) * 0.05            # persistent irrelevant regressor
        y = rng.normal(size=n + 200)
        f0, f1 = np.full(n + 200, np.nan), np.full(n + 200, np.nan)
        for t in range(200, n + 200):                              # recursive estimation
            f0[t] = y[:t].mean()
            X = np.column_stack([np.ones(t), x[:t]])
            b = np.linalg.lstsq(X, y[:t], rcond=None)[0]
            f1[t] = b[0] + b[1] * x[t]
        rej += stats.clark_west(y[200:], f0[200:], f1[200:], horizon=1)["p_one_sided"] < 0.05
    assert 0.01 < rej / reps < 0.10


def test_clark_west_formula():
    y, a, b = np.array([1.0, 2, 3, 4]), np.array([1.5, 1.5, 2.5, 3.5]), np.array([1.0, 2.5, 3.0, 3.0])
    f = (y - a) ** 2 - ((y - b) ** 2 - (a - b) ** 2)
    assert np.isclose(stats.clark_west(y, a, b, 1)["mean_adj"], f.mean())


def test_har_recursive_forecasts_are_causal():
    rng = np.random.default_rng(2)
    T = 400
    R = rng.normal(0, 0.01, T) * np.exp(np.cumsum(rng.normal(0, 0.05, T)))
    M = rng.normal(size=T)
    sig = type("S", (), {"M": M})()
    mask = np.zeros(T, bool)
    mask[150:300] = True
    a = validate.har_recursive(sig, R, mask)
    R2 = R.copy()
    R2[301:] *= 3.0                                  # change only data after the scored window
    sig2 = type("S", (), {"M": M})()
    b = validate.har_recursive(sig2, R2, mask)
    # forecasts for t <= 295 use targets known by t (rows s <= t-5 need returns <= t);
    # the 5-day targets of the last scored days do reach past 300, so compare those
    # summary statistics only through the forecasts themselves via a narrower mask
    mask_early = np.zeros(T, bool)
    mask_early[150:295] = True
    a2 = validate.har_recursive(sig, R, mask_early)
    b2 = validate.har_recursive(sig2, R2, mask_early)
    assert a2 == b2
    assert a != b


def test_prospective_panel_is_locked_without_committed_v3_spec(tmp_path, monkeypatch):
    from fearsignal import data, frozen
    monkeypatch.setattr(frozen, "FROZEN_V3_PATH", tmp_path / "frozen_v3.json")
    with pytest.raises(config.HoldoutLocked):
        data.load_prospective_panel("0" * 64)


def test_prospective_constants():
    assert config.PROSPECTIVE_START == config.V3_ESTIMATION_END + pd.Timedelta(days=1)
    assert config.PROSPECTIVE_SESSIONS == 400


def test_challenger_comparison_is_zero_for_an_identical_model():
    from fearsignal import features, pipeline
    rng = np.random.default_rng(5)
    obs, _ = simulate_halfnormal(mdl.V1_S, mdl.V1_A, None, T=120, N=12, rng=rng)
    obs.regime = features.shock_regime(obs)
    m = hmm.HMM(hmm.HalfNormalEmission(mdl.V1_S.copy(), None, "none", False), mdl.V1_A.copy())
    spec3 = {"stock_model": pipeline.model_to_dict(m), "dev_zero_rate": 0.2}
    chal = {"model": pipeline.model_to_dict(m)}
    feat = type("F", (), {"obs": obs})()
    r = pipeline.challenger_comparison(spec3, chal, feat, np.ones(120, bool))
    assert r["gain_per_obs"] == 0.0 and r["n_days"] > 0


def test_challenger_with_regime_stack_round_trips_through_frozen_dict():
    from fearsignal import pipeline
    A = np.stack([mdl.V1_A, mdl.V1_A[::-1, ::-1]])
    m = hmm.HMM(hmm.HalfNormalEmission(mdl.V1_S.copy(), None, "none", False), A)
    m2 = pipeline.model_from_dict(pipeline.model_to_dict(m))
    assert m2.A.shape == (2, 3, 3)
    np.testing.assert_allclose(m2.A, A, atol=1e-10)


def test_prospective_refuses_a_hash_that_is_not_preregistered(tmp_path, monkeypatch):
    from fearsignal import data, pipeline
    monkeypatch.setattr(pipeline, "OUT", tmp_path)
    with pytest.raises(data.HoldoutLocked):
        pipeline.prospective("0" * 64)


def test_parameter_table_handles_stacked_transitions():
    from fearsignal import bootstrap
    A = np.stack([mdl.V1_A, mdl.V1_A])
    m = hmm.HMM(hmm.HalfNormalEmission(mdl.V1_S.copy(), None, "none", False), A)
    tab = bootstrap.parameter_table(m, [m, m])
    assert np.array(tab["expected_duration"]["point"]).shape == (2, 3)


def test_tail_calibration_detail_brackets_the_full_slope():
    rng = np.random.default_rng(7)
    obs, _ = simulate_halfnormal(mdl.V1_S, mdl.V1_A, None, T=160, N=40, rng=rng)
    m = hmm.HMM(hmm.HalfNormalEmission(mdl.V1_S.copy(), None, "none", False), mdl.V1_A.copy())
    sig = mdl.compute_signal(m, obs, None, None, 0.0)
    feat = type("F", (), {"obs": obs})()
    mask = np.ones(160, bool)
    det = validate.tail_calibration_detail(sig, feat, mask)
    full = validate.tail_forecast(sig, feat, mask, 0.03)["calibration"]["slope"]
    assert det["loo_slope_min"] <= full <= det["loo_slope_max"]
    assert det["n_days"] > 100


def test_prospective_refuses_when_the_preregistered_challenger_is_missing(tmp_path, monkeypatch):
    from fearsignal import config as cfg, data, pipeline
    monkeypatch.setattr(pipeline, "OUT", tmp_path)
    monkeypatch.setattr(pipeline, "CHALLENGER_F5_PATH", tmp_path / "missing.json")
    with pytest.raises(data.HoldoutLocked):
        pipeline.prospective(cfg.PREREGISTERED_SHA256["frozen_v3.json"])


def test_forecast_ranges_and_flags():
    from fearsignal import forecast
    idx = pd.bdate_range("2024-01-01", periods=300)
    rng = np.random.default_rng(3)
    counts = pd.DataFrame({"pred": rng.uniform(2, 12, 300), "real": rng.poisson(5, 300)}, index=idx)
    c = forecast.count_forecast(8.0, counts)
    sim = counts[(counts.pred >= 6) & (counts.pred <= 10)]
    assert c["similar_days"] == len(sim)
    assert c["realized_on_similar_days"]["q50"] == float(np.quantile(sim.real, 0.5))
    mp = pd.Series(np.sin(np.arange(300) / 9) * 2 + 1, index=idx)
    m = forecast.market_forecast(2.0, mp)
    assert abs(sum(m["next_band_probabilities"].values()) - 1) < 1e-9
    # scoring: a realized count inside / outside the 90% range
    fc = {"target_session": "2024-06-03", "count_ge_3sigma": {"model_mean": 8.0, "realized_on_similar_days":
          {"q05": 0.0, "q95": 10.0}}, "market": {"next_m_pct": {"q05": 1.0, "q95": 3.0}},
          "stocks_q_next": {"A": 0.5, "B": 0.1}}
    st = pd.DataFrame({"date": pd.Timestamp("2024-06-03"), "ticker": ["A", "B"], "x": [3.5, 0.2]})
    mk = pd.DataFrame({"M_pct": [2.0]}, index=[pd.Timestamp("2024-06-03")])
    r = forecast.score(fc, mk, st)
    assert r["realized_count_ge_3sigma"] == 1 and not r["flagged"] and r["stocks_hit"] == ["A"]
    mk2 = pd.DataFrame({"M_pct": [3.5]}, index=[pd.Timestamp("2024-06-03")])
    assert forecast.score(fc, mk2, st)["flags"]["market_outside_90"]
