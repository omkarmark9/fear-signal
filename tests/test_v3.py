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
