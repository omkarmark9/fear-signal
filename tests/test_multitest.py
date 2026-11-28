"""Multiple-testing suite: known answers and operating characteristics."""

import numpy as np
import pytest

from fearsignal.trading import multitest as mt


def test_stepm_controls_familywise_error_under_the_null():
    rng = np.random.default_rng(0)
    reps, n, K, hits = 120, 400, 25, 0
    for r in range(reps):
        M = rng.standard_t(5, size=(n, K)) * 0.01
        res = mt.stepm(M, alpha=0.10, block=10, B=400, seed=r)
        hits += bool(res["rejected"].any())
    rate = hits / reps
    assert 0.03 < rate < 0.20, rate                 # nominal 0.10; MC s.e. about 0.03


def test_stepm_finds_true_effects_and_steps_down():
    rng = np.random.default_rng(1)
    n, K = 500, 20
    M = rng.normal(0, 0.01, (n, K))
    M[:, :3] += 0.004                                # three strong trials (t about 9)
    res = mt.stepm(M, alpha=0.10, block=10, B=500, seed=2)
    assert res["rejected"][:3].all() and not res["rejected"][3:].any()
    # the step-down critical value can only fall as trials leave the active set
    cs = [s["critical_t"] for s in res["steps"]]
    assert all(a >= b - 1e-12 for a, b in zip(cs, cs[1:]))


def test_pbo_is_about_one_half_for_pure_noise_and_near_zero_with_a_real_edge():
    rng = np.random.default_rng(3)
    noise = rng.normal(0, 0.01, (480, 20))
    p0 = mt.pbo_cscv(noise, S=10)["pbo"]
    assert 0.25 < p0 < 0.75, p0
    good = noise.copy()
    good[:, 0] += 0.003
    p1 = mt.pbo_cscv(good, S=10)
    assert p1["pbo"] < 0.05 and p1["p_selected_oos_sharpe_below_0"] < 0.05


def test_effective_trials_limits():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(300, 1))
    assert mt.effective_trials(np.repeat(x, 10, axis=1) + 1e-9 * rng.normal(size=(300, 10))) == pytest.approx(1.0, abs=1e-3)
    ind = rng.normal(size=(5000, 8))
    assert mt.effective_trials(ind) == pytest.approx(8.0, rel=0.02)


def test_deflated_sharpe_reduces_to_psr_with_one_trial_and_falls_with_more_trials():
    rng = np.random.default_rng(5)
    x = rng.normal(0.001, 0.01, 500)
    srs = rng.normal(0, 0.03, 50)
    d1 = mt.deflated_sharpe(x, srs, N=1)
    assert d1["dsr"] == pytest.approx(d1["psr_vs_0"])
    d50 = mt.deflated_sharpe(x, srs, N=50)
    assert d50["dsr"] < d1["dsr"]
    assert mt.expected_max_sharpe(0.03, 100) > mt.expected_max_sharpe(0.03, 10) > 0
