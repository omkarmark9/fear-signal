"""The HMM engine against brute force, theory, and its own invariants."""

import itertools

import numpy as np
import pytest
from scipy import integrate, optimize
from scipy.stats import norm

from fearsignal import hmm
from fearsignal.features import CENSORED, EXACT, MISSING, ZERO, Obs
from fearsignal.simulate import drop_random, simulate_halfnormal

S_V1 = np.array([0.95, 1.58, 3.82])
A_V1 = np.array([[0.967, 0.011, 0.022], [0.033, 0.916, 0.052], [0.263, 0.319, 0.417]])
A_V1 = A_V1 / A_V1.sum(axis=1, keepdims=True)
PI = np.array([0.25, 0.15, 0.08])


def _model(zero_mode="state", censor=True, s=S_V1, A=A_V1, pi=PI):
    pi_ = None if zero_mode == "none" else (np.full(3, pi.mean()) if zero_mode == "shared" else pi)
    return hmm.HMM(hmm.HalfNormalEmission(np.array(s, float), pi_, zero_mode, censor), A.copy())


def _brute(logE, A, init):
    """Exact filtered/smoothed marginals and likelihood by path enumeration."""
    T, K = logE.shape
    E = np.exp(logE)
    paths = np.array(list(itertools.product(range(K), repeat=T)))
    w = init[paths[:, 0]] * E[0, paths[:, 0]]
    for t in range(1, T):
        w = w * A[paths[:, t - 1], paths[:, t]] * E[t, paths[:, t]]
    lik = w.sum()
    smooth = np.array([[w[paths[:, t] == k].sum() / lik for k in range(K)] for t in range(T)])
    filt = []
    for t in range(T):
        wt = init[paths[:, 0]] * E[0, paths[:, 0]]
        for u in range(1, t + 1):
            wt = wt * A[paths[:, u - 1], paths[:, u]] * E[u, paths[:, u]]
        filt.append([wt[paths[:, t] == k].sum() / wt.sum() for k in range(K)])
    xi = np.zeros((K, K))
    for t in range(T - 1):
        for i in range(K):
            for j in range(K):
                xi[i, j] += w[(paths[:, t] == i) & (paths[:, t + 1] == j)].sum() / lik
    return np.array(filt), smooth, np.log(lik), xi


@pytest.mark.parametrize("zero_mode,censor", [("none", False), ("shared", True), ("state", True)])
def test_filter_smoother_match_brute_force(zero_mode, censor):
    rng = np.random.default_rng(0)
    obs, _ = simulate_halfnormal(S_V1, A_V1, PI, T=6, N=1, rng=rng)
    obs.x[2, 0], obs.kind[2, 0] = 0.0, ZERO
    obs.x[4, 0], obs.kind[4, 0] = 10.0, CENSORED
    obs.x[3, 0], obs.kind[3, 0] = np.nan, MISSING
    m = _model(zero_mode, censor)
    init = np.array([0.5, 0.3, 0.2])
    logE = m.emission.log_emissions(obs)
    start, end = hmm.sequence_bounds(obs.eligible)
    fr, gamma, xi, _ = hmm.smooth(logE, m.A, init, start, end)
    filt, smooth, ll, xi_b = _brute(logE[:, 0], m.A, init)
    np.testing.assert_allclose(fr.alpha[:, 0], filt, atol=1e-12)
    np.testing.assert_allclose(gamma[:, 0], smooth, atol=1e-12)
    np.testing.assert_allclose(fr.loglik, ll, rtol=1e-12)
    np.testing.assert_allclose(xi, xi_b, atol=1e-12)


@pytest.mark.parametrize("zero_mode", ["shared", "state"])
def test_emission_is_a_proper_probability_law(zero_mode):
    """P(ZERO) + integral of the EXACT density over (eps, c) + P(CENSORED) = 1."""
    em = _model(zero_mode).emission
    for k in range(3):
        s, pi = em.s[k], em.pi[k]
        p_zero = pi + (1 - pi) * (2 * norm.cdf(em.eps / s) - 1)
        p_mid = (1 - pi) * integrate.quad(lambda x: 2 / s * norm.pdf(x / s), em.eps, em.clip)[0]
        p_cens = (1 - pi) * 2 * norm.sf(em.clip / s)
        assert abs(p_zero + p_mid + p_cens - 1) < 1e-10


def test_shared_atom_carries_no_evidence():
    """A state-independent no-dip probability cannot move the filter (v1 rule)."""
    m = _model("shared")
    obs = Obs(np.zeros((1, 1)), np.full((1, 1), ZERO, np.int8), None, ["a"])
    le = m.emission.log_emissions(obs)[0, 0]
    # differences come only from the half-normal mass below eps,
    # (1-pi) * eps * sqrt(2/pi) / (s * pi), i.e. a few parts per thousand ...
    assert np.ptp(le) < 5e-3
    # ... and vanish as eps -> 0
    m.emission.eps = 1e-9
    assert np.ptp(m.emission.log_emissions(obs)[0, 0]) < 1e-8


@pytest.mark.filterwarnings("ignore:EM stopped")
@pytest.mark.parametrize("zero_mode,censor", [("none", False), ("shared", True), ("state", True)])
def test_em_is_monotone(zero_mode, censor):
    rng = np.random.default_rng(1)
    obs, _ = simulate_halfnormal(S_V1, A_V1, PI, T=300, N=20, rng=rng)
    obs = drop_random(obs, 0.05, rng)
    start = _model(zero_mode, censor, s=[0.7, 1.2, 2.5],
                   A=np.full((3, 3), 1 / 3) * 0.3 + np.eye(3) * 0.7)
    fit = hmm.em(start, obs, max_iter=60, tol=0)
    h = np.array(fit.history)
    assert np.all(np.diff(h) >= -1e-7 * np.abs(h[:-1]))


def test_em_recovers_parameters():
    rng = np.random.default_rng(2)
    obs, _ = simulate_halfnormal(S_V1, A_V1, PI, T=900, N=120, rng=rng)
    start = _model("state", s=[0.8, 1.4, 3.0], pi=np.array([0.2, 0.2, 0.2]),
                   A=np.full((3, 3), 0.05) + np.eye(3) * 0.85)
    fit = hmm.em(start, obs, max_iter=400, tol=1e-10)
    np.testing.assert_allclose(fit.emission.s, S_V1, rtol=0.04)
    np.testing.assert_allclose(fit.emission.pi, PI, atol=0.02)
    np.testing.assert_allclose(np.diag(fit.A), np.diag(A_V1), atol=0.04)


def test_censored_m_step_equals_direct_mle():
    """K=1: the EM fixed point equals the numerically maximised likelihood."""
    rng = np.random.default_rng(3)
    x = np.abs(rng.standard_normal(4000)) * 4.0
    x[rng.random(4000) < 0.2] = 0.0
    kind = np.where(x <= 1e-3, ZERO, np.where(x >= 10, CENSORED, EXACT)).astype(np.int8)
    obs = Obs(np.minimum(x, 10)[:, None], kind[:, None], None, ["a"])
    m = hmm.HMM(hmm.HalfNormalEmission(np.array([1.0]), np.array([0.5]), "state", True),
                np.ones((1, 1)))
    fit = hmm.em(m, obs, max_iter=2000, tol=1e-14)

    def nll(theta):
        e = hmm.HalfNormalEmission(np.array([np.exp(theta[0])]),
                                   np.array([1 / (1 + np.exp(-theta[1]))]), "state", True)
        return -e.log_emissions(obs).sum()

    opt = optimize.minimize(nll, [0.0, 0.0], method="Nelder-Mead",
                            options={"xatol": 1e-10, "fatol": 1e-10, "maxiter": 5000})
    np.testing.assert_allclose(fit.emission.s[0], np.exp(opt.x[0]), rtol=1e-5)
    np.testing.assert_allclose(fit.emission.pi[0], 1 / (1 + np.exp(-opt.x[1])), rtol=1e-5)


def test_filter_is_causal():
    """Changing observations after t leaves every filtered value up to t unchanged."""
    rng = np.random.default_rng(4)
    obs, _ = simulate_halfnormal(S_V1, A_V1, PI, T=200, N=10, rng=rng)
    m = _model("state")
    a = m.filter(obs).alpha
    x2, k2 = obs.x.copy(), obs.kind.copy()
    x2[120:] = np.minimum(x2[120:] * 3 + 1, 10)
    k2[120:] = np.where(x2[120:] >= 10, CENSORED, EXACT)
    b = m.filter(Obs(x2, k2, obs.dates, obs.tickers)).alpha
    np.testing.assert_array_equal(a[:120], b[:120])
    assert not np.allclose(a[120:], b[120:])


def test_missing_day_propagates_chain():
    m = _model("state")
    obs = Obs(np.array([[5.0], [np.nan]]), np.array([[EXACT], [MISSING]], np.int8), None, ["a"])
    fr = m.filter(obs)
    np.testing.assert_allclose(fr.alpha[1, 0], fr.alpha[0, 0] @ m.A, atol=1e-14)


def test_sequences_start_at_first_eligible_day():
    m = _model("state")
    x = np.array([[np.nan], [np.nan], [4.0]])
    k = np.array([[MISSING], [MISSING], [EXACT]], np.int8)
    fr = m.filter(Obs(x, k, None, ["a"]))
    e = np.exp(m.emission.log_emissions(Obs(x, k, None, ["a"]))[2, 0])
    expect = m.stationary * e / (m.stationary * e).sum()
    np.testing.assert_allclose(fr.alpha[2, 0], expect, atol=1e-14)


def test_stationary_distribution():
    p = hmm.stationary(A_V1)
    np.testing.assert_allclose(p @ A_V1, p, atol=1e-14)
    assert abs(p.sum() - 1) < 1e-14


def test_em_reports_convergence():
    rng = np.random.default_rng(9)
    obs, _ = simulate_halfnormal(S_V1, A_V1, PI, T=200, N=10, rng=rng)
    with pytest.warns(RuntimeWarning, match="EM stopped"):
        capped = hmm.em(_model("state"), obs, max_iter=3, tol=0)
    assert not capped.converged
    done = hmm.em(_model("none", False), obs, max_iter=2000, tol=1e-8)
    assert done.converged


def _brute_tv(logE, As, init):
    """Brute force with a time-varying transition: step t-1 -> t uses As[t-1]."""
    T, K = logE.shape
    E = np.exp(logE)
    paths = np.array(list(itertools.product(range(K), repeat=T)))
    w = init[paths[:, 0]] * E[0, paths[:, 0]]
    for t in range(1, T):
        w = w * As[t - 1][paths[:, t - 1], paths[:, t]] * E[t, paths[:, t]]
    lik = w.sum()
    smooth = np.array([[w[paths[:, t] == k].sum() / lik for k in range(K)] for t in range(T)])
    return smooth, np.log(lik)


def test_regime_dependent_transitions_match_brute_force():
    rng = np.random.default_rng(21)
    obs, _ = simulate_halfnormal(S_V1, A_V1, None, T=6, N=1, rng=rng)
    A0 = A_V1
    A1 = np.array([[0.80, 0.15, 0.05], [0.10, 0.85, 0.05], [0.10, 0.50, 0.40]])
    stack = np.stack([A0, A1])
    regime = np.array([[0], [1], [1], [0], [1], [0]], dtype=np.int8)
    m = hmm.HMM(hmm.HalfNormalEmission(S_V1.copy(), None, "none", False), stack)
    logE = m.emission.log_emissions(obs)
    init = np.array([0.5, 0.3, 0.2])
    start, end = hmm.sequence_bounds(obs.eligible)
    fr, gamma, xi, _ = hmm.smooth(logE, stack, init, start, end, regime)
    smooth, ll = _brute_tv(logE[:, 0], [stack[regime[t, 0]] for t in range(5)], init)
    np.testing.assert_allclose(gamma[:, 0], smooth, atol=1e-12)
    np.testing.assert_allclose(fr.loglik, ll, rtol=1e-12)
    # pair counts split by regime add up to the regime-free total
    assert xi.shape == (2, 3, 3) and np.isclose(xi.sum(), 5.0)
    assert np.isclose(xi[0].sum(), (regime[:5, 0] == 0).sum())


@pytest.mark.filterwarnings("ignore:EM stopped")
def test_em_with_regime_transitions_is_monotone_and_recovers_persistence():
    rng = np.random.default_rng(22)
    T, N = 600, 60
    A0 = np.array([[0.97, 0.02, 0.01], [0.30, 0.65, 0.05], [0.50, 0.30, 0.20]])   # short-lived stress
    A1 = np.array([[0.90, 0.07, 0.03], [0.05, 0.90, 0.05], [0.20, 0.40, 0.40]])   # persistent stress
    reg = (rng.random(T) < 0.2).astype(np.int8)
    S = np.zeros((T, N), int)
    S[0] = rng.choice(3, N, p=hmm.stationary(A0))
    for t in range(1, T):
        P = (A1 if reg[t - 1] else A0)[S[t - 1]]
        S[t] = (rng.random(N)[:, None] > np.cumsum(P, axis=1)).sum(axis=1)
    x = np.abs(rng.standard_normal((T, N))) * S_V1[S]
    kind = np.where(x >= 10, 3, 2).astype(np.int8)
    from fearsignal.features import Obs as O
    obs = O(np.minimum(x, 10), kind, None, [str(i) for i in range(N)], np.repeat(reg[:, None], N, 1))
    start = hmm.HMM(hmm.HalfNormalEmission(np.array([0.8, 1.4, 3.0]), None, "none", True),
                    np.stack([A_V1, A_V1]))
    fit = hmm.em(start, obs, max_iter=300, tol=1e-10)
    h = np.array(fit.history)
    assert np.all(np.diff(h) >= -1e-7 * np.abs(h[:-1]))
    assert fit.A[0][1, 1] < fit.A[1][1, 1] - 0.1          # stress persists less after idiosyncratic days
    np.testing.assert_allclose(fit.emission.s, S_V1, rtol=0.06)
