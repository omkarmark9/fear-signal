import numpy as np

from fearsignal import bootstrap, hmm, model as mdl
from fearsignal.simulate import simulate_halfnormal

PI = np.array([0.2, 0.2, 0.05])


def test_block_resample_keeps_cross_section_synchronised():
    rng = np.random.default_rng(0)
    obs, _ = simulate_halfnormal(mdl.V1_S, mdl.V1_A, PI, T=300, N=7, rng=rng)
    bo = bootstrap.block_resample(obs, np.random.default_rng(1), block=50)
    assert bo.x.shape == (50, 7 * 6)
    # each block is 7 consecutive columns copied from one window of real dates
    blk = bo.x[:, :7]
    hits = [s for s in range(251) if np.array_equal(obs.x[s:s + 50], blk, equal_nan=True)]
    assert hits


def test_bootstrap_intervals_cover_truth():
    rng = np.random.default_rng(2)
    obs, _ = simulate_halfnormal(mdl.V1_S, mdl.V1_A, PI, T=600, N=60, rng=rng)
    start = hmm.HMM(hmm.HalfNormalEmission(mdl.V1_S.copy(), PI.copy(), "state", True), mdl.V1_A.copy())
    point = hmm.em(start, obs, max_iter=300, tol=1e-9)
    reps = bootstrap.replicate_fits(point, obs, n_boot=20, seed=3, max_iter=150)
    tab = bootstrap.parameter_table(point, reps, level=0.98)
    lo, hi = np.array(tab["s"]["lo"]), np.array(tab["s"]["hi"])
    assert np.all(lo <= mdl.V1_S * 1.03) and np.all(hi >= mdl.V1_S * 0.97)
    assert np.all(hi > lo)
