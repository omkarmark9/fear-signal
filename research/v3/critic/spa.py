"""Hansen's (2005) test for Superior Predictive Ability, with the stationary
bootstrap, plus two companions used by the v3 protocol:

* ``spa``            Hansen (2005) SPA_c / SPA_l / SPA_u on a T x K matrix of
                     loss differentials (benchmark loss minus candidate loss).
* ``spa_functional`` the same max-t test for K smooth statistics that are not
                     plain means (e.g. a gain in Spearman correlation with VXN),
                     studentised by the bootstrap standard deviation.
* ``mcs``            Hansen, Lunde and Nason (2011) model confidence set with
                     the T_max statistic, sharing the same bootstrap.
* ``spa_calibrated`` / ``spa_functional_calibrated``
                     the same tests with a prepivoted (double-bootstrap) p-value.
                     REQUIRED for this project's data: at the dev covariance the
                     plain SPA_c is 2-2.6x oversized (c3_calib.py).
* ``episode_robustness``  leave-one-episode-out check of a single differential.
* ``pw_block_length``     Politis-White automatic block length.

Conventions (Hansen 2005, section 2):

    d_{k,t} = L_{0,t} - L_{k,t}        positive = candidate k better
    H0:  mu_k = E[d_{k,t}] <= 0 for all k   (no candidate beats the benchmark)
    T    = max(0, max_k sqrt(n) dbar_k / omega_k)

The null distribution is approximated by the stationary bootstrap (Politis and
Romano 1994) with recentred means Zbar*_k = dbar*_k - g(dbar_k):

    SPA_l  g(x) = max(x, 0)                         (lower; liberal bound)
    SPA_c  g(x) = x * 1{ x >= -omega_k sqrt(2 log log n / n) }   (consistent)
    SPA_u  g(x) = x                                 (upper; White's reality check)

so p_l <= p_c <= p_u for the same bootstrap draws. p-values are
P*(T* >= T), which is exactly 1 when T = 0 (every candidate's mean is <= 0).

omega_k^2 estimates the long-run variance of d_k, i.e. Var(sqrt(n) dbar_k):

    'sb'    Hansen's choice: the exact variance of the stationary-bootstrap mean,
            gamma_0 + 2 sum_{i=1}^{n-1} kappa(n,i) gamma_i,
            kappa(n,i) = (1 - i/n)(1-q)^i + (i/n)(1-q)^(n-i), q = 1/block
    'nw'    Newey-West (Bartlett) with ``nw_lags`` lags (default: block)
    'boot'  the empirical variance of sqrt(n) dbar*_k over the bootstrap draws

Everything is vectorised: bootstrap means are computed as (resample counts) @ d,
so B = 10,000 with n = 380 and K = 12 takes well under a second.
"""

from __future__ import annotations

import numpy as np

__all__ = ["sb_indices", "sb_counts", "lrv", "lrv_batch", "spa", "spa_functional", "mcs",
           "bootstrap_means", "spa_calibrated", "spa_functional_calibrated",
           "episode_robustness", "pw_block_length"]


# ----------------------------------------------------------------- bootstrap
def sb_indices(n: int, block: float, size: int, rng) -> np.ndarray:
    """``size`` stationary-bootstrap resamples of range(n), shape (size, n).

    Circular: a block that runs off the end continues at 0. Block lengths are
    geometric with mean ``block`` (block = 1 gives the iid bootstrap)."""
    if block < 1:
        raise ValueError("mean block length must be >= 1")
    p = 1.0 / block
    new = rng.random((size, n)) < p
    new[:, 0] = True
    starts = rng.integers(0, n, size=(size, n))
    t = np.arange(n)
    last = np.maximum.accumulate(np.where(new, t, 0), axis=1)
    base = np.take_along_axis(starts, last, axis=1)
    return (base + (t - last)) % n


def sb_counts(n: int, block: float, size: int, rng) -> np.ndarray:
    """Resample counts, shape (size, n): row b counts how often each t is drawn."""
    idx = sb_indices(n, block, size, rng)
    flat = (np.arange(size)[:, None] * n + idx).ravel()
    return np.bincount(flat, minlength=size * n).reshape(size, n).astype(float)


def bootstrap_means(d: np.ndarray, block: float, B: int, rng, chunk: int = 2000) -> np.ndarray:
    """Stationary-bootstrap sample means of the columns of d, shape (B, K)."""
    d = np.asarray(d, float)
    n = d.shape[0]
    out = np.empty((B, d.shape[1]))
    for s in range(0, B, chunk):
        m = min(chunk, B - s)
        out[s:s + m] = sb_counts(n, block, m, rng) @ d / n
    return out


# ------------------------------------------------------------------ variance
def _autocov(u: np.ndarray, lag: int) -> np.ndarray:
    n = u.shape[0]
    return (u[lag:] * u[:n - lag]).sum(axis=0) / n


def lrv(d: np.ndarray, method: str = "sb", block: float = 10.0, nw_lags: int | None = None) -> np.ndarray:
    """Long-run variance of each column (estimate of Var(sqrt(n) * mean))."""
    d = np.asarray(d, float)
    if d.ndim == 1:
        d = d[:, None]
    n = d.shape[0]
    u = d - d.mean(axis=0)
    v = _autocov(u, 0)
    if method == "sb":
        q = 1.0 / block
        i = np.arange(1, n)
        kap = (1 - i / n) * (1 - q) ** i + (i / n) * (1 - q) ** (n - i)
        # all autocovariances gamma_1..gamma_{n-1} at once (zero-padded FFT)
        m = 1 << int(np.ceil(np.log2(2 * n)))
        f = np.fft.rfft(u, n=m, axis=0)
        acov = np.fft.irfft(f * np.conj(f), n=m, axis=0)[1:n] / n
        v = v + 2 * (kap[:, None] * acov).sum(axis=0)
    elif method == "nw":
        L = int(round(block)) if nw_lags is None else int(nw_lags)
        for lag in range(1, min(L, n - 1) + 1):
            v = v + 2 * (1 - lag / (L + 1)) * _autocov(u, lag)
    else:
        raise ValueError(method)
    return np.maximum(v, 1e-300)


def pw_block_length(x: np.ndarray) -> float:
    """Politis and White (2004) automatic mean block length for the stationary
    bootstrap, with the Patton, Politis and White (2009) correction."""
    u = np.asarray(x, float)
    u = u[np.isfinite(u)] - np.nanmean(u)
    n = len(u)
    kn = max(5, int(np.sqrt(np.log10(n))))
    mmax = int(np.ceil(np.sqrt(n))) + kn
    g0 = u @ u / n
    rho = np.array([1.0] + [(u[l:] @ u[:-l]) / n / g0 for l in range(1, mmax + 1)])
    c = 2 * np.sqrt(np.log10(n) / n)
    mhat = mmax - kn
    for m in range(1, mmax - kn + 1):
        if np.all(np.abs(rho[m:m + kn]) < c):
            mhat = m
            break
    M = min(2 * mhat, mmax)
    lags = np.arange(-M, M + 1)
    g = np.array([(u[abs(l):] @ u[:n - abs(l)]) / n for l in lags])
    a = np.abs(lags / M) if M > 0 else np.zeros_like(lags, float)
    lam = np.where(a <= 0.5, 1.0, np.where(a <= 1.0, 2 * (1 - a), 0.0))
    G = np.sum(lam * np.abs(lags) * g)
    D = 2 * np.sum(lam * g) ** 2
    b = (2 * G ** 2 / D) ** (1 / 3) * n ** (1 / 3) if D > 0 else 1.0
    return float(min(max(b, 1.0), 3 * np.sqrt(n)))


# ----------------------------------------------------------------------- SPA
def _spa_from_boot(dbar: np.ndarray, se: np.ndarray, boot: np.ndarray, n: int) -> dict:
    """Core of the test on the mean scale: se_k = omega_k / sqrt(n)."""
    t = dbar / se
    T = max(0.0, float(t.max()))
    thr = -se * np.sqrt(2 * np.log(np.log(n)))
    # Hansen's Zbar*_k = dbar*_k - g(dbar_k) = (dbar*_k - dbar_k) + centre_k with
    # centre = dbar - g(dbar): 0 for 'u'; min(dbar, 0) for 'l'; and for 'c'
    # dbar if the candidate is "very bad" (t below -sqrt(2 log log n)), else 0.
    centre = {"l": np.minimum(dbar, 0.0),
              "c": np.where(dbar >= thr, 0.0, dbar),
              "u": np.zeros_like(dbar)}
    out = {"T": T, "t": t, "dbar": dbar, "se": se, "threshold_t": float(-np.sqrt(2 * np.log(np.log(n))))}
    dev = boot - dbar  # centred bootstrap means, (B, K)
    for key, mu in centre.items():
        Tb = np.maximum(((dev + mu) / se).max(axis=1), 0.0)
        out[f"p_{key}"] = float(np.mean(Tb >= T - 1e-12))
    out["n_very_bad"] = int(np.sum(dbar < thr))
    return out


def lrv_batch(x: np.ndarray, method: str = "sb", block: float = 10.0,
              nw_lags: int | None = None) -> np.ndarray:
    """``lrv`` for a stack of resampled series x of shape (R, n, K) -> (R, K)."""
    R, n, K = x.shape
    u = x - x.mean(axis=1, keepdims=True)
    if method == "sb":
        q = 1.0 / block
        i = np.arange(1, n)
        w = (1 - i / n) * (1 - q) ** i + (i / n) * (1 - q) ** (n - i)
    elif method == "nw":
        L = int(round(block)) if nw_lags is None else int(nw_lags)
        i = np.arange(1, n)
        w = np.where(i <= L, 1 - i / (L + 1), 0.0)
    else:
        raise ValueError(method)
    m = 1 << int(np.ceil(np.log2(2 * n)))
    f = np.fft.rfft(u, n=m, axis=1)
    ac = np.fft.irfft(f * np.conj(f), n=m, axis=1)[:, :n] / n
    return np.maximum(ac[:, 0] + 2 * np.einsum("i,rik->rk", w, ac[:, 1:n]), 1e-300)


def spa(d: np.ndarray, block: float = 10.0, B: int = 10_000, variance: str = "sb",
        nw_lags: int | None = None, seed: int = 0, names=None,
        studentize: str = "fixed", chunk: int = 250) -> dict:
    """Hansen (2005) SPA test.

    d : (n, K) benchmark-minus-candidate loss differentials (positive = candidate better).
    studentize : 'fixed'     Hansen's original: every bootstrap t* uses the sample omega_k.
                 'replicate' bootstrap-t: each replicate is studentised by the same
                             long-run-variance estimator computed on the resampled
                             series. Removes the oversize caused by noise in omega
                             (parametric MC, mc_bootstrap_t.py) but NOT the oversize
                             caused by a few crash days (c3_calib.py); for this
                             project's data use spa_calibrated. Not for 'boot'.
    Returns T, p_c (consistent), p_l (lower), p_u (upper = White RC), per-candidate
    means, standard errors and t-statistics."""
    d = np.asarray(d, float)
    if d.ndim == 1:
        d = d[:, None]
    if not np.all(np.isfinite(d)):
        raise ValueError("d must be finite; drop unscored days first")
    n, K = d.shape
    rng = np.random.default_rng(seed)
    dbar = d.mean(axis=0)
    if studentize == "fixed":
        boot = bootstrap_means(d, block, B, rng)
        if variance == "boot":
            omega2 = n * ((boot - dbar) ** 2).mean(axis=0)
        else:
            omega2 = lrv(d, variance, block, nw_lags)
        se = np.sqrt(omega2 / n)
        out = _spa_from_boot(dbar, se, boot, n)
    elif studentize == "replicate":
        if variance == "boot":
            raise ValueError("replicate studentisation needs 'sb' or 'nw'")
        se = np.sqrt(lrv(d, variance, block, nw_lags) / n)
        t = dbar / se
        T = max(0.0, float(t.max()))
        thr = -se * np.sqrt(2 * np.log(np.log(n)))
        centre = {"l": np.minimum(dbar, 0.0), "c": np.where(dbar >= thr, 0.0, dbar),
                  "u": np.zeros_like(dbar)}
        Tb = {k: np.empty(B) for k in centre}
        for s in range(0, B, chunk):
            m = min(chunk, B - s)
            ds = d[sb_indices(n, block, m, rng)]               # (m, n, K)
            bm = ds.mean(axis=1)
            se_b = np.sqrt(lrv_batch(ds, variance, block, nw_lags) / n)
            for k, mu in centre.items():
                Tb[k][s:s + m] = np.maximum(((bm - dbar + mu) / se_b).max(axis=1), 0.0)
        out = {"T": T, "t": t, "dbar": dbar, "se": se,
               "threshold_t": float(-np.sqrt(2 * np.log(np.log(n)))),
               "n_very_bad": int(np.sum(dbar < thr))}
        for k in centre:
            out[f"p_{k}"] = float(np.mean(Tb[k] >= T - 1e-12))
    else:
        raise ValueError(studentize)
    out.update({"n": n, "K": K, "block": block, "B": B, "variance": variance,
                "studentize": studentize})
    if names is not None:
        out["names"] = list(names)
    return out


def spa_functional(stat_fn, n: int, block: float = 20.0, B: int = 2000, seed: int = 0,
                   names=None) -> dict:
    """SPA max-t test for K smooth statistics theta_k (positive = candidate better).

    stat_fn(idx) must return a length-K array of theta computed on the rows
    ``idx`` of the (paired) data; stat_fn(np.arange(n)) is the point estimate.
    Studentised by the bootstrap s.d. of theta*_k (the 'boot' variance), which
    is what makes it usable for correlations, where no closed-form long-run
    variance is at hand. Valid asymptotically for smooth functionals of means
    (stationary bootstrap: Politis and Romano 1994)."""
    rng = np.random.default_rng(seed)
    theta = np.asarray(stat_fn(np.arange(n)), float)
    boot = np.empty((B, theta.size))
    for b in range(B):
        boot[b] = stat_fn(sb_indices(n, block, 1, rng)[0])
    se = boot.std(axis=0, ddof=0)
    out = _spa_from_boot(theta, se, boot, n)
    # percentile CIs of each theta for reporting
    out["ci95"] = np.quantile(boot, [0.025, 0.975], axis=0).T
    out.update({"n": n, "K": theta.size, "block": block, "B": B, "variance": "boot"})
    if names is not None:
        out["names"] = list(names)
    return out


# ----------------------------------------------------------------------- MCS
def mcs(losses: np.ndarray, alpha: float = 0.10, block: float = 10.0, B: int = 10_000,
        seed: int = 0, names=None) -> dict:
    """Model confidence set (Hansen, Lunde and Nason 2011), T_max statistic.

    losses : (n, M) loss of each model per period (lower is better).
    Returns the elimination order, MCS p-values, and the set at level 1 - alpha."""
    L = np.asarray(losses, float)
    n, M = L.shape
    names = list(names) if names is not None else [f"m{i}" for i in range(M)]
    rng = np.random.default_rng(seed)
    Lbar = L.mean(axis=0)
    boot = bootstrap_means(L, block, B, rng)           # (B, M)
    alive = list(range(M))
    order, pvals = [], []
    running = 0.0
    while len(alive) > 1:
        a = np.array(alive)
        dbar = Lbar[a] - Lbar[a].mean()                  # d_{i.} : model i vs the set average
        dstar = boot[:, a] - boot[:, a].mean(axis=1, keepdims=True)
        se = np.sqrt(((dstar - dbar) ** 2).mean(axis=0))
        t = dbar / se
        Tmax = t.max()
        Tb = ((dstar - dbar) / se).max(axis=1)
        p = float(np.mean(Tb >= Tmax))
        running = max(running, p)
        worst = int(a[np.argmax(t)])
        order.append(names[worst])
        pvals.append(running)
        alive.remove(worst)
    order.append(names[alive[0]])
    pvals.append(1.0)
    mcs_p = dict(zip(order, pvals))
    included = [m for m in names if mcs_p[m] >= alpha]
    return {"elimination_order": order, "mcs_pvalue": mcs_p, "included": included,
            "alpha": alpha, "block": block, "B": B, "n": n}


# --------------------------------------------- double-bootstrap calibration
def spa_calibrated(d: np.ndarray, block: float = 10.0, B: int = 999, R: int = 999,
                   variance: str = "sb", studentize: str = "fixed", seed: int = 0,
                   names=None) -> dict:
    """SPA_c with a level calibrated in the bootstrap world of the sample
    (prepivoting, Beran 1988; double bootstrap).

    Why: the loss differentials of this project are dominated by a handful of
    crash days (kurtosis 15-80; five days carry 55-77% of the variance), and at
    the dev covariance raw SPA_c rejects a true null 12.9% of the time at
    nominal 5% (c3_calib.py). Two null worlds are built by stationary bootstrap
    of the demeaned d: 'lf' (every mean 0, Hansen's least-favourable
    configuration) and 'c' (the SPA_c null: 'very bad' candidates keep their
    mean). In each the raw p-value is computed R times. The calibrated p-value
    is max over the two worlds of P_world(p_null <= p_obs), i.e. the actual
    rejection rate a test at level p_obs would have there. 'alpha_nominal_05'
    is the largest raw level whose actual rate is <= 0.05 in both worlds.

    Caveat (c3_prepivot_size.py): when the calibration sample itself may lack
    a rare crash day, no resampling scheme can see it; on synthetic samples drawn
    from the dev world the calibrated test still rejected 7-12% at 5%. Treat the
    level as approximate and keep the threshold strict."""
    d = np.asarray(d, float)
    if d.ndim == 1:
        d = d[:, None]
    n, K = d.shape
    rng = np.random.default_rng(seed)
    r0 = spa(d, block=block, B=B, variance=variance, studentize=studentize,
             seed=int(rng.integers(2**31)), names=names)
    dbar = d.mean(axis=0)
    thr = -r0["se"] * np.sqrt(2 * np.log(np.log(n)))
    worlds = {"lf": np.zeros(K), "c": np.where(dbar >= thr, 0.0, dbar)}
    dc = d - dbar
    pn = {w: np.empty(R) for w in worlds}
    for w, mu in worlds.items():
        for i in range(R):
            ds = dc[sb_indices(n, block, 1, rng)[0]] + mu
            pn[w][i] = spa(ds, block=block, B=B, variance=variance, studentize=studentize,
                           seed=int(rng.integers(2**31)))["p_c"]
    r0["p_c_raw"] = r0["p_c"]
    r0["p_c_calibrated_by_world"] = {w: float(np.mean(v <= r0["p_c"] + 1e-12)) for w, v in pn.items()}
    r0["p_c_calibrated"] = max(r0["p_c_calibrated_by_world"].values())
    grid = np.unique(np.concatenate([v for v in pn.values()]))
    ok = [a for a in grid if max(np.mean(v < a) for v in pn.values()) <= 0.05]
    r0["alpha_nominal_05"] = float(max(ok)) if ok else 0.0
    r0["R_outer"] = R
    return r0


def spa_functional_calibrated(stat_fn, n: int, block: float = 20.0, B: int = 499, R: int = 499,
                              seed: int = 0, names=None) -> dict:
    """spa_functional with a prepivoted p-value (double bootstrap on indices).

    Outer resample io of the rows; its estimate theta_o is moved to the SPA_c
    null by theta_o - theta_hat + mu0 (mu0 = theta_hat for 'very bad'
    candidates, else 0); the inner bootstrap resamples io itself
    (io[inner indices]) for the s.d. and the null distribution. The calibrated
    p-value is the share of outer null p-values at or below the observed one.
    Cost R * B evaluations of stat_fn."""
    rng = np.random.default_rng(seed)
    r0 = spa_functional(stat_fn, n, block=block, B=B, seed=int(rng.integers(2**31)), names=names)
    theta = np.asarray(r0["dbar"], float)
    thr0 = -np.asarray(r0["se"]) * np.sqrt(2 * np.log(np.log(n)))
    mu0 = np.where(theta >= thr0, 0.0, theta)
    pn = np.empty(R)
    for i in range(R):
        io = sb_indices(n, block, 1, rng)[0]
        raw_o = np.asarray(stat_fn(io), float)
        th_o = raw_o - theta + mu0
        inner = sb_indices(n, block, B, rng)
        boot = np.array([stat_fn(io[ix]) for ix in inner], float)
        se_o = boot.std(axis=0)
        # inner bootstrap means are centred at the outer *raw* estimate
        pn[i] = _spa_from_boot(th_o, se_o, boot - raw_o + th_o, n)["p_c"]
    r0["p_c_raw"] = r0["p_c"]
    r0["p_c_calibrated"] = float(np.mean(pn <= r0["p_c"] + 1e-12))
    r0["R_outer"] = R
    return r0


def episode_robustness(d: np.ndarray, n_top: int = 5, halfwidth: int = 2,
                       nw_lags: int = 10) -> dict:
    """Leave-one-episode-out check for a single differential series d (positive =
    candidate better). Episodes are +-halfwidth sessions around each of the n_top
    days with the largest |d - mean|. Returns the mean and NW t-statistic with
    each episode removed, and with all of them removed, plus the share of the
    sum of squared deviations carried by the n_top days."""
    d = np.asarray(d, float)
    n = len(d)
    u = d - d.mean()
    top = np.argsort(-np.abs(u))[:n_top]
    res = {"share_ss_top": float((u[top] ** 2).sum() / (u ** 2).sum()), "drop_one": []}
    allmask = np.ones(n, bool)
    for i in top:
        m = np.ones(n, bool)
        m[max(0, i - halfwidth): i + halfwidth + 1] = False
        allmask &= m
        x = d[m]
        se = np.sqrt(lrv(x, "nw", nw_lags=nw_lags)[0] / len(x))
        res["drop_one"].append({"day_index": int(i), "mean": float(x.mean()), "t": float(x.mean() / se)})
    x = d[allmask]
    se = np.sqrt(lrv(x, "nw", nw_lags=nw_lags)[0] / len(x))
    res["drop_all"] = {"mean": float(x.mean()), "t": float(x.mean() / se), "n": int(allmask.sum())}
    res["min_t_drop_one"] = float(min(r["t"] for r in res["drop_one"]))
    return res
