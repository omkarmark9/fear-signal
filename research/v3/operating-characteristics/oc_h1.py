"""Operating characteristics of the pre-registered H1' rule (v3 prospective read).

Rule: Spearman rho of 5-session changes of M and VXN over N sessions (N changes,
built as in fearsignal.validate._d5: diff(5) on the full calendar, so N sessions
give N changes). Percentile stationary-bootstrap 95% CI, mean block 20, as in
fearsignal.stats.corr_ci (circular SB, scipy average-rank Spearman, np.quantile
linear), inner B = 2000 = corr_ci's default n_boot. PASS iff lower bound > 0.30.

DGPs reuse /tmp/claude-0/v3/overfitting-protocol/power_h1.py:
  A   Gaussian pair, AR(1) daily changes with dev phi's (power_h1 dgp_params);
      contemporaneous innovation corr r solved ANALYTICALLY so the population
      Spearman of 5-session changes equals the target (bivariate normal:
      rho_s = 6/pi asin(rho_p/2)). Generated with lfilter (same process).
  B1  2026 daily (dM, dVXN) pairs (power_h1.load_sources), stationary-bootstrap
      resampled (block 20), cumulated, differenced over 5 sessions. Target rho
      below the native 0.527 reached with power_h1.make_attenuated (add c x a
      circularly shifted copy of dVXN); above it by adding k x scaled dM to dVXN.
  B2  dev OOS daily pairs (power_h1 source), resampled the same way, attenuated
      to the target with power_h1.make_attenuated (as power_h1's B2).
  Population rho of the resampling DGPs: brentq on c (or k) with common random
  numbers (4 paths x 1e6 sessions), then re-measured on 4 fresh 1e6 paths
  (mean, s.e. reported).
  Resampling uses one stationary-bootstrap path of exactly the needed length
  (sb_path below) instead of power_h1's concatenation of len(src)-long
  resamples; same DGP, no forced block break at len(src).

Outputs: oc_h1_params.json, oc_h1_cells/<cell>.npz, oc_h1.json, oc_h1.log
Usage: python oc_h1.py [--quick]
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402
from scipy import optimize, signal  # noqa: E402
from scipy import stats as sst  # noqa: E402

sys.path.insert(0, "/tmp/claude-0/v3/overfitting-protocol")
sys.path.insert(0, "/home/user/fear-signal/src")
import power_h1 as P  # noqa: E402
import spa  # noqa: E402

OUT = "/tmp/claude-0/v3/oc"
CELLDIR = f"{OUT}/oc_h1_cells"
BLOCK_CI, BLOCK_GEN, B_IN, RHO0 = 20.0, 20.0, 2000, 0.30
QLEV = np.array([0.005, 0.01, 0.0125, 0.02, 0.025, 0.05, 0.95, 0.975, 0.99])
I_LO, I_HI = 4, 7  # positions of 0.025 and 0.975 in QLEV
WORKERS, CHUNK = 2, 100
POP_LEN, POP_TUNE_PATHS, POP_CHECK_PATHS = 1_000_000, 4, 4
QUICK = "--quick" in sys.argv

# cells: (dgp, target rho, N, R). Boundary cells first.
CELLS = []
for _d in ("A", "B1", "B2"):
    CELLS.append((_d, 0.30, 400, 4000))
for _d in ("A", "B1", "B2"):
    CELLS.append((_d, 0.30, 300, 2000))
for _r, _R in ((0.55, 1500), (0.45, 1000), (0.60, 1000)):
    for _d in ("A", "B1", "B2"):
        CELLS.append((_d, _r, 400, _R))
if QUICK:
    CELLS = [(d, r, n, 40) for d, r, n, _ in CELLS[:3]]
    POP_LEN = 200_000

SRC = {}      # (dgp, target) -> transformed daily-change source (resampling DGPs)
GAUSS = {}    # target -> (phi_a, phi_b, r)


# ------------------------------------------------------------------ generators
def sb_path(m, length, block, rng):
    """One circular stationary-bootstrap index path of `length` into range(m)."""
    new = rng.random(length) < 1.0 / block
    new[0] = True
    starts = rng.integers(0, m, length)
    t = np.arange(length)
    last = np.maximum.accumulate(np.where(new, t, 0))
    return (starts[last] + (t - last)) % m


def five_changes(daily):
    x = np.vstack([np.zeros((1, 2)), np.cumsum(daily, axis=0)])
    return x[5:] - x[:-5]


def gen_resample(src, nchg, rng):
    return five_changes(src[sb_path(len(src), nchg + 4, BLOCK_GEN, rng)])


def gen_gauss(phi_a, phi_b, r, nchg, rng, burn=50):
    e = rng.standard_normal((nchg + 4 + burn, 2))
    e[:, 1] = r * e[:, 0] + np.sqrt(1 - r * r) * e[:, 1]
    u = np.column_stack([signal.lfilter([1.0], [1.0, -phi_a], e[:, 0]),
                         signal.lfilter([1.0], [1.0, -phi_b], e[:, 1])])
    return five_changes(u[burn:])


def make_reinforced(src, k):
    """Raise the M-VXN dependence: dVXN + k * dM scaled to dVXN's s.d."""
    s = src[:, 1].std() / src[:, 0].std()
    return np.column_stack([src[:, 0], src[:, 1] + k * s * src[:, 0]])


# ------------------------------------------------------------- populations
def gauss_spearman5(phi_a, phi_b, r):
    i = np.arange(5)
    K = i[:, None] - i[None, :]
    va = (phi_a ** np.abs(K)).sum() / (1 - phi_a ** 2)
    vb = (phi_b ** np.abs(K)).sum() / (1 - phi_b ** 2)
    c = np.where(K >= 0, phi_a ** np.abs(K), phi_b ** np.abs(K)).sum() / (1 - phi_a * phi_b)
    rp = r * c / np.sqrt(va * vb)
    return 6 / np.pi * np.arcsin(rp / 2), rp


def pop_rho_paths(src, paths):
    vals = []
    for idx in paths:
        d = five_changes(src[idx])
        vals.append(sst.spearmanr(d[:, 0], d[:, 1])[0])
    vals = np.asarray(vals)
    return float(vals.mean()), float(vals.std(ddof=1) / np.sqrt(len(vals)))


def tune_resample(src, target, rng):
    paths = [sb_path(len(src), POP_LEN, BLOCK_GEN, rng) for _ in range(POP_TUNE_PATHS)]
    native = pop_rho_paths(src, paths)[0]
    if target <= native:
        fam, f = "attenuate", (lambda c: P.make_attenuated(src, c))
    else:
        fam, f = "reinforce", (lambda k: make_reinforced(src, k))
    g = lambda th: pop_rho_paths(f(th), paths)[0] - target  # noqa: E731
    hi = 0.5
    while g(hi) * g(0.0) > 0:
        hi *= 2
        if hi > 64:
            raise RuntimeError("cannot bracket")
    th = optimize.brentq(g, 0.0, hi, xtol=1e-5) if abs(g(0.0)) > 1e-12 else 0.0
    s2 = f(th)
    check = [sb_path(len(src), POP_LEN, BLOCK_GEN, rng) for _ in range(POP_CHECK_PATHS)]
    m, se = pop_rho_paths(s2, check)
    return s2, {"family": fam, "theta": float(th), "native_rho": native,
                "pop_rho_check": m, "pop_rho_check_se": se}


# ------------------------------------------------------------------ worker
def run_chunk(args):
    key, dgp, target, nchg, reps, seed = args
    rng = np.random.default_rng(seed)
    est = np.empty(reps)
    q = np.empty((reps, len(QLEV)))
    bsd = np.empty(reps)
    bzsd = np.empty(reps)
    for j in range(reps):
        if dgp == "A":
            d = gen_gauss(*GAUSS[target], nchg, rng)
        else:
            d = gen_resample(SRC[(dgp, target)], nchg, rng)
        a5, b5 = d[:, 0], d[:, 1]
        est[j] = P.spearman_rows(a5[None, :], b5[None, :])[0]
        idx = spa.sb_indices(nchg, BLOCK_CI, B_IN, rng)
        boots = P.spearman_rows(a5[idx], b5[idx])
        q[j] = np.quantile(boots, QLEV)
        bsd[j] = boots.std(ddof=1)
        bzsd[j] = np.arctanh(np.clip(boots, -0.999999, 0.999999)).std(ddof=1)
    return key, est, q, bsd, bzsd


def pse(p, n):
    return float(np.sqrt(max(p * (1 - p), 0.0) / n))


def summarise(cell, pop, est, q, bsd, bzsd):
    dgp, target, nchg, R = cell
    lo, hi = q[:, I_LO], q[:, I_HI]
    p_pass = float(np.mean(lo > RHO0))
    above = float(np.mean(lo > pop))
    below = float(np.mean(hi < pop))
    cov = 1 - above - below
    s = {"dgp": dgp, "target_rho": target, "N": nchg, "R": int(len(est)), "pop_rho": pop,
         "P_pass": p_pass, "P_pass_mcse": pse(p_pass, len(est)),
         "coverage95": cov, "coverage95_mcse": pse(cov, len(est)),
         "P_lo_above_true": above, "P_lo_above_true_mcse": pse(above, len(est)),
         "P_hi_below_true": below, "P_hi_below_true_mcse": pse(below, len(est)),
         "mean_rho_hat": float(est.mean()), "sd_rho_hat": float(est.std(ddof=1)),
         "mean_boot_sd": float(bsd.mean()), "boot_sd_ratio": float(bsd.mean() / est.std(ddof=1)),
         "sd_atanh_rho_hat": float(np.arctanh(est).std(ddof=1)), "mean_boot_sd_atanh": float(bzsd.mean()),
         "mean_lower_bound": float(lo.mean())}
    # pass rate of "q_alpha > 0.30" for other lower percentile levels (calibration)
    s["P_q_alpha_gt_0.30"] = {f"{a:g}": float(np.mean(q[:, i] > RHO0)) for i, a in enumerate(QLEV) if a < 0.5}
    # Fisher-z normal interval with the bootstrap s.e. of atanh(r), as an alternative
    zlo = np.tanh(np.arctanh(est) - 1.959964 * bzsd)
    s["P_pass_fisherz_bootse"] = float(np.mean(zlo > RHO0))
    return s


def main():
    t0 = time.time()
    os.makedirs(CELLDIR, exist_ok=True)
    log = open(f"{OUT}/oc_h1{'_quick' if QUICK else ''}.log", "a")

    def say(*a):
        msg = " ".join(str(x) for x in a) + f"  [{time.time() - t0:.0f}s]"
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    src26, rho26, n26, srcdev, rhodev, ndev, _, _ = P.load_sources()
    phi_a = float(sst.pearsonr(srcdev[1:, 0], srcdev[:-1, 0])[0])
    phi_b = float(sst.pearsonr(srcdev[1:, 1], srcdev[:-1, 1])[0])
    say("sources", src26.shape, rho26, srcdev.shape, rhodev, "phi", phi_a, phi_b)

    params = {"phi_dM": phi_a, "phi_dVXN": phi_b, "A": {}, "B1": {}, "B2": {}}
    targets = sorted({c[1] for c in CELLS})
    prng = np.random.default_rng(20261125)
    for tgt in targets:
        r = optimize.brentq(lambda rr: gauss_spearman5(phi_a, phi_b, rr)[0] - tgt, 0.0, 0.999, xtol=1e-12)
        GAUSS[tgt] = (phi_a, phi_b, r)
        x = gen_gauss(phi_a, phi_b, r, POP_LEN, prng)
        params["A"][str(tgt)] = {"r_innov": r, "pearson5": gauss_spearman5(phi_a, phi_b, r)[1],
                                 "pop_rho_analytic": tgt,
                                 "pop_rho_sim_check": float(sst.spearmanr(x[:, 0], x[:, 1])[0])}
        for dgp, src in (("B1", src26), ("B2", srcdev)):
            s2, info = tune_resample(src, tgt, prng)
            SRC[(dgp, tgt)] = s2
            params[dgp][str(tgt)] = info
        say("tuned", tgt, params["A"][str(tgt)], params["B1"][str(tgt)], params["B2"][str(tgt)])
    with open(f"{OUT}/oc_h1_params{'_quick' if QUICK else ''}.json", "w") as fh:
        json.dump(params, fh, indent=1)

    def pop_of(dgp, tgt):
        return tgt if dgp == "A" else params[dgp][str(tgt)]["pop_rho_check"]

    ss = np.random.SeedSequence(5512026)
    cell_seeds = ss.spawn(len(CELLS))
    tasks = []
    for ci, cell in enumerate(CELLS):
        dgp, tgt, nchg, R = cell
        nch = int(np.ceil(R / CHUNK))
        seeds = cell_seeds[ci].spawn(nch)
        for k in range(nch):
            reps = min(CHUNK, R - k * CHUNK)
            tasks.append((ci, dgp, tgt, nchg, reps, seeds[k]))
    acc = {ci: [] for ci in range(len(CELLS))}
    left = {ci: sum(1 for t in tasks if t[0] == ci) for ci in range(len(CELLS))}
    results = []
    jpath = f"{OUT}/oc_h1{'_quick' if QUICK else ''}.json"
    with Pool(WORKERS) as pool:
        for key, est, q, bsd, bzsd in pool.imap(run_chunk, tasks):
            acc[key].append((est, q, bsd, bzsd))
            left[key] -= 1
            if left[key] == 0:
                cell = CELLS[key]
                est = np.concatenate([a[0] for a in acc[key]])
                q = np.vstack([a[1] for a in acc[key]])
                bsd = np.concatenate([a[2] for a in acc[key]])
                bzsd = np.concatenate([a[3] for a in acc[key]])
                acc[key] = None
                name = f"{cell[0]}_rho{cell[1]:.2f}_N{cell[2]}"
                np.savez_compressed(f"{CELLDIR}/{name}{'_quick' if QUICK else ''}.npz",
                                    est=est, q=q, bsd=bsd, bzsd=bzsd, qlev=QLEV)
                s = summarise(cell, pop_of(cell[0], cell[1]), est, q, bsd, bzsd)
                results.append(s)
                say("cell", name, json.dumps(s))
                with open(jpath, "w") as fh:
                    json.dump({"settings": {"B_inner": B_IN, "block_ci": BLOCK_CI, "block_gen": BLOCK_GEN,
                                            "rho0": RHO0, "qlev": QLEV.tolist(), "workers": WORKERS,
                                            "pop_len": POP_LEN, "cells": CELLS},
                               "params": params, "cells": results}, fh, indent=1)
    say("done")


if __name__ == "__main__":
    main()
