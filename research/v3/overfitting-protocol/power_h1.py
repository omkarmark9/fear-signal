"""Part (c): sample size for a prospective H1-type test (declared in candidates.md, section D).

Test: Spearman rho of 5-session changes of M and VXN, stationary-bootstrap
percentile 95% CI (block 20), pass if the lower bound > 0.30.
Question: how many sessions n until P(pass) >= 0.80 / 0.90 when true rho ~ 0.55?

1. Analytic: Fisher z with var(atanh r_s) = VIF * 1.06 / (n - 3); the
   variance-inflation factor VIF (overlap of 5-session changes + serial
   dependence + common volatility) is estimated as the ratio of the
   block-bootstrap variance of atanh(r_s) to 1.06/(n-3) on dev OOS and on 2026.
2. Simulation, R replications per n, inner B = 1000 bootstrap resamples:
   A  Gaussian pair, AR(1) daily changes (phi from dev), contemporaneous
      correlation tuned so the population Spearman of 5-session changes = 0.55
   B1 2026 daily (dM, dVXN) pairs, stationary-bootstrap resampled (block 20),
      cumulated, then differenced over 5 sessions (native rho ~ 0.55)
   B2 dev OOS daily pairs, with VXN changes attenuated toward rho ~ 0.55 by
      adding a circularly shifted (independent-in-time) copy of themselves
      [added after declaration as a sensitivity: dev structure at rho 0.55]
   B3 dev OOS daily pairs as they are (rho ~ 0.66, the optimistic case)
   For each DGP the population rho is measured on one long simulated series.
Output: power_h1.json
"""
import json
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats as sst

sys.path.insert(0, "/tmp/claude-0/v3/overfitting-protocol")
sys.path.insert(0, "/home/user/fear-signal/src")
import spa  # noqa: E402
from fearsignal import evaluate, scoring  # noqa: E402

OUT = "/tmp/claude-0/v3/overfitting-protocol/power_h1.json"
NS = [100, 150, 200, 250, 300, 400, 500]
R, BIN, BLOCK_CI, BLOCK_GEN = 300, 1000, 20.0, 20.0
RHO0, RHO_TRUE = 0.30, 0.55


def spearman_rows(a, b):
    ra = sst.rankdata(a, axis=1)
    rb = sst.rankdata(b, axis=1)
    ra -= ra.mean(axis=1, keepdims=True)
    rb -= rb.mean(axis=1, keepdims=True)
    return (ra * rb).sum(axis=1) / np.sqrt((ra ** 2).sum(axis=1) * (rb ** 2).sum(axis=1))


def ci_lower(a5, b5, rng):
    m = len(a5)
    idx = spa.sb_indices(m, BLOCK_CI, BIN, rng)
    boots = spearman_rows(a5[idx], b5[idx])
    return float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975)), boots


def d5(x):
    return x[5:] - x[:-5]


# ------------------------------------------------------------------- sources
def load_sources():
    # 2026: frozen-run M and VXN, daily changes on the full session calendar
    mf = pd.read_csv("/home/user/fear-signal/outputs/holdout/market_fear.csv", parse_dates=["date"]).set_index("date")
    vx = evaluate.load_vxn("/home/user/fear-signal/data/raw/vxn_full.csv", dev_only=False)
    df = pd.DataFrame({"m": mf["M"], "v": vx.reindex(mf.index)})
    dd = df.diff()
    dd = dd[(dd.index >= "2026-01-02")].dropna()
    src2026 = dd.to_numpy()
    # check: the realised H1 statistic on the same construction
    d5df = df.diff(5)[df.index >= "2026-01-02"].dropna()
    rho2026 = float(sst.spearmanr(d5df["m"], d5df["v"])[0])
    # dev OOS: stitched logit breadth of S1 and VXN, changes within each fold's test block
    z = np.load("/home/user/fear-signal/outputs/dev/walkforward_daily.npz")
    dates = pd.to_datetime(z["dates"])
    lb = scoring.logit(z["breadth__S1_v1_refit"])
    vd = evaluate.load_vxn().reindex(dates).to_numpy()
    blk = z["block"]
    rows = []
    for b in range(3):
        m = blk == b
        s, v = lb[m], vd[m]
        ok = np.isfinite(s) & np.isfinite(v)
        s, v = s[ok], v[ok]
        rows.append(np.column_stack([np.diff(s), np.diff(v)]))
    srcdev = np.vstack(rows)
    d5dev = evaluate.d5_within_blocks(lb, vd, blk)
    rhodev = float(sst.spearmanr(d5dev["s"], d5dev["v"])[0])
    return src2026, rho2026, len(d5df), srcdev, rhodev, len(d5dev), d5df, d5dev


def gen_resample(src, n, rng):
    idx = spa.sb_indices(len(src), BLOCK_GEN, 1, rng)[0]
    reps = int(np.ceil((n + 1) / len(src)))
    if reps > 1:
        idx = np.concatenate([spa.sb_indices(len(src), BLOCK_GEN, 1, rng)[0] for _ in range(reps)])
    x = np.cumsum(src[idx[: n + 1]], axis=0)
    return x[:, 0], x[:, 1]


def make_attenuated(src, c):
    shift = len(src) // 2
    v2 = np.roll(src[:, 1], shift)
    return np.column_stack([src[:, 0], src[:, 1] + c * v2])


def gen_gauss(phi_a, phi_b, r, n, rng):
    burn = 50
    e = rng.multivariate_normal([0, 0], [[1, r], [r, 1]], size=n + 1 + burn)
    u = np.empty_like(e)
    u[0] = e[0]
    for t in range(1, len(e)):
        u[t, 0] = phi_a * u[t - 1, 0] + e[t, 0]
        u[t, 1] = phi_b * u[t - 1, 1] + e[t, 1]
    x = np.cumsum(u[burn:], axis=0)
    return x[:, 0], x[:, 1]


def pop_rho(gen, rng, n=60_000):
    a, b = gen(n, rng)
    return float(sst.spearmanr(d5(a), d5(b))[0])


def main():
    t0 = time.time()
    rng = np.random.default_rng(55)
    src26, rho26, n26, srcdev, rhodev, ndev, d5_26, d5_dev = load_sources()
    out = {"realised": {"rho_2026": rho26, "n_changes_2026": n26, "rho_dev_oos": rhodev,
                        "n_changes_dev": ndev}}
    print("realised", out["realised"], flush=True)

    # ---------------- analytic Fisher-z with VIF from the data
    vif = {}
    for lab, a5, b5 in (("2026", d5_26["m"].to_numpy(), d5_26["v"].to_numpy()),
                        ("dev_oos", d5_dev["s"].to_numpy(), d5_dev["v"].to_numpy())):
        m = len(a5)
        idx = spa.sb_indices(m, BLOCK_CI, 4000, rng)
        bz = np.arctanh(spearman_rows(a5[idx], b5[idx]))
        vif[lab] = float(bz.var() / (1.06 / (m - 3)))
    out["vif"] = vif
    zcrit = {"0.80": sst.norm.ppf(0.975) + sst.norm.ppf(0.80), "0.90": sst.norm.ppf(0.975) + sst.norm.ppf(0.90)}
    gap = np.arctanh(RHO_TRUE) - np.arctanh(RHO0)
    analytic = {}
    for lab, v in vif.items():
        for pw, zc in zcrit.items():
            analytic[f"{lab}:power{pw}"] = float(np.ceil(3 + 1.06 * v * (zc / gap) ** 2) + 5)  # +5 sessions per change
    for pw, zc in zcrit.items():
        analytic[f"iid_no_overlap:power{pw}"] = float(np.ceil(3 + 1.06 * (zc / gap) ** 2) + 5)
    out["analytic_sessions"] = analytic
    print("vif", vif, "analytic", analytic, flush=True)

    # ---------------- DGPs
    phi_a = float(sst.pearsonr(srcdev[1:, 0], srcdev[:-1, 0])[0])
    phi_b = float(sst.pearsonr(srcdev[1:, 1], srcdev[:-1, 1])[0])
    # tune Gaussian r so that the population Spearman of 5-session changes is 0.55
    lo, hi = 0.3, 0.9
    for _ in range(18):
        r = (lo + hi) / 2
        pr = pop_rho(lambda n, g: gen_gauss(phi_a, phi_b, r, n, g), np.random.default_rng(1))
        lo, hi = (r, hi) if pr < RHO_TRUE else (lo, r)
    r_gauss = (lo + hi) / 2
    # tune attenuation c for the dev source
    lo, hi = 0.0, 3.0
    for _ in range(16):
        c = (lo + hi) / 2
        s2 = make_attenuated(srcdev, c)
        pr = pop_rho(lambda n, g: gen_resample(s2, n, g), np.random.default_rng(2))
        lo, hi = (c, hi) if pr > RHO_TRUE else (lo, c)
    c_att = (lo + hi) / 2
    srcatt = make_attenuated(srcdev, c_att)
    gens = {
        "A_gauss_ar1": lambda n, g: gen_gauss(phi_a, phi_b, r_gauss, n, g),
        "B1_resample_2026": lambda n, g: gen_resample(src26, n, g),
        "B2_dev_attenuated": lambda n, g: gen_resample(srcatt, n, g),
        "B3_dev_raw_optimistic": lambda n, g: gen_resample(srcdev, n, g),
    }
    out["dgp_params"] = {"phi_dM": phi_a, "phi_dVXN": phi_b, "r_gauss": r_gauss, "c_attenuation": c_att}
    out["pop_rho"] = {k: pop_rho(f, np.random.default_rng(3)) for k, f in gens.items()}
    print("dgp", out["dgp_params"], out["pop_rho"], f"[{time.time()-t0:.0f}s]", flush=True)

    res = {}
    for name, gen in gens.items():
        prho = out["pop_rho"][name]
        res[name] = {}
        for n in NS:
            lows, cover, est = [], 0, []
            for _ in range(R):
                a, b = gen(n, rng)
                a5, b5 = d5(a), d5(b)
                est.append(sst.spearmanr(a5, b5)[0])
                lo_, hi_, _ = ci_lower(a5, b5, rng)
                lows.append(lo_)
                cover += (lo_ <= prho <= hi_)
            lows = np.asarray(lows)
            res[name][str(n)] = {"power_lo_gt_0.30": float(np.mean(lows > RHO0)),
                                 "mc_se": float(np.sqrt(np.mean(lows > RHO0) * (1 - np.mean(lows > RHO0)) / R)),
                                 "coverage95": cover / R, "mean_rho_hat": float(np.mean(est)),
                                 "sd_rho_hat": float(np.std(est))}
            print(name, n, res[name][str(n)], f"[{time.time()-t0:.0f}s]", flush=True)
    out["simulation"] = res
    out["settings"] = dict(R=R, B_inner=BIN, block_ci=BLOCK_CI, block_gen=BLOCK_GEN, ns=NS)
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
