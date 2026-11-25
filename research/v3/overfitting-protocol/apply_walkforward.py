"""Part (b): SPA, pairwise S0/S1 and MCS on the committed dev walk-forward log scores.

Tests T1-T4 exactly as declared in candidates.md. Reads only
outputs/dev/walkforward_daily.npz (read-only). Output: apply_walkforward.json
"""
import json
import sys

import numpy as np

sys.path.insert(0, "/tmp/claude-0/v3/overfitting-protocol")
import spa  # noqa: E402

NPZ = "/home/user/fear-signal/outputs/dev/walkforward_daily.npz"
OUT = "/tmp/claude-0/v3/overfitting-protocol/apply_walkforward.json"
NAMES = ["S0_v1_published", "S1_v1_refit", "S2_shared_K3", "S3_state_K3", "S4_state_K2", "S5_state_K4"]
NPAR = {"S0_v1_published": 11, "S1_v1_refit": 11, "S2_shared_K3": 12, "S3_state_K3": 14,
        "S4_state_K2": 7, "S5_state_K4": 23}


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def main():
    z = np.load(NPZ, allow_pickle=False)
    sc = z["scored"]
    nobs = z["nobs"][sc]
    blk = z["block"][sc]
    scale = nobs.mean()
    ls = {k: z[f"logscore__{k}"][sc] / scale for k in NAMES}   # per-obs-scaled daily log score
    n = int(sc.sum())
    out = {"n_days": n, "mean_obs_per_day": float(scale), "n_stock_days": float(nobs.sum())}

    # ---------------- T1: does any of S2-S5 beat S1?
    fam = ["S2_shared_K3", "S3_state_K3", "S4_state_K2", "S5_state_K4"]
    d = np.column_stack([ls[k] - ls["S1_v1_refit"] for k in fam])
    prim = spa.spa(d, block=10, B=10_000, variance="sb", seed=0, names=fam)
    out["T1_primary"] = prim
    sens = {}
    for blkl in (5, 10, 20, 40):
        for var in ("sb", "nw", "boot"):
            r = spa.spa(d, block=blkl, B=10_000, variance=var, nw_lags=10 if var == "nw" else None,
                        seed=0)
            sens[f"block{blkl}:{var}"] = {k: r[k] for k in ("T", "p_c", "p_l", "p_u")} | {
                "t": r["t"], "se": r["se"]}
    out["T1_sensitivity"] = sens

    # ---------------- T2: S0 vs S1, both directions, pooled and per fold
    t2 = {}
    d01 = ls["S0_v1_published"] - ls["S1_v1_refit"]   # positive = S0 better
    for lab, dd in (("S0_beats_S1", d01), ("S1_beats_S0", -d01)):
        r = spa.spa(dd, block=10, B=10_000, variance="sb", seed=0)
        t2[lab] = {k: r[k] for k in ("T", "p_c", "p_l", "p_u", "dbar", "se", "t")}
        r = spa.spa(dd, block=10, B=10_000, variance="nw", nw_lags=10, seed=0)
        t2[lab + ":nw10"] = {k: r[k] for k in ("T", "p_c", "dbar", "se", "t")}
    folds = {}
    for b, lab in enumerate(["2024H2", "2025H1", "2025H2"]):
        m = blk == b
        dd = d01[m]
        se = np.sqrt(spa.lrv(dd, "nw", nw_lags=10)[0] / m.sum())
        # per-fold mean in nats/obs on that fold's own obs count (as RESULTS.md quotes)
        per_obs = float((z[f"logscore__S0_v1_published"][sc][m] - z[f"logscore__S1_v1_refit"][sc][m]).sum()
                        / nobs[m].sum())
        folds[lab] = {"days": int(m.sum()), "S0_minus_S1_per_obs": per_obs,
                      "mean_scaled": float(dd.mean()), "nw_se_scaled": float(se),
                      "t": float(dd.mean() / se)}
    t2["per_fold"] = folds
    out["T2_S0_vs_S1"] = t2

    # ---------------- T3: model confidence set on S1-S5 (S0 excluded: not OOS)
    oos = ["S1_v1_refit"] + fam
    losses = np.column_stack([-ls[k] for k in oos])
    m90 = spa.mcs(losses, alpha=0.10, block=10, B=10_000, seed=0, names=oos)
    m90["most_parsimonious_in_set"] = min(m90["included"], key=lambda k: (NPAR[k], -ls[k].mean()))
    out["T3_mcs90"] = m90
    out["T3_mcs90_block20"] = spa.mcs(losses, alpha=0.10, block=20, B=10_000, seed=0, names=oos)
    # also with S0 included, for reference only
    losses0 = np.column_stack([-ls[k] for k in ["S0_v1_published"] + oos])
    out["T3_mcs90_with_S0_reference_only"] = spa.mcs(losses0, alpha=0.10, block=10, B=10_000, seed=0,
                                                    names=["S0_v1_published"] + oos)

    # ---------------- T4: minimum detectable effect at the empirical covariance
    # Synthetic samples: stationary bootstrap (block 10) of the observed matrix d,
    # demeaned, plus a mean vector mu. One candidate (each in turn) is shifted to
    # +delta nats/obs; the others keep their observed (negative) means ("realistic")
    # or are set to 0 ("least favourable"). Inner SPA_c: block 10, B = 499.
    rng = np.random.default_rng(7)
    dc = d - d.mean(axis=0)
    obs_mu = d.mean(axis=0)
    R, Bi = 600, 499
    t4 = {}
    deltas = [0.0005, 0.001, 0.0015, 0.002, 0.003]
    for world in ("realistic", "least_favourable"):
        for j, cand in enumerate(fam):
            if cand == "S4_state_K2":
                continue
            res = {}
            for delta in deltas:
                mu = (obs_mu.copy() if world == "realistic" else np.zeros(len(fam)))
                mu[j] = delta
                rej05 = rej10 = adopt_1se = adopt_gate10 = 0
                for _ in range(R):
                    idx = spa.sb_indices(n, 10.0, 1, rng)[0]
                    ds = dc[idx] + mu
                    boot = spa.sb_counts(n, 10.0, Bi, rng) @ ds / n
                    dbar = ds.mean(axis=0)
                    se = np.sqrt(spa.lrv(ds, "sb", 10.0) / n)
                    r = spa._spa_from_boot(dbar, se, boot, n)
                    se_nw = np.sqrt(spa.lrv(ds, "nw", nw_lags=10) / n)
                    one_se = np.any(dbar >= se_nw)
                    rej05 += r["p_c"] < 0.05
                    rej10 += r["p_c"] < 0.10
                    adopt_1se += one_se
                    adopt_gate10 += one_se and r["p_c"] < 0.10
                res[str(delta)] = {"spa_c_rej05": rej05 / R, "spa_c_rej10": rej10 / R,
                                   "adopt_1se_only": adopt_1se / R, "adopt_1se_and_spa10": adopt_gate10 / R}
            t4[f"{world}:{cand}"] = res
            print(world, cand, {k: (v["spa_c_rej05"], v["spa_c_rej10"]) for k, v in res.items()}, flush=True)
    out["T4_mde"] = t4
    with open(OUT, "w") as fh:
        json.dump(clean(out), fh, indent=1)


if __name__ == "__main__":
    main()
