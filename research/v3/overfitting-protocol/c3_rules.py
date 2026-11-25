"""C3: operating characteristics of the adoption rules at the empirical dev covariance.

Synthetic dev samples = stationary bootstrap (block 10) of the observed, demeaned
S2..S5-minus-S1 differentials, plus a chosen mean vector. Rules (adopt a change):
  R1  1-SE only:            some candidate has dbar_k >= NW(10) s.e._k
  R2  1-SE and SPA_c < 0.10
  R3  1-SE and SPA_c < 0.05
Worlds under H0 (nothing truly better than S1):
  LF       all four means 0 (least favourable)
  mixed    S2, S3, S5 at 0; S4 at its observed mean (-3.3 s.e.)
  observed all at their observed (negative) means
Plus power at delta = +0.002 nats/obs on S2 (others 0), for reference.
R = 2000, inner B = 499. Output: c3_rules.json
"""
import json, sys
import numpy as np
sys.path.insert(0, "/tmp/claude-0/v3/overfitting-protocol")
import spa

z = np.load("/home/user/fear-signal/outputs/dev/walkforward_daily.npz")
sc = z["scored"]; scale = z["nobs"][sc].mean()
fam = ["S2_shared_K3", "S3_state_K3", "S4_state_K2", "S5_state_K4"]
d = np.column_stack([(z[f"logscore__{k}"][sc] - z["logscore__S1_v1_refit"][sc]) / scale for k in fam])
n = d.shape[0]; dc = d - d.mean(0); obs = d.mean(0)
worlds = {"LF": np.zeros(4), "mixed": np.array([0, 0, obs[2], 0]), "observed": obs.copy(),
          "power_S2_+0.002_rest0": np.array([0.002, 0, 0, 0]),
          "power_S2_+0.002_restobs": np.r_[0.002, obs[1:]]}
rng = np.random.default_rng(99)
R, Bi = 2000, 499
out = {}
for w, mu in worlds.items():
    acc = np.zeros(5)
    for _ in range(R):
        ds = dc[spa.sb_indices(n, 10.0, 1, rng)[0]] + mu
        dbar = ds.mean(0)
        boot = spa.sb_counts(n, 10.0, Bi, rng) @ ds / n
        r = spa._spa_from_boot(dbar, np.sqrt(spa.lrv(ds, "sb", 10.0) / n), boot, n)
        one = bool(np.any(dbar >= np.sqrt(spa.lrv(ds, "nw", nw_lags=10) / n)))
        acc += [one, one and r["p_c"] < 0.10, one and r["p_c"] < 0.05, r["p_c"] < 0.10, r["p_c"] < 0.05]
    acc /= R
    out[w] = dict(zip(["R1_1se_only", "R2_1se_spa10", "R3_1se_spa05", "spa_c_rej10", "spa_c_rej05"], acc.tolist()))
    out[w]["mc_se_at_0.10"] = float(np.sqrt(0.09 / R))
    print(w, out[w], flush=True)
json.dump(out, open("/tmp/claude-0/v3/overfitting-protocol/c3_rules.json", "w"), indent=1)
