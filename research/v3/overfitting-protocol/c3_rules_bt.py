"""C3 repeated with the bootstrap-t (replicate-studentised) SPA_c, same worlds and
seed scheme as c3_rules.py, so the two gates can be compared on the empirical dev
covariance. R = 2000, inner B = 499. Output: c3_rules_bt.json"""
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
          "power_S2_+0.002_restobs": np.r_[0.002, obs[1:]],
          "power_S2_+0.003_rest0": np.array([0.003, 0, 0, 0]),
          "power_S3_+0.003_rest0": np.array([0, 0.003, 0, 0]),
          "power_S5_+0.003_rest0": np.array([0, 0, 0, 0.003])}
R = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
rng = np.random.default_rng(99)
out = {}
for w, mu in worlds.items():
    acc = np.zeros(5)
    for i in range(R):
        ds = dc[spa.sb_indices(n, 10.0, 1, rng)[0]] + mu
        dbar = ds.mean(0)
        r = spa.spa(ds, block=10.0, B=499, variance="sb", studentize="replicate",
                    seed=int(rng.integers(2**31)), chunk=500)
        one = bool(np.any(dbar >= np.sqrt(spa.lrv(ds, "nw", nw_lags=10) / n)))
        acc += [one, one and r["p_c"] < 0.10, one and r["p_c"] < 0.05, r["p_c"] < 0.10, r["p_c"] < 0.05]
    acc /= R
    out[w] = dict(zip(["R1_1se_only", "R2_1se_spaBT10", "R3_1se_spaBT05", "spaBT_c_rej10", "spaBT_c_rej05"],
                      acc.tolist()))
    out[w]["R"] = R
    print(w, out[w], flush=True)
json.dump(out, open("/tmp/claude-0/v3/overfitting-protocol/c3_rules_bt.json", "w"), indent=1)
