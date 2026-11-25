"""Size of the per-sample prepivoted SPA_c (spa.spa_calibrated, fixed
studentisation) at the empirical dev covariance, least-favourable and mixed
nulls, plus power at S2 +0.003. R_mc = 300, outer R = 199, inner B = 199.
Output: c3_prepivot_size.json"""
import json, sys
import numpy as np
sys.path.insert(0, "/tmp/claude-0/v3/overfitting-protocol")
import spa

z = np.load("/home/user/fear-signal/outputs/dev/walkforward_daily.npz")
sc = z["scored"]; scale = z["nobs"][sc].mean()
fam = ["S2_shared_K3", "S3_state_K3", "S4_state_K2", "S5_state_K4"]
d = np.column_stack([(z[f"logscore__{k}"][sc] - z["logscore__S1_v1_refit"][sc]) / scale for k in fam])
n = d.shape[0]; dc = d - d.mean(0); obs = d.mean(0)
worlds = {"LF": np.zeros(4), "mixed": np.array([0, 0, obs[2], 0]), "power_S2_+0.003": np.array([0.003, 0, 0, 0])}
if len(sys.argv) > 3:
    worlds = {k: worlds[k] for k in sys.argv[3].split(",")}
RMC = int(sys.argv[1]) if len(sys.argv) > 1 else 300
rng = np.random.default_rng(2468)
out = {}
for w, mu in worlds.items():
    raw, cal, one = [], [], []
    for _ in range(RMC):
        ds = dc[spa.sb_indices(n, 10.0, 1, rng)[0]] + mu
        r = spa.spa_calibrated(ds, block=10.0, B=199, R=199, seed=int(rng.integers(2**31)))
        raw.append(r["p_c_raw"]); cal.append(r["p_c_calibrated"])
        one.append(np.any(ds.mean(0) >= np.sqrt(spa.lrv(ds, "nw", nw_lags=10) / n)))
    raw, cal, one = map(np.asarray, (raw, cal, one))
    out[w] = {"raw_rej05": float((raw < .05).mean()), "cal_rej05": float((cal < .05).mean()),
              "adopt_1se_and_cal05": float(((cal < .05) & one).mean()), "cal_rej10": float((cal < .10).mean())}
    print(w, out[w], flush=True)
json.dump(out, open(sys.argv[2] if len(sys.argv) > 2 else "/tmp/claude-0/v3/overfitting-protocol/c3_prepivot_size.json", "w"), indent=1)
