"""C3c: does a leave-one-episode-out requirement help the adoption rule at the
empirical dev covariance? Same worlds as c3_calib.py (fresh draws), R = 1000,
inner B = 499. Rules (k* = candidate with the largest t among those passing 1-SE):
  R1  1-SE only
  R3  1-SE and SPA_c (fixed studentisation) p < alpha_cal (fixed:target0.05 in c3_calib.json)
  R5  R3 and LOEO: k*'s mean stays > 0 after dropping any one of its 5 dominant episodes
  R6  1-SE and LOEO (no SPA)
Output: c3_loeo.json"""
import json, sys
import numpy as np
sys.path.insert(0, "/tmp/claude-0/v3/overfitting-protocol")
import spa

cal = json.load(open("/tmp/claude-0/v3/overfitting-protocol/c3_calib.json"))
a_cal = cal["fixed:target0.05"]["alpha_nominal_calibrated"]
z = np.load("/home/user/fear-signal/outputs/dev/walkforward_daily.npz")
sc = z["scored"]; scale = z["nobs"][sc].mean()
fam = ["S2_shared_K3", "S3_state_K3", "S4_state_K2", "S5_state_K4"]
d = np.column_stack([(z[f"logscore__{k}"][sc] - z["logscore__S1_v1_refit"][sc]) / scale for k in fam])
n = d.shape[0]; dc = d - d.mean(0); obs = d.mean(0)
worlds = {"LF": np.zeros(4), "mixed": np.array([0, 0, obs[2], 0]),
          "power_S2_+0.002": np.array([0.002, 0, 0, 0]), "power_S2_+0.003": np.array([0.003, 0, 0, 0]),
          "power_S5_+0.003": np.array([0, 0, 0, 0.003])}
R = 1000
rng = np.random.default_rng(4321)
out = {"alpha_cal": a_cal, "R": R}
for w, mu in worlds.items():
    acc = np.zeros(4)
    for _ in range(R):
        ds = dc[spa.sb_indices(n, 10.0, 1, rng)[0]] + mu
        dbar = ds.mean(0)
        se = np.sqrt(spa.lrv(ds, "nw", nw_lags=10) / n)
        t = dbar / se
        one = bool(np.any(t >= 1.0))
        if not one:
            continue
        k = int(np.argmax(np.where(t >= 1.0, t, -np.inf)))
        p = spa.spa(ds, block=10.0, B=499, studentize="fixed", seed=int(rng.integers(2**31)))["p_c"]
        loeo = min(e["mean"] for e in spa.episode_robustness(ds[:, k])["drop_one"]) > 0
        acc += [1, p < a_cal, (p < a_cal) and loeo, loeo]
    out[w] = dict(zip(["R1_1se", "R3_1se_spa_cal", "R5_1se_spa_cal_loeo", "R6_1se_loeo"], (acc / R).tolist()))
    print(w, out[w], flush=True)
json.dump(out, open("/tmp/claude-0/v3/overfitting-protocol/c3_loeo.json", "w"), indent=1)
