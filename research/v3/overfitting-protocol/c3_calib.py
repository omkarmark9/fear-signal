"""C3b: null distribution of the bootstrap-t SPA_c p-value at the empirical dev
covariance, to calibrate the nominal level (Beran-style prepivoting), and power
at the calibrated level. Stores every p-value. R = 1000 per world, inner B = 499.
Same world definitions as c3_rules.py. Output: c3_calib.npz / c3_calib.json"""
import json, sys
import numpy as np
sys.path.insert(0, "/tmp/claude-0/v3/overfitting-protocol")
import spa

z = np.load("/home/user/fear-signal/outputs/dev/walkforward_daily.npz")
sc = z["scored"]; scale = z["nobs"][sc].mean()
fam = ["S2_shared_K3", "S3_state_K3", "S4_state_K2", "S5_state_K4"]
d = np.column_stack([(z[f"logscore__{k}"][sc] - z["logscore__S1_v1_refit"][sc]) / scale for k in fam])
n = d.shape[0]; dc = d - d.mean(0); obs = d.mean(0)
worlds = {"LF": np.zeros(4), "mixed": np.array([0, 0, obs[2], 0]),
          "power_S2_+0.002": np.array([0.002, 0, 0, 0]),
          "power_S2_+0.003": np.array([0.003, 0, 0, 0]),
          "power_S3_+0.003": np.array([0, 0.003, 0, 0]),
          "power_S5_+0.003": np.array([0, 0, 0, 0.003])}
R = 1000
rng = np.random.default_rng(1234)
P = {}
for w, mu in worlds.items():
    pf, pb, one = np.empty(R), np.empty(R), np.empty(R, bool)
    for i in range(R):
        ds = dc[spa.sb_indices(n, 10.0, 1, rng)[0]] + mu
        s = int(rng.integers(2**31))
        pb[i] = spa.spa(ds, block=10.0, B=499, studentize="replicate", seed=s, chunk=500)["p_c"]
        pf[i] = spa.spa(ds, block=10.0, B=499, studentize="fixed", seed=s)["p_c"]
        one[i] = np.any(ds.mean(0) >= np.sqrt(spa.lrv(ds, "nw", nw_lags=10) / n))
    P[w] = (pf, pb, one)
    print(w, "fixed<.05", (pf < .05).mean(), "bt<.05", (pb < .05).mean(), "bt<.025", (pb < .025).mean(),
          "1se", one.mean(), flush=True)
np.savez("/tmp/claude-0/v3/overfitting-protocol/c3_calib.npz",
         **{f"{w}__{k}": v for w, t in P.items() for k, v in zip(("fixed", "bt", "one_se"), t)})
out = {}
for var, j in (("fixed", 0), ("bt", 1)):
    for tgt in (0.05, 0.10):
        # largest nominal level whose actual false-adoption is <= target in BOTH null worlds
        grid = np.linspace(0.001, tgt, 200)
        ok = [a for a in grid if max(((P[w][j] < a) & P[w][2]).mean() for w in ("LF", "mixed")) <= tgt]
        a_cal = float(max(ok)) if ok else None
        out[f"{var}:target{tgt}"] = {
            "alpha_nominal_calibrated": a_cal,
            "actual_at_nominal": {w: float(((P[w][j] < tgt) & P[w][2]).mean()) for w in worlds},
            "actual_at_calibrated": {w: float(((P[w][j] < a_cal) & P[w][2]).mean()) for w in worlds} if a_cal else None}
out["one_se_only"] = {w: float(P[w][2].mean()) for w in worlds}
out["R"] = R
print(json.dumps(out, indent=1))
json.dump(out, open("/tmp/claude-0/v3/overfitting-protocol/c3_calib.json", "w"), indent=1)
