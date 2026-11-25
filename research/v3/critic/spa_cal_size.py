"""Critic: FRESH-SEED size check of spa.spa_calibrated (the version the protocol adopts).
Worlds: (1) dev-bootstrap LF, (2) dev-bootstrap mixed (S4 keeps its mean), both as in
c3_prepivot_size.py but a new seed; (3) parametric LF null (t5, AR(0.3), equicorr 0.6, K=4),
a world the calibration was never revised against."""
import sys, json, time
import numpy as np
sys.path.insert(0, "/tmp/claude-0/v3/critic")
import spa
RMC = int(sys.argv[1]) if len(sys.argv) > 1 else 300
z = np.load("/home/user/fear-signal/outputs/dev/walkforward_daily.npz")
sc = z["scored"]; scale = z["nobs"][sc].mean()
fam = ["S2_shared_K3", "S3_state_K3", "S4_state_K2", "S5_state_K4"]
d = np.column_stack([(z[f"logscore__{k}"][sc] - z["logscore__S1_v1_refit"][sc]) / scale for k in fam])
n = d.shape[0]; dc = d - d.mean(0); obs = d.mean(0)
def param(rng, K=4, T=380, phi=0.3, rho=0.6, nu=5):
    C = np.full((K, K), rho) + np.eye(K) * (1 - rho)
    zz = rng.standard_normal((T + 50, K)) @ np.linalg.cholesky(C).T
    zz = zz / np.sqrt(rng.chisquare(nu, size=(T + 50, 1)) / nu) * np.sqrt((nu - 2) / nu)
    u = np.empty_like(zz); u[0] = zz[0]
    for t in range(1, T + 50): u[t] = phi * u[t-1] + zz[t]
    return u[50:]
worlds = {"devLF": lambda r: dc[spa.sb_indices(n, 10.0, 1, r)[0]],
          "devMixed": lambda r: dc[spa.sb_indices(n, 10.0, 1, r)[0]] + np.array([0, 0, obs[2], 0]),
          "paramLF": lambda r: param(r)}
rng = np.random.default_rng(31337)
out = {}
t0 = time.time()
for w, gen in worlds.items():
    raw, cal = [], []
    for i in range(RMC):
        ds = gen(rng)
        r = spa.spa_calibrated(ds, block=10.0, B=199, R=199, seed=int(rng.integers(2**31)))
        raw.append(r["p_c_raw"]); cal.append(r["p_c_calibrated"])
    raw, cal = np.array(raw), np.array(cal)
    k = int((cal < .05).sum())
    out[w] = {"R": RMC, "raw_rej05": float((raw < .05).mean()), "cal_rej05": float((cal < .05).mean()),
              "cal_rej10": float((cal < .10).mean()), "cal_rej05_mcse": float(np.sqrt(k / RMC * (1 - k / RMC) / RMC))}
    print(w, out[w], f"{time.time()-t0:.0f}s", flush=True)
    json.dump(out, open("/tmp/claude-0/v3/critic/spa_cal_size.json", "w"), indent=1)
