"""Critic: independent minimal Hansen (2005) SPA implementation, compared with spa.py."""
import sys, time, json
import numpy as np
sys.path.insert(0, "/tmp/claude-0/v3/critic")
import spa
def my_sb(n, b, rng):
    idx = np.empty(n, int); idx[0] = rng.integers(n)
    for t in range(1, n):
        idx[t] = rng.integers(n) if rng.random() < 1 / b else (idx[t-1] + 1) % n
    return idx
def my_spa(d, block, B, rng):
    n, K = d.shape; dbar = d.mean(0)
    # Hansen's omega: use NW-free "boot" variance from the same draws
    bm = np.array([d[my_sb(n, block, rng)].mean(0) for _ in range(B)])
    om = np.sqrt(n * ((bm - dbar) ** 2).mean(0))
    T = max(0, (np.sqrt(n) * dbar / om).max())
    gc = dbar * (np.sqrt(n) * dbar / om >= -np.sqrt(2 * np.log(np.log(n))))
    Z = bm - gc  # Zbar*_k = dbar*_k - g_c(dbar_k)
    Tb = np.maximum((np.sqrt(n) * Z / om).max(1), 0)
    return float(np.mean(Tb >= T))
rng = np.random.default_rng(99)
res = []
for case in range(4):
    mu = [np.array([0.08, 0, -0.05, -0.5]), np.array([0.0, 0.0, 0.0, 0.0]), np.array([0.12, -0.3, -0.3, -0.3]), np.array([0.05, 0.05, -1, 0])][case]
    e = rng.standard_normal((380, 4)); u = np.empty_like(e); u[0] = e[0]
    for t in range(1, 380): u[t] = 0.3 * u[t-1] + e[t]
    d = u + mu
    t0 = time.time(); mine = my_spa(d, 10.0, 4000, np.random.default_rng(case))
    theirs = spa.spa(d, block=10.0, B=20000, variance="boot", seed=case)["p_c"]
    res.append({"case": case, "mine_pc_boot": mine, "theirs_pc_boot": theirs})
    print(res[-1], flush=True)
# timing of one calibrated call on the dev family
z = np.load("/home/user/fear-signal/outputs/dev/walkforward_daily.npz")
sc = z["scored"]; scale = z["nobs"][sc].mean()
fam = ["S2_shared_K3", "S3_state_K3", "S4_state_K2", "S5_state_K4"]
d = np.column_stack([(z[f"logscore__{k}"][sc] - z["logscore__S1_v1_refit"][sc]) / scale for k in fam])
t0 = time.time(); r = spa.spa_calibrated(d - d.mean(0), block=10.0, B=199, R=199, seed=1); print("time one calibrated call", time.time() - t0)
json.dump(res, open("/tmp/claude-0/v3/critic/spa_indep.json", "w"), indent=1)
