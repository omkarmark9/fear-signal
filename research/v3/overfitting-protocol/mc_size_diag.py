"""Diagnose the least-favourable-null oversize seen in mc_size_power (K=8, T=380)."""
import sys, json
import numpy as np
sys.path.insert(0, "/tmp/claude-0/v3/overfitting-protocol")
import spa

T, R, B = 380, 1000, 499

def sim(K, phi, rho, nu, rng):
    C = np.full((K, K), rho) + np.eye(K) * (1 - rho)
    z = rng.standard_normal((T + 50, K)) @ np.linalg.cholesky(C).T
    if nu:
        z = z / np.sqrt(rng.chisquare(nu, size=(T + 50, 1)) / nu) * np.sqrt((nu - 2) / nu)
    u = np.empty_like(z); u[0] = z[0]
    for t in range(1, T + 50):
        u[t] = phi * u[t - 1] + z[t]
    return u[50:]

def size(K, phi, rho, nu, block, seed=1):
    rng = np.random.default_rng(seed)
    rej = {"sb": 0, "true_se": 0, "u_nw_T": 0}
    for _ in range(R):
        d = sim(K, phi, rho, nu, rng)
        n = T
        dbar = d.mean(0)
        boot = spa.sb_counts(n, block, B, rng) @ d / n
        se = np.sqrt(spa.lrv(d, "sb", block) / n)
        rej["sb"] += spa._spa_from_boot(dbar, se, boot, n)["p_c"] < 0.05
        # oracle: true long-run s.e. in numerator studentisation, bootstrap for the max
        lr = 1 / (1 - phi) ** 2 * (1.0)
        se0 = np.full(K, np.sqrt(lr / n))
        rej["true_se"] += spa._spa_from_boot(dbar, se0, boot, n)["p_c"] < 0.05
    return {k: v / R for k, v in rej.items()}

out = {}
for lab, args in {"iid_normal_block1": (8, 0.0, 0.6, 0, 1.0),
                  "iid_normal_block10": (8, 0.0, 0.6, 0, 10.0),
                  "ar03_normal_block10": (8, 0.3, 0.6, 0, 10.0),
                  "ar03_t5_block10": (8, 0.3, 0.6, 5, 10.0),
                  "ar03_t5_block20": (8, 0.3, 0.6, 5, 20.0),
                  "ar03_t5_block5": (8, 0.3, 0.6, 5, 5.0),
                  "iid_t5_block1": (8, 0.0, 0.6, 5, 1.0),
                  "ar03_t5_block10_rho0": (8, 0.3, 0.0, 5, 10.0),
                  "ar03_t5_block10_rho09": (8, 0.3, 0.9, 5, 10.0)}.items():
    out[lab] = size(*args)
    print(lab, out[lab], flush=True)
json.dump(out, open("/tmp/claude-0/v3/overfitting-protocol/mc_size_diag.json", "w"), indent=1)
