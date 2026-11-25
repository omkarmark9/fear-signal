"""Does re-studentising each bootstrap replicate (bootstrap-t) remove the
least-favourable-null oversize found in mc_size_diag? K = 8, T = 380, R = 500."""
import sys, json
import numpy as np
sys.path.insert(0, "/tmp/claude-0/v3/overfitting-protocol")
import spa

T, R, B = 380, 500, 499

def sim(K, phi, rho, nu, rng, mu=None):
    C = np.full((K, K), rho) + np.eye(K) * (1 - rho)
    z = rng.standard_normal((T + 50, K)) @ np.linalg.cholesky(C).T
    if nu:
        z = z / np.sqrt(rng.chisquare(nu, size=(T + 50, 1)) / nu) * np.sqrt((nu - 2) / nu)
    u = np.empty_like(z); u[0] = z[0]
    for t in range(1, T + 50):
        u[t] = phi * u[t - 1] + z[t]
    u = u[50:]
    if mu is not None:
        u = u + mu / (1 - phi) / np.sqrt(T)
    return u

def lrv_sb_batch(x, block):
    """SB-kernel long-run variance of each column of each replicate; x (B, n, K)."""
    Bn, n, K = x.shape
    u = x - x.mean(axis=1, keepdims=True)
    q = 1.0 / block
    i = np.arange(1, n)
    kap = (1 - i / n) * (1 - q) ** i + (i / n) * (1 - q) ** (n - i)
    m = 1 << int(np.ceil(np.log2(2 * n)))
    f = np.fft.rfft(u, n=m, axis=1)
    ac = np.fft.irfft(f * np.conj(f), n=m, axis=1)[:, :n] / n
    return ac[:, 0] + 2 * np.einsum("i,bik->bk", kap, ac[:, 1:n])

def rep(d, block, rng):
    n, K = d.shape
    dbar = d.mean(0)
    idx = spa.sb_indices(n, block, B, rng)
    ds = d[idx]                                   # (B, n, K)
    boot = ds.mean(axis=1)
    se = np.sqrt(spa.lrv(d, "sb", block) / n)
    p_std = spa._spa_from_boot(dbar, se, boot, n)["p_c"]
    se_star = np.sqrt(np.maximum(lrv_sb_batch(ds, block), 1e-300) / n)
    t = dbar / se
    Tobs = max(0.0, t.max())
    thr = -se * np.sqrt(2 * np.log(np.log(n)))
    centre = np.where(dbar >= thr, 0.0, dbar)
    Tb = np.maximum(((boot - dbar + centre) / se_star).max(axis=1), 0.0)
    p_bt = float(np.mean(Tb >= Tobs - 1e-12))
    return p_std, p_bt

out = {}
cases = {"iid_normal_b10": (8, 0.0, 0.6, 0, 10.0, None),
         "ar03_normal_b10": (8, 0.3, 0.6, 0, 10.0, None),
         "ar03_t5_b10": (8, 0.3, 0.6, 5, 10.0, None),
         "ar03_t5_b10_rho0": (8, 0.3, 0.0, 5, 10.0, None),
         "power_ar03_t5_b10_plus3": (8, 0.3, 0.6, 5, 10.0, np.r_[3.0, np.zeros(7)]),
         "power_ar03_t5_b10_plus3_rest_m3": (8, 0.3, 0.6, 5, 10.0, np.r_[3.0, np.full(7, -3.0)])}
for lab, (K, phi, rho, nu, block, mu) in cases.items():
    rng = np.random.default_rng(5)
    ps = np.array([rep(sim(K, phi, rho, nu, rng, mu), block, rng) for _ in range(R)])
    out[lab] = {"standard_rej05": float((ps[:, 0] < 0.05).mean()), "boot_t_rej05": float((ps[:, 1] < 0.05).mean()),
                "standard_rej10": float((ps[:, 0] < 0.10).mean()), "boot_t_rej10": float((ps[:, 1] < 0.10).mean())}
    print(lab, out[lab], flush=True)
json.dump(out, open("/tmp/claude-0/v3/overfitting-protocol/mc_bootstrap_t.json", "w"), indent=1)
