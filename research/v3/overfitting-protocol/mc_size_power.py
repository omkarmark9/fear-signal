"""C1/C2: Monte Carlo size and power of SPA_c / SPA_l / SPA_u (declared in candidates.md).

DGP: d_{k,t} = mu_k + u_{k,t}; u AR(1), phi = 0.3; innovations multivariate
Student-t(5) (common chi-square mixing, so crash days hit every candidate),
equicorrelation 0.6, unit variance. mu_k is given in standard-error units:
mu_k = delta_k * lr_sd / sqrt(T), lr_sd = 1 / (1 - phi).
T = 380, K in {4, 8, 12}, 1000 replications, B = 499, block 10.
Output: mc_size_power.json
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, "/tmp/claude-0/v3/overfitting-protocol")
import spa  # noqa: E402

T, PHI, RHO, NU, BLOCK, B, R = 380, 0.3, 0.6, 5, 10.0, 499, 1000
CONFIGS = {
    "null_all_zero": lambda K: np.zeros(K),
    "null_one_zero_rest_minus3": lambda K: np.r_[0.0, np.full(K - 1, -3.0)],
    "power_plus2_rest_zero": lambda K: np.r_[2.0, np.zeros(K - 1)],
    "power_plus3_rest_zero": lambda K: np.r_[3.0, np.zeros(K - 1)],
    "power_plus2_rest_minus3": lambda K: np.r_[2.0, np.full(K - 1, -3.0)],
    "power_plus3_rest_minus3": lambda K: np.r_[3.0, np.full(K - 1, -3.0)],
}


def simulate(K, mu_se, rng):
    C = np.full((K, K), RHO) + np.eye(K) * (1 - RHO)
    Lc = np.linalg.cholesky(C)
    burn = 50
    z = rng.standard_normal((T + burn, K)) @ Lc.T
    w = rng.chisquare(NU, size=(T + burn, 1)) / NU
    e = z / np.sqrt(w) * np.sqrt((NU - 2) / NU)
    u = np.empty_like(e)
    u[0] = e[0]
    for t in range(1, T + burn):
        u[t] = PHI * u[t - 1] + e[t]
    u = u[burn:]
    lr_sd = 1.0 / (1.0 - PHI)
    return mu_se * lr_sd / np.sqrt(T) + u


def one_rep(d, rng):
    n = d.shape[0]
    dbar = d.mean(axis=0)
    boot = spa.sb_counts(n, BLOCK, B, rng) @ d / n
    res = {}
    for var in ("sb", "nw", "boot"):
        if var == "boot":
            om2 = n * ((boot - dbar) ** 2).mean(axis=0)
        else:
            om2 = spa.lrv(d, var, BLOCK)
        se = np.sqrt(om2 / n)
        r = spa._spa_from_boot(dbar, se, boot, n)
        for k in ("p_c", "p_l", "p_u"):
            res[f"{var}:{k}"] = r[k]
        if var == "nw":
            res["one_se_any"] = float(np.any(r["t"] >= 1.0))
            res["best_is_first"] = float(np.argmax(r["t"]) == 0)
    return res


def main():
    out = {"settings": dict(T=T, phi=PHI, rho=RHO, nu=NU, block=BLOCK, B=B, R=R)}
    rng = np.random.default_rng(20261124)
    t0 = time.time()
    for K in (4, 8, 12):
        for cname, f in CONFIGS.items():
            mu = f(K)
            acc = {}
            for _ in range(R):
                d = simulate(K, mu, rng)
                for k, v in one_rep(d, rng).items():
                    acc.setdefault(k, []).append(v)
            cell = {}
            for k, v in acc.items():
                v = np.asarray(v)
                if k.startswith(("sb:", "nw:", "boot:")):
                    cell[k] = {"rej05": float(np.mean(v < 0.05)), "rej10": float(np.mean(v < 0.10))}
                else:
                    cell[k] = float(v.mean())
            out[f"K{K}:{cname}"] = cell
            print(f"K={K:2d} {cname:28s} sb:c05={cell['sb:p_c']['rej05']:.3f} "
                  f"sb:l05={cell['sb:p_l']['rej05']:.3f} sb:u05={cell['sb:p_u']['rej05']:.3f} "
                  f"nw:c05={cell['nw:p_c']['rej05']:.3f} boot:c05={cell['boot:p_c']['rej05']:.3f} "
                  f"sb:c10={cell['sb:p_c']['rej10']:.3f} 1se_any={cell['one_se_any']:.3f} "
                  f"[{time.time() - t0:.0f}s]", flush=True)
    with open("/tmp/claude-0/v3/overfitting-protocol/mc_size_power.json", "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
