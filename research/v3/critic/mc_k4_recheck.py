"""Critic: re-run the parametric K=4 least-favourable-null size cell of mc_size_power.py with the
CURRENT spa.py (it was modified after mc_size_power.json was written), fresh seed."""
import sys, json
import numpy as np
sys.path.insert(0, "/tmp/claude-0/v3/critic"); sys.path.insert(1, "/tmp/claude-0/v3/overfitting-protocol")
import spa
from mc_size_power import simulate, T, BLOCK
rng = np.random.default_rng(424242); R, B = 1000, 499
rej = {"fixed_sb": 0, "one_se_any": 0}
for _ in range(R):
    d = simulate(4, np.zeros(4), rng)
    r = spa.spa(d, block=BLOCK, B=B, variance="sb", seed=int(rng.integers(2**31)))
    rej["fixed_sb"] += r["p_c"] < 0.05
    rej["one_se_any"] += bool(np.any(d.mean(0) >= np.sqrt(spa.lrv(d, "nw", nw_lags=10) / T)))
out = {k: v / R for k, v in rej.items()}; out["mc_se"] = float(np.sqrt(0.07 * 0.93 / R))
json.dump(out, open("/tmp/claude-0/v3/critic/mc_k4_recheck.json", "w")); print(out)
