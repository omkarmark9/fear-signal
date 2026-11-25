"""Size/power check of spa_functional for a family of correlation gains (the
market-layer use case). Daily white-noise innovations, 5-session overlapping
changes (n = 363 changes), benchmark b and K = 3 candidates a_k with
corr(change, v) = 0.62 for all under H0; power: one candidate at 0.70.
Statistic theta_k = Spearman(a_k, v) - Spearman(b, v). Block 20, B = 499, R = 300."""
import json, sys
import numpy as np
from scipy import stats as sst
sys.path.insert(0, "/tmp/claude-0/v3/overfitting-protocol")
import spa

def corr_rows(x, y):
    x = x - x.mean(); y = y - y.mean()
    return (x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum())

def make(rng, n, rhos, common=0.8):
    m = n + 5
    v = rng.standard_normal(m)
    f = rng.standard_normal(m)          # shared component of the market readings
    series = []
    for r in rhos:
        # a = r v + sqrt(1-r^2) (c f + sqrt(1-c^2) e): readings correlated with each other
        e = rng.standard_normal(m)
        series.append(r * v + np.sqrt(1 - r * r) * (common * f + np.sqrt(1 - common ** 2) * e))
    c5 = lambda x: np.convolve(x, np.ones(5), "valid")[1:]
    return c5(v), [c5(s) for s in series]

def run(rhos, R, rng):
    pc = []
    for _ in range(R):
        v, ser = make(rng, 363, rhos)
        rk = [sst.rankdata(s) for s in ser]; rv = sst.rankdata(v)
        def stat(ix):
            # Spearman on the resample approximated by Pearson of the original ranks (fast)
            b = corr_rows(rk[0][ix], rv[ix])
            return np.array([corr_rows(r[ix], rv[ix]) - b for r in rk[1:]])
        pc.append(spa.spa_functional(stat, 363, block=20.0, B=499, seed=int(rng.integers(2**31)))["p_c"])
    pc = np.asarray(pc)
    return {"rej05": float((pc < 0.05).mean()), "rej10": float((pc < 0.10).mean())}

if __name__ == "__main__":
    rng = np.random.default_rng(77)
    out = {"null": run([0.62, 0.62, 0.62, 0.62], 300, rng), "power_one_+0.08": run([0.62, 0.70, 0.62, 0.62], 300, rng)}
    print(out)
    json.dump(out, open("/tmp/claude-0/v3/overfitting-protocol/mc_functional.json", "w"), indent=1)
