"""Critic: P1 family gate for the market layer with the AS-SHIPPED v2 M (fold-map M) as the
incumbent and {BL (ship-the-selected-object), C1..C4} as the family (K=5)."""
import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, "/tmp/claude-0/v3/critic"); sys.path.insert(1, "/tmp/claude-0/v3/market-aggregation")
import spa
from ma_common import load_dev, aggregates, standardise
from fearsignal import evaluate
d, dates, vxn, folds = load_dev(); blk = d["block"]; T = len(dates)
CANDS = {"C1_B_lin": "B", "C2_sev": "sev", "C3_Zbar_fixed": "Zbar", "C4_Lbar": "Lbar"}
ser = {"inc_M_foldmap": d["M"].copy(), "BL": d["BL"].copy()}
for nm in CANDS: ser[nm] = np.full(T, np.nan)
for b, ff in enumerate(folds):
    ag = aggregates(ff); test = blk == b
    for nm, key in CANDS.items():
        z, mu, sd = standardise(ag[key], ff["train"]); ser[nm][test] = z[test]
d5 = {k: evaluate.d5_within_blocks(s, vxn, blk) for k, s in ser.items()}
idx = d5["BL"].index
for k in d5: assert d5[k].index.equals(idx)
V = d5["BL"]["v"].to_numpy(); names = ["BL"] + list(CANDS)
S = np.column_stack([d5[k]["s"].to_numpy() for k in ["inc_M_foldmap"] + names])
n = len(V)
def rk(a, axis=0): return np.argsort(np.argsort(a, axis=axis), axis=axis).astype(float)
def stat_fn(ix):
    v = rk(V[ix]); s = rk(S[ix], 0)
    v -= v.mean(); s -= s.mean(0)
    r = (s * v[:, None]).sum(0) / np.sqrt((s ** 2).sum(0) * (v ** 2).sum())
    return r[1:] - r[0]
pt = stat_fn(np.arange(n))
g = spa.spa_functional_calibrated(stat_fn, n, block=20.0, B=299, R=299, seed=5, names=names)
out = {"n": n, "names": names, "gain_vs_shipped_M": pt.tolist(), "t": np.asarray(g["t"]).tolist(),
       "p_c_raw": g["p_c_raw"], "p_c_calibrated": g["p_c_calibrated"], "ci95": np.asarray(g["ci95"]).tolist()}
# K=1 version: BL only
g1 = spa.spa_functional_calibrated(lambda ix: stat_fn(ix)[:1], n, block=20.0, B=299, R=299, seed=6)
out["BL_only"] = {"p_c_raw": g1["p_c_raw"], "p_c_calibrated": g1["p_c_calibrated"], "t": float(np.asarray(g1["t"])[0])}
json.dump(out, open("/tmp/claude-0/v3/critic/mkt_gate.json", "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
