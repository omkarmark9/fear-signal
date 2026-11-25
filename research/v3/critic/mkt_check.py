"""Critic: independent re-computation of the market-aggregation dev OOS chain
(BL vs fold-map M vs legacy) from COMMITTED walkforward_daily.npz + shared cache maps."""
import pickle, json, sys
import numpy as np, pandas as pd
from scipy import stats as sst
from fearsignal import evaluate, scoring, stats
z = np.load("/home/user/fear-signal/outputs/dev/walkforward_daily.npz")
print([k for k in z.files if "S1" in k or k in ("dates","block","scored")][:30])
dates = pd.to_datetime(z["dates"]); blk = z["block"]
B = z["breadth__S1_v1_refit"]; BL = scoring.logit(B); Mleg = z["mlegacy__S1_v1_refit"]
cache = pickle.load(open("/tmp/claude-0/v3/shared/fold_models_S1.pkl", "rb"))
Mf = np.full(len(B), np.nan)
for f, fd in enumerate(cache["folds"]):
    m = blk == f
    Mf[m] = fd["m_map"](BL[m])
vxn = evaluate.load_vxn().reindex(dates).to_numpy()
ser = {"BL": BL, "M_foldmap": Mf, "Mleg": Mleg, "B": B}
d5 = {k: evaluate.d5_within_blocks(s, vxn, blk) for k, s in ser.items()}
idx = d5["BL"].index
for k in d5: assert d5[k].index.equals(idx), k
V = d5["BL"]["v"].to_numpy(); S = {k: d5[k]["s"].to_numpy() for k in d5}
bk = blk[idx]
out = {"n": len(idx)}
out["rho"] = {k: float(sst.spearmanr(S[k], V)[0]) for k in S}
out["per_fold"] = {k: [float(sst.spearmanr(S[k][bk == b], V[bk == b])[0]) for b in range(3)] for k in S}
rng = np.random.default_rng(7)
n = len(V); R = 2000
def rk(x): return sst.rankdata(x)
bs = {p: [] for p in ("Mf-BL", "BL-Mleg", "Mf-Mleg", "B-BL")}
bsfold = {b: [] for b in range(3)}
for r in range(R):
    ix = stats.stationary_bootstrap_indices(n, 20.0, rng)
    v = V[ix]
    c = {k: sst.spearmanr(S[k][ix], v)[0] for k in S}
    bs["Mf-BL"].append(c["M_foldmap"] - c["BL"]); bs["BL-Mleg"].append(c["BL"] - c["Mleg"])
    bs["Mf-Mleg"].append(c["M_foldmap"] - c["Mleg"]); bs["B-BL"].append(c["B"] - c["BL"])
pt = {"Mf-BL": out["rho"]["M_foldmap"] - out["rho"]["BL"], "BL-Mleg": out["rho"]["BL"] - out["rho"]["Mleg"],
      "Mf-Mleg": out["rho"]["M_foldmap"] - out["rho"]["Mleg"], "B-BL": out["rho"]["B"] - out["rho"]["BL"]}
out["pairs"] = {k: {"diff": pt[k], "se": float(np.std(v, ddof=1)), "z": pt[k] / float(np.std(v, ddof=1)),
                    "ci": np.quantile(v, [0.025, 0.975]).tolist()} for k, v in bs.items()}
# per-fold paired s.e. for Mf-BL (bootstrap within fold)
pf = {}
for b in range(3):
    m = bk == b; Sm = {k: S[k][m] for k in S}; Vm = V[m]; nb = m.sum()
    vals = []
    for r in range(1000):
        ix = stats.stationary_bootstrap_indices(nb, 20.0, rng)
        vals.append(sst.spearmanr(Sm["M_foldmap"][ix], Vm[ix])[0] - sst.spearmanr(Sm["BL"][ix], Vm[ix])[0])
    d = sst.spearmanr(Sm["M_foldmap"], Vm)[0] - sst.spearmanr(Sm["BL"], Vm)[0]
    pf[f"fold{b}"] = {"n": int(nb), "diff": float(d), "se": float(np.std(vals, ddof=1)), "z": float(d / np.std(vals, ddof=1))}
out["per_fold_Mf_minus_BL"] = pf
# leave-one-month-out on dev for Mf-BL
mon = dates[idx].to_period("M")
lomo = {}
for mth in mon.unique():
    k = mon != mth
    lomo[str(mth)] = float(sst.spearmanr(S["M_foldmap"][k], V[k])[0] - sst.spearmanr(S["BL"][k], V[k])[0])
out["dev_lomo_Mf_minus_BL_range"] = [min(lomo.values()), max(lomo.values())]
out["dev_lomo_top"] = sorted(lomo.items(), key=lambda kv: kv[1])[-3:]
np.savez("/tmp/claude-0/v3/critic/mkt_d5.npz", V=V, bk=bk, **S)
json.dump(out, open("/tmp/claude-0/v3/critic/mkt_check.json", "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
