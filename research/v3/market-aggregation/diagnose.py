"""Diagnosis of the H1b loss (2026 used for diagnosis only) + dev-chain check.

Only the frozen chain (B -> logit B -> M) and the legacy index are scored
against VXN here; the declared candidates are scored in score.py.
"""
import json
import numpy as np
import pandas as pd
from scipy import stats as sst
from ma_common import *  # noqa
from fearsignal import frozen, scoring, market

res = {}
# ------------------------------------------------------------------ 2026 chain
f, dates, vxn = load_full()
m26 = np.asarray(dates >= pd.Timestamp("2026-01-01")) & f["el"].any(axis=1)
devm = np.asarray(dates <= DEV_END) & f["el"].any(axis=1)
chain26 = {"B": f["B"], "BL": f["BL"], "M": f["M"], "Mleg_v2filter": f["Mleg"], "Mleg_v1pub": f["Mleg_v1"]}
d5 = {k: d5_full(s, vxn, dates, m26) for k, s in chain26.items()}
res["2026_chain_d5"] = {k: corr_pair(v["s"], v["v"]) for k, v in d5.items()}
lv = {k: pd.DataFrame({"s": s, "v": vxn})[m26].dropna() for k, s in chain26.items()}
res["2026_chain_level_spearman"] = {k: float(sst.spearmanr(v.s, v.v)[0]) for k, v in lv.items()}
res["2026_chain_level_pearson"] = {k: float(np.corrcoef(v.s, v.v)[0, 1]) for k, v in lv.items()}


def pair(dd, a, b):
    j = dd[a].index.intersection(dd[b].index)
    return boot_pair(dd[a].loc[j, "s"], dd[b].loc[j, "s"], dd[a].loc[j, "v"])


res["2026_pairs"] = {
    "M_minus_Mleg_v1pub (H1b repro)": pair(d5, "M", "Mleg_v1pub"),
    "M_minus_BL (percentile map effect)": pair(d5, "M", "BL"),
    "BL_minus_B (logit effect)": pair(d5, "BL", "B"),
    "BL_minus_Mleg_v1pub (baseline vs v1)": pair(d5, "BL", "Mleg_v1pub"),
    "Mleg_v2filter_minus_Mleg_v1pub (filter effect)": pair(d5, "Mleg_v2filter", "Mleg_v1pub"),
}

# ------------------------------------------------------------------ dev OOS chain (status quo only)
d, ddates, dvxn, folds = load_dev()
blk = d["block"]
dev_chain = {"BL": d["BL"], "M_foldmap": d["M"], "Mleg": d["Mleg"]}
dd5 = {k: evaluate.d5_within_blocks(s, dvxn, blk) for k, s in dev_chain.items()}
res["dev_oos_chain_d5"] = {k: corr_pair(v["s"], v["v"]) for k, v in dd5.items()}
res["dev_oos_pairs"] = {
    "BL_minus_Mleg (committed +0.037 repro)": pair(dd5, "BL", "Mleg"),
    "M_foldmap_minus_BL (percentile map effect, OOS)": pair(dd5, "M_foldmap", "BL"),
    "M_foldmap_minus_Mleg (what v2 froze vs legacy, OOS)": pair(dd5, "M_foldmap", "Mleg"),
}
# per-fold
pf = {}
for b in range(3):
    row = {}
    for k, v in dd5.items():
        idx = v.index[blk[v.index] == b]
        row[k] = corr_pair(v.loc[idx, "s"], v.loc[idx, "v"])["spearman"]
    pf[f"fold{b}"] = row
res["dev_oos_chain_per_fold_spearman"] = pf

# ------------------------------------------------------------------ map gain profile (frozen map)
spec = frozen.load()
mmap = scoring.QuantileMap.from_dict(spec["m_map"])
grid = np.linspace(0.05, 0.97, 400)
g = mmap(scoring.logit(grid))
gp = np.gradient(g, grid)
Bd, B26 = f["B"][devm], f["B"][m26]
bins = [0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
prof = []
dB = pd.Series(f["B"]).diff(5).to_numpy()
dV = pd.Series(vxn).diff(5).to_numpy()
Bmid = pd.Series(f["B"]).rolling(6).mean().to_numpy()  # level over the change window
for lo, hi in zip(bins[:-1], bins[1:]):
    sel = (grid >= lo) & (grid < hi)
    row = {"bin": f"[{lo},{hi})",
           "share_dev_days": float(((Bd >= lo) & (Bd < hi)).mean()),
           "share_2026_days": float(((B26 >= lo) & (B26 < hi)).mean()),
           "dM_dB_frozen_map": float(np.median(gp[sel])) if sel.any() else None,
           "dlogit_dB": float(np.median(1 / (grid[sel] * (1 - grid[sel])))) if sel.any() else None}
    for nm, mk in (("dev", devm), ("2026", m26)):
        w = mk & (Bmid >= lo) & (Bmid < hi) & np.isfinite(dB) & np.isfinite(dV)
        if w.sum() >= 8:
            row[f"vxn_per_B_slope_{nm}"] = float(np.polyfit(dB[w], dV[w], 1)[0])
            row[f"n_{nm}"] = int(w.sum())
    prof.append(row)
res["map_gain_profile"] = prof
res["frozen_m_map"] = {"logitB_range": [float(mmap.knots_L[0]), float(mmap.knots_L[-1])],
                       "B_range": [float(1 / (1 + np.exp(-mmap.knots_L[0]))), float(1 / (1 + np.exp(-mmap.knots_L[-1])))],
                       "slope_hi": mmap.slope_hi,
                       "2026_days_below_dev_min": int((f["BL"][m26] < mmap.knots_L[0]).sum()),
                       "2026_days_above_dev_max": int((f["BL"][m26] > mmap.knots_L[-1]).sum())}
# how much of the M-vs-B discrepancy in 2026 changes is explained by the map's level-dependent gain
gB = np.interp(f["B"], grid, gp)
res["2026_gain_on_days"] = {"median_dM_dB_2026": float(np.nanmedian(gB[m26])), "median_dM_dB_dev": float(np.nanmedian(gB[devm])),
                            "iqr_ratio_2026": float(np.nanquantile(gB[m26], .9) / np.nanquantile(gB[m26], .1))}

# ------------------------------------------------------------------ expanding yardstick (legacy) in 2026
Zbar = rowmean(f["Z_v1"])  # the v1-published filter's mean Z (the H1b comparator)
mu = np.full(len(Zbar), np.nan); sd = np.full(len(Zbar), np.nan)
hist = []
for t, v in enumerate(Zbar):
    if np.isfinite(v) and len(hist) >= 20:
        h = np.asarray(hist); mu[t] = h.mean(); sd[t] = h.std(ddof=1)
    if np.isfinite(v):
        hist.append(v)
Mleg = (Zbar - mu) / sd
print("Mleg recompute max diff (float32 Z):", np.nanmax(np.abs(Mleg - f["Mleg_v1"])))
dM = pd.Series(Mleg).diff(5).to_numpy()
pure = (Zbar - pd.Series(Zbar).shift(5).to_numpy()) / sd          # change in mean Z at today's yardstick
drift = dM - pure                                                 # yardstick drift part
w = m26 & np.isfinite(dM)
res["legacy_yardstick_2026"] = {
    "corr_dMleg_vs_dZbar_over_sd": float(np.corrcoef(dM[w], pure[w])[0, 1]),
    "var_share_drift": float(np.var(drift[w]) / np.var(dM[w])),
    "sd_t_start_2026": float(sd[m26][0]), "sd_t_end_2026": float(sd[m26][-1]),
    "mu_t_start_2026": float(mu[m26][0]), "mu_t_end_2026": float(mu[m26][-1]),
    "Mleg_mean_2026": float(np.nanmean(Mleg[m26])),
}

# ------------------------------------------------------------------ per-stock saturation, severity gradient
el = f["el"]
pnc = 1 - f["P_calm"]
def sat(mask):
    m = el & mask[:, None]
    v = pnc[m]
    return {"share_Pnotcalm_gt_0.95": float((v > .95).mean()), "share_gt_0.99": float((v > .99).mean()),
            "share_lt_0.05": float((v < .05).mean()),
            "mean_Ppanic_over_B": float(np.nanmean(f["P_panic"][m]) / np.nanmean(v))}
res["per_stock_saturation"] = {"dev_frozenmodel": sat(devm), "2026": sat(m26)}
# within-day dispersion of severity among not-calm mass (gradient breadth throws away)
sev = rowmean(f["sev"])
for nm, mk in (("dev", devm), ("2026", m26)):
    ds = pd.Series(sev).diff(5).to_numpy(); db = dB
    w = mk & np.isfinite(ds) & np.isfinite(db)
    res.setdefault("dsev_on_dB_R2", {})[nm] = float(np.corrcoef(ds[w], db[w])[0, 1] ** 2)

# ------------------------------------------------------------------ where in 2026 does M lose to v1? (leave-one-month-out)
a, b = d5["M"], d5["Mleg_v1pub"]
j = a.index.intersection(b.index)
A = a.loc[j]; Bv = b.loc[j]
months = A.index.to_period("M")
full = sst.spearmanr(A.s, A.v)[0] - sst.spearmanr(Bv.s, Bv.v)[0]
lomo = {}
for mth in months.unique():
    k = months != mth
    lomo[str(mth)] = float(sst.spearmanr(A.s[k], A.v[k])[0] - sst.spearmanr(Bv.s[k], Bv.v[k])[0] - full)
res["2026_H1b_leave_one_month_out_shift"] = lomo
# same for M vs B (map effect)
Bb = d5["B"].loc[j]
fullmb = sst.spearmanr(A.s, A.v)[0] - sst.spearmanr(Bb.s, Bb.v)[0]
res["2026_M_minus_B_leave_one_month_out_shift"] = {
    str(mth): float(sst.spearmanr(A.s[months != mth], A.v[months != mth])[0]
                    - sst.spearmanr(Bb.s[months != mth], Bb.v[months != mth])[0] - fullmb)
    for mth in months.unique()}

json.dump(res, open(f"{OUT}/diagnosis.json", "w"), indent=1)
print(json.dumps(res, indent=1))
