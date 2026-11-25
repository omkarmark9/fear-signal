"""Score the 4 declared candidates (candidates.md) on dev walk-forward OOS; then 2026 post hoc."""
import json
import numpy as np
import pandas as pd
from scipy import stats as sst
from ma_common import *  # noqa

CANDS = {"C1_breadth_lin": "B", "C2_sev_mean": "sev", "C3_meanZ_fixed": "Zbar", "C4_sev_logit_mean": "Lbar"}
res = {"dev": {}, "post_hoc_2026": {}}

# =============================================================== DEV (selection evidence)
d, dates, vxn, folds = load_dev()
blk = d["block"]
T = len(dates)
series = {"baseline_BL": d["BL"].copy(), "ref_M_foldmap": d["M"].copy(), "ref_Mleg": d["Mleg"].copy()}
for nm in list(CANDS) + ["sens_BL_std"]:
    series[nm] = np.full(T, np.nan)
consts = {}
for b, ff in enumerate(folds):
    ag = aggregates(ff)
    test = blk == b
    train = ff["train"]
    for nm, key in list(CANDS.items()) + [("sens_BL_std", "BL")]:
        z, mu, sd = standardise(ag[key], train)
        series[nm][test] = z[test]
        consts.setdefault(nm, []).append({"mu": mu, "sd": sd})
    # sanity: fold aggregate B equals stitched B on test days
    assert np.allclose(ag["B"][test], d["B"][test], equal_nan=True)
res["dev"]["standardisation_constants"] = consts

d5 = {k: evaluate.d5_within_blocks(s, vxn, blk) for k, s in series.items()}
scored = blk >= 0
res["dev"]["d5"] = {k: corr_pair(v["s"], v["v"]) for k, v in d5.items()}
res["dev"]["level_spearman"] = {}
res["dev"]["level_pearson"] = {}
for k, s in series.items():
    lv = pd.DataFrame({"s": s, "v": vxn})[scored].dropna()
    res["dev"]["level_spearman"][k] = stats.corr_ci(lv.s, lv.v, "spearman", block=60, n_boot=1000)
    res["dev"]["level_pearson"][k] = float(np.corrcoef(lv.s, lv.v)[0, 1])


def pair(dd, a, b, **kw):
    j = dd[a].index.intersection(dd[b].index)
    return boot_pair(dd[a].loc[j, "s"], dd[b].loc[j, "s"], dd[a].loc[j, "v"], **kw)


res["dev"]["vs_baseline"] = {k: pair(d5, k, "baseline_BL") for k in list(CANDS) + ["sens_BL_std", "ref_M_foldmap", "ref_Mleg"]}
res["dev"]["vs_legacy"] = {k: pair(d5, k, "ref_Mleg") for k in list(CANDS) + ["baseline_BL"]}
# block-length sensitivity for the primary comparison
res["dev"]["vs_baseline_block10"] = {k: pair(d5, k, "baseline_BL", block=10.0)["spearman"] for k in CANDS}
res["dev"]["vs_baseline_block40"] = {k: pair(d5, k, "baseline_BL", block=40.0)["spearman"] for k in CANDS}
# per-fold Spearman and difference vs baseline
pf = {}
for b in range(3):
    row = {}
    for k, v in d5.items():
        idx = v.index[blk[v.index] == b]
        row[k] = float(sst.spearmanr(v.loc[idx, "s"], v.loc[idx, "v"])[0])
    row["n"] = int((blk[d5["baseline_BL"].index] == b).sum())
    pf[f"fold{b}"] = row
res["dev"]["per_fold_spearman"] = pf
res["dev"]["per_fold_diff_vs_baseline"] = {
    k: [pf[f"fold{b}"][k] - pf[f"fold{b}"]["baseline_BL"] for b in range(3)] for k in CANDS}

# =============================================================== 2026 (post hoc diagnostic only)
f, fd, fv = load_full()
el_days = f["el"].any(axis=1)
m26 = np.asarray(fd >= pd.Timestamp("2026-01-01")) & el_days
devm = np.asarray(fd <= DEV_END) & el_days
ag = aggregates(f)
s26 = {"baseline_BL": ag["BL"], "v2_M_frozen": f["M"], "v1pub_Mleg": f["Mleg_v1"], "v2filter_Mleg": f["Mleg"]}
c26 = {}
for nm, key in CANDS.items():
    z, mu, sd = standardise(ag[key], devm)
    s26[nm] = z
    c26[nm] = {"mu": mu, "sd": sd}
res["post_hoc_2026"]["standardisation_constants_dev"] = c26
e5 = {k: d5_full(s, fv, fd, m26) for k, s in s26.items()}
res["post_hoc_2026"]["d5"] = {k: corr_pair(v["s"], v["v"]) for k, v in e5.items()}
res["post_hoc_2026"]["level_spearman"] = {}
for k, s in s26.items():
    lv = pd.DataFrame({"s": s, "v": fv})[m26].dropna()
    res["post_hoc_2026"]["level_spearman"][k] = stats.corr_ci(lv.s, lv.v, "spearman", block=60, n_boot=2000)
res["post_hoc_2026"]["vs_baseline_BL"] = {k: pair(e5, k, "baseline_BL") for k in CANDS}
res["post_hoc_2026"]["vs_v2_M_frozen"] = {k: pair(e5, k, "v2_M_frozen") for k in list(CANDS) + ["baseline_BL"]}
res["post_hoc_2026"]["vs_v1pub_Mleg (H1b analogue)"] = {k: pair(e5, k, "v1pub_Mleg") for k in list(CANDS) + ["baseline_BL"]}
# alerts: days with candidate >= its dev 97.72th percentile (the M>=2 analogue) -- descriptive
thr = {}
for k, s in s26.items():
    if k.startswith("v1pub") or k.startswith("v2filter"):
        continue
    q = np.nanquantile(s[devm], 0.97725)
    thr[k] = {"dev_q97.7": float(q), "n_2026_days_above": int(np.nansum(s[m26] >= q))}
res["post_hoc_2026"]["alert_analogue_days"] = thr

json.dump(res, open(f"{OUT}/score_results.json", "w"), indent=1)


def show(title, dd):
    print(f"\n== {title}")
    for k, v in dd.items():
        sp, pe = v["spearman"], v["pearson"]
        print(f"{k:22s} dSpearman {sp['diff']:+.4f} (se {sp['se']:.4f}, z {sp['z']:+.2f}, CI {sp['lo']:+.3f},{sp['hi']:+.3f})"
              f" | dPearson {pe['diff']:+.4f} (se {pe['se']:.4f}, z {pe['z']:+.2f}) n={v['n']}")


print("DEV d5:", json.dumps(res["dev"]["d5"], indent=0))
print("DEV level spearman:", {k: round(v["r"], 3) for k, v in res["dev"]["level_spearman"].items()})
print("DEV level pearson:", {k: round(v, 3) for k, v in res["dev"]["level_pearson"].items()})
show("DEV vs baseline BL", res["dev"]["vs_baseline"])
show("DEV vs legacy", res["dev"]["vs_legacy"])
print("block10:", {k: (round(v['diff'], 4), round(v['se'], 4)) for k, v in res['dev']['vs_baseline_block10'].items()})
print("block40:", {k: (round(v['diff'], 4), round(v['se'], 4)) for k, v in res['dev']['vs_baseline_block40'].items()})
print("per fold:", json.dumps(res["dev"]["per_fold_spearman"], indent=0))
print("per fold diff:", res["dev"]["per_fold_diff_vs_baseline"])
print("\n#### POST HOC 2026 (diagnostic only) ####")
print("2026 d5:", json.dumps(res["post_hoc_2026"]["d5"], indent=0))
print("2026 level spearman:", {k: (round(v["r"], 3), round(v["lo"], 3), round(v["hi"], 3)) for k, v in res["post_hoc_2026"]["level_spearman"].items()})
show("2026 vs baseline BL", res["post_hoc_2026"]["vs_baseline_BL"])
show("2026 vs v2 M frozen", res["post_hoc_2026"]["vs_v2_M_frozen"])
show("2026 vs v1 published Mleg", res["post_hoc_2026"]["vs_v1pub_Mleg (H1b analogue)"])
print(res["post_hoc_2026"]["alert_analogue_days"])
