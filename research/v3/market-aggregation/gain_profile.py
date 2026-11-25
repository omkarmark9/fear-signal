"""Mechanism check: level-dependent gain of each market reading per unit change in breadth,
against VXN's own gain per unit breadth change. Dev OOS (selection period) and 2026 (diagnostic)."""
import json
import numpy as np
import pandas as pd
from ma_common import *  # noqa

BINS = [0, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0]


def profile(series, B, vxn, mask, block=None):
    """slope of d5(series) on d5(B) within bins of the window-mean breadth level."""
    def d5(x):
        s = pd.Series(x).diff(5)
        if block is not None:
            b = pd.Series(block)
            s[~(b.eq(b.shift(5)) & (b >= 0))] = np.nan
        return s.to_numpy()
    dB = d5(B)
    lvl = pd.Series(B).rolling(6).mean().to_numpy()
    out = {}
    for nm, s in list(series.items()) + [("VXN", vxn)]:
        ds = d5(s)
        row = []
        for lo, hi in zip(BINS[:-1], BINS[1:]):
            w = mask & np.isfinite(dB) & np.isfinite(ds) & (lvl >= lo) & (lvl < hi)
            row.append(float(np.polyfit(dB[w], ds[w], 1)[0]) if w.sum() >= 10 else np.nan)
        row = np.array(row)
        out[nm] = row
    # normalise each profile to its own mean so shapes are comparable
    norm = {k: (v / np.nanmean(v)).round(2).tolist() for k, v in out.items()}
    shape_corr = {k: float(pd.Series(v).corr(pd.Series(out["VXN"]))) for k, v in out.items() if k != "VXN"}
    n = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        n.append(int((mask & (lvl >= lo) & (lvl < hi)).sum()))
    return {"bins": [f"[{a},{b})" for a, b in zip(BINS[:-1], BINS[1:])], "n_days": n,
            "relative_gain": norm, "shape_corr_with_VXN_gain": shape_corr}


res = {}
d, dates, vxn, folds = load_dev()
blk = d["block"]
res["dev_oos"] = profile({"B": d["B"], "BL": d["BL"], "M_foldmap": d["M"], "Mleg": d["Mleg"]},
                         d["B"], vxn, blk >= 0, block=blk)
# block-length sensitivity of the pre-declared reference comparison (as-frozen M vs baseline)
d5 = {k: evaluate.d5_within_blocks(s, vxn, blk) for k, s in {"BL": d["BL"], "M": d["M"]}.items()}
j = d5["BL"].index.intersection(d5["M"].index)
res["dev_M_foldmap_minus_BL_by_block"] = {
    str(bl): boot_pair(d5["M"].loc[j, "s"], d5["BL"].loc[j, "s"], d5["BL"].loc[j, "v"], block=bl)["spearman"]
    for bl in (10.0, 20.0, 40.0)}

f, fd, fv = load_full()
m26 = np.asarray(fd >= pd.Timestamp("2026-01-01")) & f["el"].any(axis=1)
res["post_hoc_2026"] = profile({"B": f["B"], "BL": f["BL"], "M": f["M"], "Mleg_v1pub": f["Mleg_v1"]},
                               f["B"], fv, m26)
json.dump(res, open(f"{OUT}/gain_profile.json", "w"), indent=1)
print(json.dumps(res, indent=1))
