"""Robustness checks on the dev result (no new candidates) + coefficient look-ups for mechanism."""
import json
import numpy as np, pandas as pd
from fearsignal import stats, validate
from fearsignal.evaluate import FOLDS
OUT = "/tmp/claude-0/v3/volatility-increment"
s = np.load(f"{OUT}/dev_stitched.npz")
y, rows, blocks = s["y"], s["rows"], s["blocks"]
def d(a, b):
    return ((y - s[a]) ** 2 - (y - s[b]) ** 2)
out = {}
for a, b in (("B0_HAR_p_M", "C1_HAR_p_M"), ("B0_HAR_p_M", "C2_HAR_p_M"), ("B0_HAR", "B0_HAR_p_M"), ("C1_HAR", "C1_HAR_p_M")):
    dd = d(a, b)
    r = {}
    for L in (5, 10, 20):
        se = stats.newey_west_se_mean(dd[rows], L)
        r[f"bartlett{L}_t"] = float(dd[rows].mean() / se)
    for k in range(3):
        m = rows & (blocks != k)
        dm = stats.diebold_mariano(dd[m], 5)
        r[f"drop_fold{k}"] = {"gain": dm["mean"], "t": dm["t"]}
    # block-level: one number per 5-day non-overlapping window (Hansen-style crude)
    idx = np.flatnonzero(rows)
    nonover = idx[::5]
    r["nonoverlap_every5_mean_t"] = float(dd[nonover].mean() / (dd[nonover].std(ddof=1) / np.sqrt(len(nonover))))
    out[f"{a} -> {b}"] = r
# coefficients
c = np.load(f"{OUT}/cache.npz")
dates = pd.DatetimeIndex(c["dates_dev"]); R = c["R_dev"]; el = c["el_dev"]
yy, X0 = validate.har_design(R)
co = {}
for f, (tr_end, te_end) in enumerate(FOLDS):
    M = c[f"M_{f}"]
    idx_end = np.flatnonzero(dates <= pd.Timestamp(tr_end))[-1]
    ok = el & np.isfinite(yy) & np.isfinite(X0).all(1) & np.isfinite(M) & (np.arange(len(yy)) <= idx_end - 5)
    X1 = np.column_stack([X0, M])
    r1 = stats.ols_hac(yy[ok], X1[ok], 5)
    m = X0[:, 3]
    Xl = np.column_stack([np.ones(len(yy)), X0[:, 1] - m, X0[:, 2] - m, M])
    r2 = stats.ols_hac(yy[ok] - m[ok], Xl[ok], 5)
    co[f"fold{f}"] = {"B0_HAR+M_beta": np.round(r1["beta"], 3).tolist(), "B0_t_M": round(float(r1["t"][-1]), 2),
                      "slope_sum": round(float(r1["beta"][1:4].sum()), 3),
                      "C2_HAR+M_beta": np.round(r2["beta"], 3).tolist(), "C2_t_M": round(float(r2["t"][-1]), 2),
                      "corr_M_with_RV22": round(float(np.corrcoef(M[ok], X0[ok, 3])[0, 1]), 3),
                      "corr_M_with_RV5_minus_RV22": round(float(np.corrcoef(M[ok], X0[ok, 2] - X0[ok, 3])[0, 1]), 3)}
out["coefficients"] = co
# power of the 2026 H4 test given dev OOS effect sizes (normal approx to t)
from scipy.stats import norm
pw = {}
for k, (g, se, n) in {"B0": (0.01531, 0.00914, 373), "C1": (0.01243, 0.00947, 373)}.items():
    et = g / (se * np.sqrt(n / 220))
    pw[k] = {"expected_t_at_n220": round(float(et), 2), "power_one_sided_5pct": round(float(1 - norm.cdf(1.653 - et)), 2)}
out["h4_power_2026"] = pw
json.dump(out, open(f"{OUT}/checks.json", "w"), indent=1)
print(json.dumps(out, indent=1))
