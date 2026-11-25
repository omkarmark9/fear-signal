"""Critic: independent re-implementation of the H4 dev walk-forward (B0 per-fold fixed vs
C1 recursive expanding), look-ahead audit, and the P1 family gate on {C1, C2, C3}."""
import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, "/tmp/claude-0/v3/critic")
import spa
from fearsignal import validate, stats
from fearsignal.evaluate import FOLDS
c = np.load("/tmp/claude-0/v3/volatility-increment/cache.npz")
dates = pd.DatetimeIndex(c["dates_dev"]); R = c["R_dev"]; el = c["el_dev"]; T = len(R); H = 5
y, X0 = validate.har_design(R)
# audit the target: y_t = 0.5 ln mean(R_{t+1..t+5}^2)
t = 300; chk = 0.5 * np.log(np.mean(R[t+1:t+6] ** 2)); assert abs(chk - y[t]) < 1e-5, (chk, y[t])
ols = lambda yy, XX: np.linalg.lstsq(XX, yy, rcond=None)[0]
fc = {k: np.full(T, np.nan) for k in ("B0", "C1", "C2", "C3", "B0h", "C1h")}
rows = np.zeros(T, bool); blk = np.full(T, -1)
for f, (tr_end, te_end) in enumerate(FOLDS):
    M, BL = c[f"M_{f}"], c[f"BL_{f}"]
    X1 = np.column_stack([X0, M]); X3 = np.column_stack([X0, BL])
    ok = el & np.isfinite(y) & np.isfinite(X1).all(1) & np.isfinite(BL)
    iend = np.flatnonzero(dates <= pd.Timestamp(tr_end))[-1]
    tr = ok & (np.arange(T) <= iend - H)
    te = ok & (dates > pd.Timestamp(tr_end)) & (dates <= pd.Timestamp(te_end))
    rows |= te; blk[te] = f
    b1 = ols(y[tr], X1[tr]); fc["B0"][te] = X1[te] @ b1
    fc["B0h"][te] = X0[te] @ ols(y[tr], X0[tr])
    fc["C3"][te] = X3[te] @ ols(y[tr], X3[tr])
    m = X0[:, 3]; Xl = np.column_stack([np.ones(T), X0[:, 1] - m, X0[:, 2] - m, M])
    fc["C2"][te] = Xl[te] @ ols(y[tr] - m[tr], Xl[tr]) + m[te]
    first = np.flatnonzero(tr)[0]
    pool = ok & (np.arange(T) >= first)
    for tt in np.flatnonzero(te):
        r_ = pool & (np.arange(T) <= tt - H)        # targets fully observed by close of tt
        fc["C1"][tt] = X1[tt] @ ols(y[r_], X1[r_])
        fc["C1h"][tt] = X0[tt] @ ols(y[r_], X0[r_])
e2 = {k: (y - v) ** 2 for k, v in fc.items()}
s = np.load("/tmp/claude-0/v3/volatility-increment/dev_stitched.npz")
out = {"n": int(rows.sum()),
       "max_abs_diff_vs_agent": {k: float(np.nanmax(np.abs(fc[k][rows] - s[a][rows]))) for k, a in
                                 (("B0", "B0_HAR_p_M"), ("C1", "C1_HAR_p_M"), ("C2", "C2_HAR_p_M"), ("C3", "B0_HAR_p_BL"))}}
def cmp(a, b, mask=rows):
    d = (e2[a] - e2[b])[mask]
    dm = stats.diebold_mariano(d, H)
    return {"gain": float(d.mean()), "t_hln": float(dm["t"]), "t_nw10": float(d.mean() / stats.newey_west_se_mean(d, 10)),
            "per_fold": [float((e2[a] - e2[b])[mask & (blk == k)].mean()) for k in range(3)]}
out["B0->C1"] = cmp("B0", "C1"); out["B0->C2"] = cmp("B0", "C2"); out["B0->C3"] = cmp("B0", "C3")
out["B0->C1, folds B+C only"] = cmp("B0", "C1", rows & (blk >= 1))
out["B0->C1, fold C only"] = cmp("B0", "C1", rows & (blk == 2))
out["H4 inside C1 (C1h->C1)"] = cmp("C1h", "C1"); out["H4 inside B0 (B0h->B0)"] = cmp("B0h", "B0")
# fold-A training-window size and coefficient pathology
out["n_train_by_fold"] = []
# P1 gate: family {C1, C2, C3} vs incumbent B0 (positive = candidate better), daily loss diffs
D = np.column_stack([(e2["B0"] - e2[k])[rows] for k in ("C1", "C2", "C3")])
g = spa.spa_calibrated(D, block=10.0, B=999, R=499, seed=0)
out["P1_gate_vol_family"] = {"t": g["t"].tolist(), "p_c_raw": g["p_c_raw"], "p_c_calibrated": g["p_c_calibrated"],
                             "by_world": g["p_c_calibrated_by_world"], "alpha_nominal_05": g["alpha_nominal_05"]}
Dbc = np.column_stack([(e2["B0"] - e2[k])[rows & (blk >= 1)] for k in ("C1", "C2", "C3")])
g2 = spa.spa_calibrated(Dbc, block=10.0, B=999, R=499, seed=1)
out["P1_gate_vol_family_folds_B_C"] = {"t": g2["t"].tolist(), "p_c_raw": g2["p_c_raw"], "p_c_calibrated": g2["p_c_calibrated"]}
er = spa.episode_robustness(D[:, 0], n_top=5, halfwidth=2, nw_lags=10)
out["C1_episode_robustness"] = {"share_ss_top5": er["share_ss_top"], "min_t_drop_one": er["min_t_drop_one"], "drop_all": er["drop_all"],
                                "top_days": [str(dates[np.flatnonzero(rows)[r["day_index"]]].date()) for r in er["drop_one"]]}
json.dump(out, open("/tmp/claude-0/v3/critic/vol_check.json", "w"), indent=1)
print(json.dumps(out, indent=1))
