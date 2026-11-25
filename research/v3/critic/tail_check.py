"""Critic: re-derive the tail-calibration slopes (pooled, per fold, within/between-day) on dev OOS and 2026."""
import json
import numpy as np
from fearsignal import stats
z = np.load("/tmp/claude-0/v3/tail-calibration/cache.npz", allow_pickle=True)
def logit(p): p = np.clip(p, 1e-9, 1 - 1e-9); return np.log(p / (1 - p))
def irls(X, y, g):
    b = np.zeros(X.shape[1]); b[1:] = 1
    for _ in range(60):
        p = 1 / (1 + np.exp(-(X @ b))); W = p * (1 - p)
        b = b + np.linalg.solve(X.T @ (X * W[:, None]), X.T @ (y - p))
    p = 1 / (1 + np.exp(-(X @ b))); Hn = np.linalg.inv(X.T @ (X * (p * (1 - p))[:, None]))
    sc = X * (y - p)[:, None]; G = np.zeros((X.shape[1], X.shape[1]))
    for u in np.unique(g):
        s = sc[g == u].sum(0); G += np.outer(s, s)
    return b, np.sqrt(np.diag(Hn @ G @ Hn))
def prep(q, y, mask_days):
    T, N = q.shape; day = np.repeat(np.arange(T)[:, None], N, 1)
    ok = np.isfinite(q) & np.isfinite(y) & mask_days[:, None]
    return q[ok], y[ok], day[ok]
out = {}
dq, dy, blk = z["dev_q"], z["dev_y"], z["dev_block"]
sets = {"dev_all": blk >= 0, "foldA": blk == 0, "foldB": blk == 1, "foldC": blk == 2, "h2026": z["h_mask"]}
for nm, m in sets.items():
    q, y, g = prep(z["h_q"] if nm == "h2026" else dq, z["h_y"] if nm == "h2026" else dy, m)
    L = logit(q)
    b, se = irls(np.column_stack([np.ones_like(L), L]), y, g)
    Lbar = np.zeros_like(L)
    for u in np.unique(g):
        k = g == u; Lbar[k] = L[k].mean()
    b2, se2 = irls(np.column_stack([np.ones_like(L), Lbar, L - Lbar]), y, g)
    out[nm] = {"n": int(len(y)), "days": int(len(np.unique(g))), "slope": float(b[1]), "se": float(se[1]),
               "between": float(b2[1]), "se_between": float(se2[1]), "within": float(b2[2]), "se_within": float(se2[2]),
               "mean_q": float(q.mean()), "mean_y": float(y.mean())}
    print(nm, out[nm], flush=True)
# heterogeneity across dev folds (Cochran Q on slopes)
s = np.array([out[f]["slope"] for f in ("foldA", "foldB", "foldC")]); w = 1 / np.array([out[f]["se"] for f in ("foldA", "foldB", "foldC")]) ** 2
pw = (s * w).sum() / w.sum(); Q = float(((s - pw) ** 2 * w).sum())
out["cochranQ"] = Q; out["precision_weighted_slope"] = float(pw); out["pw_se"] = float(1 / np.sqrt(w.sum()))
json.dump(out, open("/tmp/claude-0/v3/critic/tail_check.json", "w"), indent=1)
print(json.dumps({k: out[k] for k in ("cochranQ", "precision_weighted_slope", "pw_se")}))
