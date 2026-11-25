"""Dev walk-forward scoring of B0 and C1-C3 (declared in candidates.md), then 2026 post-hoc.

Selection uses only the dev section. The 2026 section is computed afterwards and
is a post-hoc diagnostic: it is never fed back into the decision.
"""
import json
import numpy as np
import pandas as pd
from fearsignal import stats, validate
from fearsignal.evaluate import FOLDS

OUT = "/tmp/claude-0/v3/volatility-increment"
c = np.load(f"{OUT}/cache.npz")
H = 5


def ols(y, X):
    return np.linalg.lstsq(X, y, rcond=None)[0]


def lf_design(X0, extra=None):
    """Level-free HAR: y - RV22 on const, RV1 - RV22, RV5 - RV22 [, extra]."""
    m = X0[:, 3]
    cols = [np.ones(len(X0)), X0[:, 1] - m, X0[:, 2] - m]
    if extra is not None:
        cols.append(extra)
    return np.column_stack(cols), m


def forecasts(R, xvars, train_rows, test_rows, pool):
    """All forecasts for one training window / test set.

    xvars: dict name -> extra regressor array (M, BL) for this fold.
    Returns dict design -> forecast array (NaN off test rows), plus the train mean of y.
    """
    T = len(R)
    y, X0 = validate.har_design(R)
    fc = {}
    tr = train_rows
    # B0 / C3: fixed coefficients
    fc["B0_HAR"] = np.full(T, np.nan)
    fc["B0_HAR"][test_rows] = X0[test_rows] @ ols(y[tr], X0[tr])
    for nm, v in xvars.items():
        X1 = np.column_stack([X0, v])
        fc[f"B0_HAR+{nm}"] = np.full(T, np.nan)
        fc[f"B0_HAR+{nm}"][test_rows] = X1[test_rows] @ ols(y[tr], X1[tr])
    # C2 level-free, fixed coefficients
    for nm, v in [(None, None)] + [("M", xvars["M"])]:
        Xl, m = lf_design(X0, v)
        b = ols(y[tr] - m[tr], Xl[tr])
        key = "C2_HAR" if nm is None else f"C2_HAR+{nm}"
        fc[key] = np.full(T, np.nan)
        fc[key][test_rows] = Xl[test_rows] @ b + m[test_rows]
    # C1 recursive expanding window: refit each test day t on rows s <= t - H
    first_train = np.flatnonzero(tr)[0]
    base_ok = pool & np.isfinite(y) & np.all(np.isfinite(X0), 1)
    for nm, v in [(None, None), ("M", xvars["M"])]:
        X = X0 if v is None else np.column_stack([X0, v])
        ok_all = base_ok & np.isfinite(X).all(1) & train_rows_mask_from(tr, first_train, T)
        key = "C1_HAR" if nm is None else f"C1_HAR+{nm}"
        fc[key] = np.full(T, np.nan)
        for t in np.flatnonzero(test_rows):
            rows = ok_all.copy()
            rows[t - H + 1:] = False
            fc[key][t] = X[t] @ ols(y[rows], X[rows])
    return fc, y, float(y[tr].mean())


def train_rows_mask_from(tr, first, T):
    """Eligible-day mask from the first training row onward (the recursive window grows from it)."""
    m = np.zeros(T, bool)
    m[first:] = True
    return m


def rows_for(dates, el, y, X, tr_end, te_end=None):
    idx_end = np.flatnonzero(dates <= pd.Timestamp(tr_end))[-1]
    ok = el & np.isfinite(y) & np.all(np.isfinite(X), 1)
    train = ok & (np.arange(len(dates)) <= idx_end - H)
    test = None
    if te_end is not None:
        test = ok & (dates > pd.Timestamp(tr_end)) & (dates <= pd.Timestamp(te_end))
    return train, test


def boot_ci(d, n_boot=2000, block=20, seed=0):
    rng = np.random.default_rng(seed)
    d = d[np.isfinite(d)]
    bs = np.array([d[stats.stationary_bootstrap_indices(len(d), block, rng)].mean() for _ in range(n_boot)])
    return [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975)), float(bs.std())]


def compare(y, fa, fb, rows, blocks):
    """Loss differential d = e_a^2 - e_b^2 (positive: b better)."""
    ea, eb = y - fa, y - fb
    d = (ea**2 - eb**2)[rows]
    dm = stats.diebold_mariano(d, H)
    hln_se = dm["mean"] / dm["t"] if dm["t"] != 0 else np.nan
    per = {f"fold{k}": float(np.mean((ea**2 - eb**2)[rows & (blocks == k)])) for k in range(3)}
    ba = np.mean(ea[rows]) ** 2
    bb = np.mean(eb[rows]) ** 2
    return {"mean_gain": dm["mean"], "dm_hln_t": dm["t"], "p_one_sided": dm["p_one_sided"],
            "hln_se": float(hln_se), "n": dm["n"], "boot95_and_se": boot_ci(d),
            "per_fold_gain": per, "folds_positive": int(sum(v > 0 for v in per.values())),
            "gain_from_bias_term": float(ba - bb),
            "gain_from_variance_term": float(np.var(ea[rows]) - np.var(eb[rows]))}


def score_set(y, fc, rows, ybar_train, blocks):
    out = {}
    for k, f in fc.items():
        e = (y - f)[rows]
        sst_o = np.sum((y[rows] - y[rows].mean()) ** 2)
        sst_f = np.sum((y[rows] - ybar_train[rows]) ** 2)
        out[k] = {"mse": float(np.mean(e**2)), "bias": float(e.mean()),
                  "bias_hac_se": stats.newey_west_se_mean(e, H),
                  "oos_r2_oracle_testmean": float(1 - e @ e / sst_o),
                  "oos_r2_vs_train_mean": float(1 - e @ e / sst_f),
                  "per_fold_mse": {f"fold{b}": float(np.mean(((y - f) ** 2)[rows & (blocks == b)])) for b in range(3)
                                   if (rows & (blocks == b)).any()}}
    return out


def main():
    res = {}
    # ================= DEV WALK-FORWARD =================
    dates = pd.DatetimeIndex(c["dates_dev"])
    R, el = c["R_dev"], c["el_dev"]
    T = len(R)
    y, X0 = validate.har_design(R)
    keys = None
    stitched = {}
    rows_all = np.zeros(T, bool)
    blocks = np.full(T, -1)
    ybar = np.full(T, np.nan)
    fold_info = []
    for f, (tr_end, te_end) in enumerate(FOLDS):
        assert c[f"train_end_{f}"] == tr_end
        xv = {"M": c[f"M_{f}"], "BL": c[f"BL_{f}"]}
        Xchk = np.column_stack([X0, xv["M"], xv["BL"]])
        tr, te = rows_for(dates, el, y, Xchk, tr_end, te_end)
        pool = el & np.isfinite(y) & np.all(np.isfinite(Xchk), 1)
        fc, _, yb = forecasts(R, xv, tr, te, pool)
        if keys is None:
            keys = list(fc)
            stitched = {k: np.full(T, np.nan) for k in keys}
        for k in keys:
            stitched[k][te] = fc[k][te]
        rows_all |= te
        blocks[te] = f
        ybar[te] = yb
        fold_info.append({"train_end": tr_end, "n_train": int(tr.sum()), "n_test": int(te.sum()),
                          "first_train": str(dates[np.flatnonzero(tr)[0]].date()),
                          "y_mean_train": yb, "y_mean_test": float(y[te].mean())})
    res["dev_folds"] = fold_info
    res["dev_scores"] = score_set(y, stitched, rows_all, ybar, blocks)
    cmp = {}
    base = stitched["B0_HAR+M"]
    # primary: each candidate's HAR+X vs B0 HAR+M
    for cand in ("C1_HAR+M", "C2_HAR+M", "B0_HAR+BL"):
        cmp[f"primary: B0_HAR+M -> {cand}"] = compare(y, base, stitched[cand], rows_all, blocks)
    # same-design HAR comparisons (level mechanism without M)
    for cand in ("C1_HAR", "C2_HAR"):
        cmp[f"B0_HAR -> {cand}"] = compare(y, stitched["B0_HAR"], stitched[cand], rows_all, blocks)
    # dev OOS H4: increment of X over the same-design HAR
    for d0, d1 in (("B0_HAR", "B0_HAR+M"), ("B0_HAR", "B0_HAR+BL"), ("C1_HAR", "C1_HAR+M"), ("C2_HAR", "C2_HAR+M")):
        cmp[f"increment: {d0} -> {d1}"] = compare(y, stitched[d0], stitched[d1], rows_all, blocks)
    res["dev_comparisons"] = cmp
    np.savez_compressed(f"{OUT}/dev_stitched.npz", y=y, rows=rows_all, blocks=blocks, ybar=ybar,
                        **{k.replace("+", "_p_"): v for k, v in stitched.items()})

    # ================= 2026 POST-HOC (diagnostic only) =================
    dF = pd.DatetimeIndex(c["dates_full"])
    RF, elF = c["R_full"], c["el_full"]
    yF, X0F = validate.har_design(RF)
    xvF = {"M": c["M_frz"], "BL": c["BL_frz"]}
    XchkF = np.column_stack([X0F, xvF["M"], xvF["BL"]])
    trF, _ = rows_for(dF, elF, yF, XchkF, "2025-12-31")
    teF = elF & np.isfinite(yF) & np.all(np.isfinite(XchkF), 1) & (dF >= "2026-01-02")
    poolF = elF & np.isfinite(yF) & np.all(np.isfinite(XchkF), 1)
    fcF, _, ybF = forecasts(RF, xvF, trF, teF, poolF)
    blkF = np.where(teF, 0, -1)
    ybarF = np.full(len(RF), ybF)
    res["posthoc_2026_n_train"] = int(trF.sum())
    res["posthoc_2026_scores"] = score_set(yF, fcF, teF, ybarF, blkF)
    # check: B0 on 2026 equals the frozen H4 numbers
    cmpF = {}
    for cand in ("C1_HAR+M", "C2_HAR+M", "B0_HAR+BL"):
        cmpF[f"primary: B0_HAR+M -> {cand}"] = compare(yF, fcF["B0_HAR+M"], fcF[cand], teF, blkF)
    for cand in ("C1_HAR", "C2_HAR"):
        cmpF[f"B0_HAR -> {cand}"] = compare(yF, fcF["B0_HAR"], fcF[cand], teF, blkF)
    for d0, d1 in (("B0_HAR", "B0_HAR+M"), ("B0_HAR", "B0_HAR+BL"), ("C1_HAR", "C1_HAR+M"), ("C2_HAR", "C2_HAR+M")):
        cmpF[f"increment: {d0} -> {d1}"] = compare(yF, fcF[d0], fcF[d1], teF, blkF)
    for v in cmpF.values():
        v.pop("per_fold_gain"); v.pop("folds_positive")
    res["posthoc_2026_comparisons"] = cmpF
    json.dump(res, open(f"{OUT}/walkforward_results.json", "w"), indent=1, default=float)

    # compact print
    def p(sc):
        for k, v in sc.items():
            print(f"  {k:12s} mse {v['mse']:.4f} bias {v['bias']:+.3f} ({v['bias_hac_se']:.3f}) "
                  f"R2orc {v['oos_r2_oracle_testmean']:+.3f} R2feas {v['oos_r2_vs_train_mean']:+.3f} "
                  + " ".join(f"{a}:{b:.4f}" for a, b in v.get("per_fold_mse", {}).items()))

    def q(cm):
        for k, v in cm.items():
            print(f"  {k:36s} gain {v['mean_gain']:+.5f} se {v['hln_se']:.5f} t {v['dm_hln_t']:+.2f} "
                  f"boot95 [{v['boot95_and_se'][0]:+.5f},{v['boot95_and_se'][1]:+.5f}] "
                  f"bias-term {v['gain_from_bias_term']:+.5f} var-term {v['gain_from_variance_term']:+.5f} "
                  + (f"folds+ {v['folds_positive']} {json.dumps({a: round(b, 5) for a, b in v['per_fold_gain'].items()})}"
                     if "per_fold_gain" in v else ""))
    print("DEV folds", json.dumps(fold_info))
    print("DEV scores"); p(res["dev_scores"])
    print("DEV comparisons"); q(cmp)
    print("2026 POST-HOC scores (n_train %d)" % res["posthoc_2026_n_train"]); p(res["posthoc_2026_scores"])
    print("2026 POST-HOC comparisons"); q(cmpF)


if __name__ == "__main__":
    main()
