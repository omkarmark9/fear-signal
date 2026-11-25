"""Diagnosis of the H4 failure on 2026 (diagnostic only; nothing here selects anything).

Questions: (a) level/intercept shift? (b) frozen coefficients? (c) M's percentile scale?
"""
import json
import numpy as np
import pandas as pd
from fearsignal import stats, validate

OUT = "/tmp/claude-0/v3/volatility-increment"
c = np.load(f"{OUT}/cache.npz")
spec = json.load(open("/home/user/fear-signal/config/frozen.json"))
har = spec["har"]
dates = pd.DatetimeIndex(c["dates_full"])
R = c["R_full"]
M, BL, MS = c["M_frz"], c["BL_frz"], c["MS_frz"]
el = c["el_full"]
hold = el & (dates >= "2026-01-02")
dev = el & (dates <= "2025-12-31")

y, X0 = validate.har_design(R)
_, X1 = validate.har_design(R, M)
okh = hold & np.isfinite(y) & np.all(np.isfinite(X1), 1)
okd = dev & np.isfinite(y) & np.all(np.isfinite(X1), 1)
# NB: the dev rows here use the full panel, where the last 5 dev days have y that uses 2026 returns.
# The frozen fit used the dev panel only (y NaN there). Reproduce that exactly:
okd &= ~((dates > "2025-12-31") | (np.arange(len(dates)) > np.flatnonzero(dates <= "2025-12-31")[-1] - 5))
b0, b1 = np.array(har["beta_har"]), np.array(har["beta_har_M"])
print("n hold", okh.sum(), "n dev", okd.sum())

out = {}
f0, f1 = X0 @ b0, X1 @ b1
e0, e1 = y - f0, y - f1


def summ(e, ok, ybar_bench):
    ee = e[ok]
    yy = y[ok]
    sst_oracle = np.sum((yy - yy.mean()) ** 2)
    sst_dev = np.sum((yy - ybar_bench) ** 2)
    return {"mse": float(np.mean(ee**2)), "bias": float(ee.mean()), "bias_sq": float(ee.mean() ** 2),
            "var_e": float(ee.var()), "bias_hac_se": stats.newey_west_se_mean(ee, 5),
            "oos_r2_vs_holdout_mean(oracle, as published)": float(1 - ee @ ee / sst_oracle),
            "oos_r2_vs_dev_mean(feasible)": float(1 - ee @ ee / sst_dev)}


ybar_dev = y[okd].mean()
out["level"] = {
    "y_mean_dev": float(ybar_dev), "y_mean_2026": float(y[okh].mean()),
    "y_mean_diff_2026_minus_dev": float(y[okh].mean() - ybar_dev),
    "y_mean_2026_hac_se": stats.newey_west_se_mean(y[okh], 10),
    "y_sd_dev": float(y[okd].std()), "y_sd_2026": float(y[okh].std()),
    "ann_vol_dev_pct": float(np.exp(ybar_dev) * np.sqrt(252) * 100),
    "ann_vol_2026_pct": float(np.exp(y[okh].mean()) * np.sqrt(252) * 100),
    "har_regressor_means_dev": X0[okd, 1:].mean(0).tolist(),
    "har_regressor_means_2026": X0[okh, 1:].mean(0).tolist(),
    "M_mean_dev": float(M[okd].mean()), "M_mean_2026": float(M[okh].mean()),
    "forecast_mean_HAR_2026": float(f0[okh].mean()), "forecast_mean_HARM_2026": float(f1[okh].mean()),
    "slope_sum_HAR": float(b0[1:].sum()), "slope_sum_HARM_excl_M": float(b1[1:4].sum()),
    "implied_unconditional_mean_HAR": float(b0[0] / (1 - b0[1:].sum())),
}
out["HAR_frozen_2026"] = summ(e0, okh, ybar_dev)
out["HARM_frozen_2026"] = summ(e1, okh, ybar_dev)
# share of the HAR+M MSE gain explained by the bias term
g_bias = out["HAR_frozen_2026"]["bias_sq"] - out["HARM_frozen_2026"]["bias_sq"]
g_var = out["HAR_frozen_2026"]["var_e"] - out["HARM_frozen_2026"]["var_e"]
out["gain_decomposition_2026"] = {"mse_gain": g_bias + g_var, "from_bias": g_bias, "from_variance": g_var}
dm_demeaned = stats.diebold_mariano((e0[okh] - e0[okh].mean()) ** 2 - (e1[okh] - e1[okh].mean()) ** 2, 5)
out["gain_decomposition_2026"]["dm_hln_on_demeaned_errors"] = dm_demeaned

# Mincer-Zarnowitz: y = a + b * forecast, HAC(5)
for name, f in (("HAR", f0), ("HARM", f1)):
    r = stats.ols_hac(y[okh], np.column_stack([np.ones(okh.sum()), f[okh]]), lags=5)
    out[f"MZ_2026_{name}"] = {"a": float(r["beta"][0]), "b": float(r["beta"][1]),
                              "se_b": float(r["se"][1]), "t_b_minus_1": float((r["beta"][1] - 1) / r["se"][1])}

# oracle decompositions (2026 info; diagnosis only)
for name, X, b, e in (("HAR", X0, b0, e0), ("HARM", X1, b1, e1)):
    mse_frozen = np.mean(e[okh] ** 2)
    mse_icpt = np.mean((e[okh] - e[okh].mean()) ** 2)      # frozen slopes, 2026-mean intercept
    r = stats.ols_hac(y[okh], X[okh], lags=5)                # full 2026 refit (in-sample on 2026)
    mse_full = np.mean((y[okh] - X[okh] @ r["beta"]) ** 2)
    out[f"oracle_{name}_2026"] = {"mse_frozen": float(mse_frozen), "mse_oracle_intercept": float(mse_icpt),
                                  "mse_oracle_full_refit": float(mse_full),
                                  "share_of_excess_mse_from_intercept": float((mse_frozen - mse_icpt) / (mse_frozen - mse_full)),
                                  "beta_2026_refit": r["beta"].tolist(), "se_2026_refit": r["se"].tolist()}
# coefficient stability dev in-sample vs 2026 refit, plus Chow-type stacked test with HAC
for name, X in (("HAR", X0), ("HARM", X1)):
    ok = okd | okh
    D = (dates >= "2026-01-02").astype(float)
    Xs = np.column_stack([X[ok], X[ok] * D[ok, None]])
    r = stats.ols_hac(y[ok], Xs, lags=5)
    k = X.shape[1]
    db = r["beta"][k:]
    # Wald on interaction slopes (excl intercept shift) and on intercept only
    # use HAC covariance reconstructed from se is not enough; refit with explicit cov
    yy, XX = y[ok], Xs
    n = len(yy)
    XtXi = np.linalg.inv(XX.T @ XX)
    bb = XtXi @ XX.T @ yy
    u = yy - XX @ bb
    g = XX * u[:, None]
    S = g.T @ g / n
    for l in range(1, 6):
        w = 1 - l / 6
        G = g[l:].T @ g[:-l] / n
        S += w * (G + G.T)
    V = n * XtXi @ S @ XtXi
    sl = list(range(k + 1, 2 * k))
    W_sl = float(db[1:] @ np.linalg.solve(V[np.ix_(sl, sl)], db[1:]))
    out[f"break_test_{name}"] = {"delta_beta_2026": db.tolist(), "se": r["se"][k:].tolist(),
                                 "wald_slopes": W_sl, "df_slopes": k - 1,
                                 "intercept_shift_t": float(db[0] / r["se"][k])}

# M scale: distribution dev vs 2026, and the frozen map's floor
mm = np.array(spec["m_map"]["knots_z"]) if "knots_z" in spec["m_map"] else None
q = [0.05, 0.25, 0.5, 0.75, 0.95]
out["M_scale"] = {
    "M_quantiles_dev": np.quantile(M[okd], q).tolist(), "M_quantiles_2026": np.quantile(M[okh], q).tolist(),
    "BL_quantiles_dev": np.quantile(BL[okd], q).tolist(), "BL_quantiles_2026": np.quantile(BL[okh], q).tolist(),
    "share_2026_BL_above_dev_max": float((BL[okh] > BL[okd].max()).mean()),
    "share_2026_BL_below_dev_min": float((BL[okh] < BL[okd].min()).mean()),
    "M_floor": float(np.nanmin(M[el])), "share_dev_at_floor": float((M[okd] <= np.nanmin(M[el]) + 1e-9).mean()),
    "share_2026_at_floor": float((M[okh] <= np.nanmin(M[el]) + 1e-9).mean()),
    "corr_M_BL_dev_pearson": float(np.corrcoef(M[okd], BL[okd])[0, 1]),
}
# Linearity: binned HAR residual (dev in-sample, frozen HAR) against M and against BL
r0 = y - f0
for var, v in (("M", M), ("BL", BL), ("MS", MS)):
    for per, ok in (("dev", okd), ("2026", okh)):
        qs = np.quantile(v[okd], [0, .2, .4, .6, .8, 1.0])
        qs[0], qs[-1] = -np.inf, np.inf
        bins = np.digitize(v[ok], qs[1:-1])
        out.setdefault("binned_HAR_residual", {})[f"{var}_{per}"] = [
            {"bin": int(b), "n": int((bins == b).sum()), "mean_v": float(v[ok][bins == b].mean()),
             "mean_resid": float(r0[ok][bins == b].mean())} for b in range(5) if (bins == b).any()]
# Partial correlation of HAR residual with each candidate regressor, dev in-sample vs 2026
for var, v in (("M", M), ("BL", BL), ("MS", MS)):
    for per, ok in (("dev", okd), ("2026", okh)):
        # residualise v on the HAR design within the period
        bv = np.linalg.lstsq(X0[ok], v[ok], rcond=None)[0]
        by = np.linalg.lstsq(X0[ok], y[ok], rcond=None)[0]
        rv, ry = v[ok] - X0[ok] @ bv, y[ok] - X0[ok] @ by
        out.setdefault("partial_corr_given_HAR_within_period", {})[f"{var}_{per}"] = float(np.corrcoef(rv, ry)[0, 1])

# monthly bias profile, 2026
df = pd.DataFrame({"e0": e0, "e1": e1, "y": y, "f0": f0, "M": M}, index=dates)[okh]
out["monthly_2026"] = df.groupby(df.index.month).mean().round(3).reset_index().to_dict("records")
json.dump(out, open(f"{OUT}/diag2026.json", "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
