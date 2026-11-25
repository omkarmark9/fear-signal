"""The pre-registered tests, as functions of a finished signal.

Every function takes the signal and a boolean day mask, so the identical code
produces the in-sample dev reference numbers and the one-shot 2026 holdout
numbers. Nothing here fits anything to the period being scored: maps, HAR
coefficients, climatology and VXN thresholds arrive pre-computed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import stats
from .features import Features
from .model import Signal, TAIL_THRESHOLD, tail_event

HAR_LAGS = (1, 5, 22)


# ----------------------------------------------------------------------------- helpers

def basket_returns(feat: Features) -> np.ndarray:
    """Equal-weight daily return of the AI basket (stocks with a valid return)."""
    simple = np.expm1(feat.r)
    n = np.isfinite(simple).sum(axis=1)
    return np.where(n > 0, np.nansum(simple, axis=1) / np.maximum(n, 1), np.nan)


def _fwd_sum(x: np.ndarray, h: int) -> np.ndarray:
    """sum_{u=t+1}^{t+h} x_u (NaN when the window runs past the end)."""
    out = np.full(len(x), np.nan)
    c = np.concatenate([[0.0], np.cumsum(np.nan_to_num(x))])
    for t in range(len(x) - h):
        out[t] = c[t + 1 + h] - c[t + 1]
    return out


def _back_mean(x: np.ndarray, h: int) -> np.ndarray:
    s = pd.Series(x).rolling(h, min_periods=h).mean()
    return s.to_numpy()


def har_design(R: np.ndarray, extra: np.ndarray | None = None):
    """log 5-day forward realised vol on HAR lags of daily squared returns."""
    r2 = R**2
    y = 0.5 * np.log(_fwd_sum(r2, 5) / 5 + 1e-10)
    cols = [np.ones(len(R))] + [0.5 * np.log(_back_mean(r2, h) + 1e-10) for h in HAR_LAGS]
    if extra is not None:
        cols.append(extra)
    return y, np.column_stack(cols)


# ----------------------------------------------------------------------------- tests

def _d5(s: np.ndarray, v: np.ndarray, dates, mask) -> pd.DataFrame:
    """5-session changes on the full session calendar, then restricted to mask.

    Differencing before dropping missing values keeps every change exactly
    five sessions long; a change whose endpoints are not both observed is
    dropped rather than silently stretched.
    """
    df = pd.DataFrame({"s": s, "v": v}, index=dates)
    return df.diff(5)[mask].dropna()


def vxn_construct(sig: Signal, vxn: pd.Series, dates: pd.DatetimeIndex, mask: np.ndarray,
                  n_boot: int = 2000, v1: Signal | None = None) -> dict:
    """H1 / H1b: correlation of the market reading with VXN.

    ``v1`` is the published v1 model's signal; H1b pairs v2's M with v1's
    legacy M on the common days.
    """
    v = vxn.reindex(dates).to_numpy()
    out = {}
    for name, s in (("M", sig.M), ("M_legacy", sig.M_legacy), ("breadth", sig.breadth)):
        lv = pd.DataFrame({"s": s, "v": v}, index=dates)[mask].dropna()
        d5 = _d5(s, v, dates, mask)
        out[name] = {
            # levels are far more persistent than changes: longer blocks
            "level_spearman": stats.corr_ci(lv.s, lv.v, "spearman", block=60, n_boot=n_boot),
            "d5_pearson": stats.corr_ci(d5.s, d5.v, "pearson", n_boot=n_boot),
            "d5_spearman": stats.corr_ci(d5.s, d5.v, "spearman", n_boot=n_boot),
        }
    if v1 is not None:
        a = _d5(sig.M, v, dates, mask)
        b = _d5(v1.M_legacy, v, dates, mask)
        j = a.index.intersection(b.index)
        out["H1b_d5_spearman_v2M_minus_v1legacy"] = stats.corr_diff_ci(
            a.loc[j, "s"], b.loc[j, "s"], a.loc[j, "v"], n_boot=n_boot)
    return out


def thermometer(sig: Signal, feat: Features, mask: np.ndarray) -> dict:
    """How the stock score reads on days the stock is visibly hit, and vice versa."""
    el = feat.obs.eligible & mask[:, None]
    rz = feat.r / feat.sigma
    hit3 = el & (rz <= -3)
    Z = sig.Z
    x = np.where(feat.obs.eligible, feat.obs.x, 0.0)
    recent_hit = pd.DataFrame(x >= 2).rolling(5, min_periods=1).max().to_numpy() > 0
    panic = el & (Z >= 2.5)
    return {
        "n_3sigma_loss_days": int(hit3.sum()),
        "share_3sigma_loss_days_Z_ge_1.5": float((Z[hit3] >= 1.5).mean()),
        "share_3sigma_loss_days_Z_ge_2.0": float((Z[hit3] >= 2.0).mean()),
        "n_panic_readings": int(panic.sum()),
        "share_panic_readings_without_2sigma_dip_in_5d": float((~recent_hit[panic]).mean()) if panic.any() else None,
        "share_stock_days_by_band": {
            b: float(v) for b, v in zip(
                ("calm", "uneasy", "stressed", "acute fear", "panic"),
                np.histogram(Z[el], bins=[-np.inf, 1, 1.5, 2, 2.5, np.inf])[0] / el.sum())
        },
    }


def tail_forecast(sig: Signal, feat: Features, mask: np.ndarray, climatology: float,
                  baselines: dict[str, np.ndarray] | None = None) -> dict:
    """H2 / H2b: Brier skill and calibration of q_t = P(x_{t+1} >= 3 | x_<=t)."""
    y = tail_event(feat.obs, TAIL_THRESHOLD)
    q = np.vstack([np.full((1, y.shape[1]), np.nan), sig.q_next[:-1]])  # aligned to the scored day
    ok = np.isfinite(q) & np.isfinite(y) & mask[:, None]
    for b in (baselines or {}).values():
        ok &= np.isfinite(b)
    day = np.broadcast_to(np.arange(len(y))[:, None], y.shape)[ok]
    qv, yv = q[ok], y[ok]
    brier = float(np.mean((qv - yv) ** 2))
    b_clim = float(np.mean((climatology - yv) ** 2))
    out = {"n": int(ok.sum()), "event_rate": float(yv.mean()), "mean_forecast": float(qv.mean()),
           "brier": brier, "brier_climatology": b_clim, "bss_vs_climatology": 1 - brier / b_clim,
           "calibration": stats.logistic_calibration(qv, yv, groups=day),
           "reliability": stats.reliability_table(qv, yv)}
    for name, b in (baselines or {}).items():
        bb = float(np.mean((b[ok] - yv) ** 2))
        out[f"brier_{name}"] = bb
        out[f"bss_vs_{name}"] = 1 - brier / bb
    # Brier gain vs climatology: averaged within day, then HAC across days
    d = np.where(ok, (q - y) ** 2 - (climatology - y) ** 2, np.nan)
    has = ok.any(axis=1)
    daily = -np.nansum(d[has], axis=1) / ok[has].sum(axis=1)
    out["brier_gain_vs_clim_daily_mean"] = float(np.mean(daily))
    out["brier_gain_vs_clim_hac_se"] = stats.newey_west_se_mean(daily, 5)
    out["brier_gain_vs_clim_t"] = out["brier_gain_vs_clim_daily_mean"] / out["brier_gain_vs_clim_hac_se"]
    return out


def har_fit(sig: Signal, R: np.ndarray, mask: np.ndarray) -> dict:
    """Fit HAR and HAR+M for 5-day forward basket volatility (dev only)."""
    y, X0 = har_design(R)
    _, X1 = har_design(R, sig.M)
    ok = mask & np.isfinite(y) & np.all(np.isfinite(X1), axis=1)
    f0 = stats.ols_hac(y[ok], X0[ok], lags=5)
    f1 = stats.ols_hac(y[ok], X1[ok], lags=5)
    return {"beta_har": f0["beta"].tolist(), "beta_har_M": f1["beta"].tolist(),
            "se_har_M": f1["se"].tolist(), "t_M": float(f1["t"][-1]),
            "r2_har": f0["r2"], "r2_har_M": f1["r2"], "n": f1["n"]}


def har_oos(sig: Signal, R: np.ndarray, mask: np.ndarray, fitted: dict) -> dict:
    """H4: out-of-sample R^2 of frozen HAR and HAR+M; DM-HLN test of the gain."""
    y, X0 = har_design(R)
    _, X1 = har_design(R, sig.M)
    ok = mask & np.isfinite(y) & np.all(np.isfinite(X1), axis=1)
    e0 = y[ok] - X0[ok] @ np.asarray(fitted["beta_har"])
    e1 = y[ok] - X1[ok] @ np.asarray(fitted["beta_har_M"])
    sst = np.sum((y[ok] - y[ok].mean()) ** 2)
    dm = stats.diebold_mariano(e0**2 - e1**2, horizon=5)
    return {"n": int(ok.sum()), "oos_r2_har": float(1 - e0 @ e0 / sst),
            "oos_r2_har_M": float(1 - e1 @ e1 / sst),
            "mse_gain_mean": dm["mean"], "dm_hln_t": dm["t"], "dm_hln_p_one_sided": dm["p_one_sided"]}


def har_recursive(sig: Signal, R: np.ndarray, mask: np.ndarray, start: int | None = None) -> dict:
    """H4' (v3 convention): HAR vs HAR+M re-estimated daily on an expanding window.

    The forecast for day t uses coefficients fitted on rows s <= t - 5 only:
    the target at s needs returns through s + 5, so this is the latest row
    whose target is known at the close of t. Nested models under a recursive
    scheme make a Diebold-Mariano test undersized, so the primary statistic is
    Clark-West (2007); DM-HLN on the raw MSE gain is reported beside it.
    """
    y, X0 = har_design(R)
    _, X1 = har_design(R, sig.M)
    good = np.isfinite(y) & np.all(np.isfinite(X1), axis=1)
    first = int(np.argmax(good)) if start is None else start
    f0 = np.full(len(y), np.nan)
    f1 = np.full(len(y), np.nan)
    for t in np.flatnonzero(mask & good):
        rows = np.flatnonzero(good[: max(t - 4, 0)])   # s <= t - 5
        rows = rows[rows >= first]
        if len(rows) < 60:
            continue
        b0 = np.linalg.lstsq(X0[rows], y[rows], rcond=None)[0]
        b1 = np.linalg.lstsq(X1[rows], y[rows], rcond=None)[0]
        f0[t], f1[t] = X0[t] @ b0, X1[t] @ b1
    ok = np.isfinite(f0) & np.isfinite(f1)
    e0, e1 = y[ok] - f0[ok], y[ok] - f1[ok]
    # feasible benchmark: the expanding-window mean of y known at t (rows s <= t - 5)
    fm = np.array([np.nanmean(y[first: t - 4][good[first: t - 4]]) for t in np.flatnonzero(ok)])
    sst_feasible = np.sum((y[ok] - fm) ** 2)
    sst_oracle = np.sum((y[ok] - y[ok].mean()) ** 2)
    cw = stats.clark_west(y[ok], f0[ok], f1[ok], horizon=5)
    dm = stats.diebold_mariano(e0**2 - e1**2, horizon=5)
    return {"n": int(ok.sum()),
            "oos_r2_har_feasible": float(1 - e0 @ e0 / sst_feasible),
            "oos_r2_har_M_feasible": float(1 - e1 @ e1 / sst_feasible),
            "oos_r2_har_oracle": float(1 - e0 @ e0 / sst_oracle),
            "oos_r2_har_M_oracle": float(1 - e1 @ e1 / sst_oracle),
            "clark_west_t": cw["t"], "clark_west_p_one_sided": cw["p_one_sided"],
            "dm_hln_t": dm["t"], "dm_hln_p_one_sided": dm["p_one_sided"]}


def forward_returns(sig: Signal, R: np.ndarray, mask: np.ndarray, horizon: int = 20) -> dict:
    """H5: does market fear predict the basket's forward return? (secondary)

    The slope of the overlapping h-day regression is reported for its size;
    inference uses Hodrick's (1992) 1B reverse regression.
    """
    lr = np.log1p(R)
    fwd = _fwd_sum(lr, horizon)
    ok = mask & np.isfinite(fwd) & np.isfinite(sig.M)
    res = stats.ols_hac(fwd[ok], np.column_stack([np.ones(ok.sum()), sig.M[ok]]), lags=2 * horizon)
    # reverse regression restricted to the scored days (regressor history may precede them)
    r1 = np.where(mask, lr, np.nan)
    hb = stats.hodrick_1b(r1, sig.M, horizon)
    hi = ok & (sig.M_pct >= 2.0)          # the alert event (M_pct); equals M >= 2 for v2
    return {"horizon": horizon, "n": res["n"], "slope_per_M": float(res["beta"][1]),
            "hodrick_1b_t": hb["t"], "hodrick_1b_p": hb["p"],
            "mean_fwd_all": float(fwd[ok].mean()),
            "n_days_M_ge_2": int(hi.sum()),
            "mean_fwd_M_ge_2": float(fwd[hi].mean()) if hi.any() else None}


def alerts(sig: Signal, M_lo: np.ndarray | None, vxn: pd.Series, dates: pd.DatetimeIndex,
           mask: np.ndarray, vxn_elevated: float, threshold: float = 2.0) -> dict:
    """Alert days vs VXN: precision of M_pct >= threshold and of the conviction
    rule (``M_lo`` = lower band of the percentile score M_pct). For v2,
    M_pct is M."""
    v = vxn.reindex(dates).to_numpy()
    ok = mask & np.isfinite(v) & np.isfinite(sig.M_pct)
    elev = v >= vxn_elevated

    def summarise(flag):
        f = ok & flag
        return {"n_days": int(f.sum()),
                "precision_vxn_elevated": float(elev[f].mean()) if f.any() else None,
                "mean_vxn": float(v[f].mean()) if f.any() else None}

    out = {"vxn_elevated_threshold": vxn_elevated,
           "base_rate_vxn_elevated": float(elev[ok].mean()),
           "mean_vxn_all": float(v[ok].mean()),
           f"M_ge_{threshold}": summarise(sig.M_pct >= threshold)}
    if M_lo is not None:
        out[f"conviction_M_lo_ge_{threshold}"] = summarise(np.nan_to_num(M_lo, nan=-9) >= threshold)
    return out


def tail_calibration_detail(sig: Signal, feat: Features, mask: np.ndarray) -> dict:
    """P-H2 companions: leave-one-day-out range of the calibration slope, and
    its between-day and within-day parts (day-clustered s.e.)."""
    y = tail_event(feat.obs, TAIL_THRESHOLD)
    q = np.vstack([np.full((1, y.shape[1]), np.nan), sig.q_next[:-1]])
    ok = np.isfinite(q) & np.isfinite(y) & mask[:, None]
    day = np.broadcast_to(np.arange(len(y))[:, None], y.shape)[ok]
    qv, yv = q[ok], y[ok]
    slopes = np.array([stats.logistic_calibration(qv[day != d], yv[day != d])["slope"]
                       for d in np.unique(day)])
    return {"loo_slope_min": float(slopes.min()), "loo_slope_max": float(slopes.max()),
            "loo_days_below_0.8": int((slopes < 0.8).sum()), "n_days": int(len(slopes)),
            "within_between": stats.within_between_calibration(qv, yv, day)}
