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

def vxn_construct(sig: Signal, vxn: pd.Series, dates: pd.DatetimeIndex, mask: np.ndarray,
                  n_boot: int = 2000) -> dict:
    v = vxn.reindex(dates).to_numpy()
    out = {}
    for name, s in (("M", sig.M), ("M_legacy", sig.M_legacy), ("breadth", sig.breadth)):
        df = pd.DataFrame({"s": s, "v": v}, index=dates)[mask].dropna()
        d5 = df.diff(5).dropna()
        out[name] = {
            "level_spearman": stats.corr_ci(df.s, df.v, "spearman", n_boot=n_boot),
            "d5_pearson": stats.corr_ci(d5.s, d5.v, "pearson", n_boot=n_boot),
            "d5_spearman": stats.corr_ci(d5.s, d5.v, "spearman", n_boot=n_boot),
        }
    df = pd.DataFrame({"a": sig.M, "b": sig.M_legacy, "v": v}, index=dates)[mask].dropna()
    d5 = df.diff(5).dropna()
    out["d5_spearman_M_minus_legacy"] = stats.corr_diff_ci(d5.a, d5.b, d5.v, n_boot=n_boot)
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
                  q_baseline: np.ndarray | None = None) -> dict:
    """Brier skill and calibration of q_t = P(x_{t+1} >= 3 | x_<=t)."""
    y = tail_event(feat.obs, TAIL_THRESHOLD)
    q = np.vstack([np.full((1, y.shape[1]), np.nan), sig.q_next[:-1]])  # aligned to the scored day
    ok = np.isfinite(q) & np.isfinite(y) & mask[:, None]
    if q_baseline is not None:
        ok &= np.isfinite(q_baseline)
    qv, yv = q[ok], y[ok]
    brier = float(np.mean((qv - yv) ** 2))
    b_clim = float(np.mean((climatology - yv) ** 2))
    out = {"n": int(ok.sum()), "event_rate": float(yv.mean()), "mean_forecast": float(qv.mean()),
           "brier": brier, "brier_climatology": b_clim, "bss_vs_climatology": 1 - brier / b_clim,
           "calibration": stats.logistic_calibration(qv, yv),
           "reliability": stats.reliability_table(qv, yv)}
    if q_baseline is not None:
        b_base = float(np.mean((q_baseline[ok] - yv) ** 2))
        out["brier_ewma"] = b_base
        out["bss_vs_ewma"] = 1 - brier / b_base
    # day-clustered (then HAC across days) standard error of the Brier gain
    d = np.where(ok, (q - y) ** 2 - (climatology - y) ** 2, np.nan)
    has = ok.any(axis=1)
    daily = np.nansum(d[has], axis=1) / ok[has].sum(axis=1)
    out["brier_minus_clim_daily_mean"] = float(np.mean(daily))
    out["brier_minus_clim_hac_se"] = stats.newey_west_se_mean(daily, 5)
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
    """Out-of-sample R^2 of frozen HAR and HAR+M coefficients."""
    y, X0 = har_design(R)
    _, X1 = har_design(R, sig.M)
    ok = mask & np.isfinite(y) & np.all(np.isfinite(X1), axis=1)
    e0 = y[ok] - X0[ok] @ np.asarray(fitted["beta_har"])
    e1 = y[ok] - X1[ok] @ np.asarray(fitted["beta_har_M"])
    sst = np.sum((y[ok] - y[ok].mean()) ** 2)
    d = e0**2 - e1**2
    return {"n": int(ok.sum()), "oos_r2_har": float(1 - e0 @ e0 / sst),
            "oos_r2_har_M": float(1 - e1 @ e1 / sst),
            "mse_gain_mean": float(d.mean()), "mse_gain_hac_se": stats.newey_west_se_mean(d, 5)}


def forward_returns(sig: Signal, R: np.ndarray, mask: np.ndarray, horizon: int = 20) -> dict:
    """Does market fear predict the basket's forward return? (secondary)"""
    fwd = _fwd_sum(np.log1p(R), horizon)
    ok = mask & np.isfinite(fwd) & np.isfinite(sig.M)
    res = stats.ols_hac(fwd[ok], np.column_stack([np.ones(ok.sum()), sig.M[ok]]), lags=horizon)
    hi = ok & (sig.M >= 2.0)
    return {"horizon": horizon, "n": res["n"], "slope_per_M": float(res["beta"][1]),
            "slope_t_hac": float(res["t"][1]), "slope_p_hac": float(res["p"][1]),
            "mean_fwd_all": float(fwd[ok].mean()),
            "n_days_M_ge_2": int(hi.sum()),
            "mean_fwd_M_ge_2": float(fwd[hi].mean()) if hi.any() else None}


def alerts(sig: Signal, M_lo: np.ndarray | None, vxn: pd.Series, dates: pd.DatetimeIndex,
           mask: np.ndarray, vxn_elevated: float, threshold: float = 2.0) -> dict:
    """Alert days vs VXN: precision of M >= threshold and of the conviction rule."""
    v = vxn.reindex(dates).to_numpy()
    ok = mask & np.isfinite(v) & np.isfinite(sig.M)
    elev = v >= vxn_elevated

    def summarise(flag):
        f = ok & flag
        return {"n_days": int(f.sum()),
                "precision_vxn_elevated": float(elev[f].mean()) if f.any() else None,
                "mean_vxn": float(v[f].mean()) if f.any() else None}

    out = {"vxn_elevated_threshold": vxn_elevated,
           "base_rate_vxn_elevated": float(elev[ok].mean()),
           "mean_vxn_all": float(v[ok].mean()),
           f"M_ge_{threshold}": summarise(sig.M >= threshold)}
    if M_lo is not None:
        out[f"conviction_M_lo_ge_{threshold}"] = summarise(np.nan_to_num(M_lo, nan=-9) >= threshold)
    return out
