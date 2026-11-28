"""Live sizing, as declared (research/trading/declaration.json, "sizing_and_risk").

This is never a selection input. It turns an adopted family's ticket into a
risk budget and a notional:

    sigma_tilde_i(h) = kappa_h * sigma_i * sqrt(h) * exp(l~_i)
    l_i  = 0.5 ln(U_i / h),  U_i = sum_{j<=h} (p_i A^j) . s^2  (risk.forward_var_units)
    l~_i = 0.5 l_i + 0.5 mean_j(l_j)          (halfway to the day's cross-section)
    kappa_h = sqrt(sum RV / sum (sigma_tilde / kappa_h)^2) over dev stock-days

The yardstick is used only if its QLIKE loss is within 1 s.e. of EWMA(0.94)'s
at h = 5 and h = 20. Otherwise EWMA, level-calibrated the same way, is used.

    IR_post = IRhat * 0.05^2 / (0.05^2 + 1/N_eff)          (prior IR ~ N(0, 0.05^2))
    RB      = min(TierCap, 0.25 * IR_post) / (1 + (n - 1) rho)
    notional = RB * NAV / sigma_tilde_i(H)
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .. import spa
from ..data import RAW
from . import risk

TIER_CAP = {0: 0.0, 1: 0.0025, 2: 0.005, 3: 0.01}
H_GRID = (1, 5, 10, 20, 40)
PRIOR_SD = 0.05
KELLY = 0.25


def refit_of(dates: pd.DatetimeIndex, refits: list[dict]) -> list[int]:
    """Index of the refit in force on each date (the latest refit date <= d)."""
    starts = pd.to_datetime([r["refit"] for r in refits])
    return [int(np.searchsorted(starts, d, side="right") - 1) for d in dates]


def log_multiplier(P: np.ndarray, A: np.ndarray, s: np.ndarray, h: int) -> np.ndarray:
    return 0.5 * np.log(risk.forward_var_units(P, A, s, h) / h)


def unscaled_sigma_tilde(sigma: pd.DataFrame, Pc: pd.DataFrame, Ps: pd.DataFrame, Pp: pd.DataFrame,
                         refits: list[dict], h: int, A_live=None, s_live=None) -> pd.DataFrame:
    """sigma * sqrt(h) * exp(l~) for every stock-day (kappa not applied)."""
    out = pd.DataFrame(np.nan, index=sigma.index, columns=sigma.columns)
    ri = refit_of(sigma.index, refits) if refits else [None] * len(sigma)
    for t, d in enumerate(sigma.index):
        sg = sigma.iloc[t].to_numpy(float)
        P = np.column_stack([Pc.iloc[t], Ps.iloc[t], Pp.iloc[t]]).astype(float)
        ok = np.isfinite(sg) & np.all(np.isfinite(P), axis=1)
        if not ok.any():
            continue
        if A_live is not None:
            A, s = np.asarray(A_live, float), np.asarray(s_live, float)
        else:
            if ri[t] is None or ri[t] < 0:
                continue
            A, s = np.asarray(refits[ri[t]]["A"], float), np.asarray(refits[ri[t]]["s"], float)
        l = log_multiplier(P[ok], A, s, h)
        lt = 0.5 * l + 0.5 * l.mean()
        row = np.full(len(sg), np.nan)
        row[ok] = sg[ok] * np.sqrt(h) * np.exp(lt)
        out.iloc[t] = row
    return out


def realized_var(r: pd.DataFrame, h: int) -> pd.DataFrame:
    """RV_i,t(h) = sum_{j=1..h} r_{i,t+j}^2 (NaN unless all h returns exist)."""
    r2 = r ** 2
    return r2[::-1].rolling(h, min_periods=h).sum()[::-1].shift(-1)


def ewma_sigma(r: pd.DataFrame, lam: float = 0.94) -> pd.DataFrame:
    """RiskMetrics daily sigma known at the close of t (includes r_t)."""
    return np.sqrt((r ** 2).ewm(alpha=1 - lam, adjust=False, min_periods=20).mean())


def qlike(f2: np.ndarray, rv: np.ndarray) -> np.ndarray:
    x = rv / f2
    return x - np.log(x) - 1.0


def calibrate(rf, refits: list[dict], start, end) -> dict:
    """kappa_h for the regime yardstick and for EWMA, and the QLIKE validation, on dev only.

    Stock-days are dev dates whose realised window ends by ``end``.
    """
    r = rf.r
    el = rf.eligible
    out = {"kappa": {}, "kappa_ewma": {}, "qlike": {}}
    ew = ewma_sigma(r.where(el))
    dev = (r.index >= pd.Timestamp(start)) & (r.index <= pd.Timestamp(end))
    for h in H_GRID:
        rv = realized_var(r, h)
        ends_ok = pd.Series(r.index, index=r.index).shift(-h) <= pd.Timestamp(end)
        rows = dev & ends_ok.fillna(False).to_numpy()
        st = unscaled_sigma_tilde(rf.sigma, rf.P_calm, rf.P_stressed, rf.P_panic, refits, h)
        f_reg = st[rows].where(el[rows])
        f_ew = (ew * np.sqrt(h))[rows].where(el[rows])
        y = rv[rows]
        m = f_reg.notna() & f_ew.notna() & y.notna() & (y > 0)
        k = float(np.sqrt(y[m].sum().sum() / (f_reg[m] ** 2).sum().sum()))
        ke = float(np.sqrt(y[m].sum().sum() / (f_ew[m] ** 2).sum().sum()))
        out["kappa"][h], out["kappa_ewma"][h] = k, ke
        if h in (5, 20):
            lr = qlike((k * f_reg) ** 2, y).where(m)
            le = qlike((ke * f_ew) ** 2, y).where(m)
            d = (lr - le).mean(axis=1).dropna()              # daily mean loss differential
            se = float(np.sqrt(spa.lrv(d.to_numpy(), "sb", 20.0)[0] / len(d)))
            out["qlike"][h] = {"regime": float(lr.stack().mean()), "ewma": float(le.stack().mean()),
                               "diff_mean": float(d.mean()), "diff_se": se,
                               "regime_worse_by_more_than_1se": bool(d.mean() > se)}
    out["use"] = "ewma" if any(v["regime_worse_by_more_than_1se"] for v in out["qlike"].values()) else "regime"
    return out


def equicorrelation(r: pd.DataFrame, el: pd.DataFrame, t: int) -> float:
    """max(21-session, 63-session) mean off-diagonal correlation, ending at row t."""
    vals = []
    for w in (21, 63):
        x = r.iloc[max(0, t - w + 1): t + 1].where(el.iloc[max(0, t - w + 1): t + 1])
        c = x.corr(min_periods=max(10, w // 2)).to_numpy()
        iu = np.triu_indices_from(c, 1)
        vals.append(float(np.nanmean(c[iu])))
    return max(vals)


def ir_post(event_excess: np.ndarray, entry_dates: pd.Series) -> dict:
    e = np.asarray(event_excess, float)
    irhat = float(e.mean() / e.std(ddof=1)) if len(e) > 1 and e.std(ddof=1) > 0 else 0.0
    weeks = pd.to_datetime(entry_dates).dt.isocalendar()
    n_eff = int(len(set(zip(weeks["year"], weeks["week"]))))
    post = irhat * PRIOR_SD ** 2 / (PRIOR_SD ** 2 + 1.0 / max(n_eff, 1))
    return {"IRhat": irhat, "N_eff": n_eff, "IR_post": float(post)}


def risk_budget(tier: int, irp: float, n_recent: int, rho: float) -> float:
    if tier <= 0 or irp <= 0:
        return 0.0
    rb = min(TIER_CAP[tier], KELLY * irp)
    return rb / (1 + (max(n_recent, 1) - 1) * rho)


def load_refits() -> list[dict]:
    return json.loads((RAW / "pit_refits.json").read_text())


def qlike_oos(rf, refits: list[dict], kappa: dict, kappa_ewma: dict, start, end, hs=(5, 20),
              A_live=None, s_live=None) -> dict:
    """Out-of-sample QLIKE of the regime yardstick against EWMA, using level factors
    calibrated elsewhere (dev). Nothing is fitted on [start, end]. With ``A_live`` and
    ``s_live`` (the frozen live model) the refits are ignored."""
    r, el = rf.r, rf.eligible
    ew = ewma_sigma(r.where(el))
    win = (r.index >= pd.Timestamp(start)) & (r.index <= pd.Timestamp(end))
    out = {}
    for h in hs:
        rv = realized_var(r, h)
        ends_ok = (pd.Series(r.index, index=r.index).shift(-h) <= pd.Timestamp(end)).fillna(False).to_numpy()
        rows = win & ends_ok
        st = unscaled_sigma_tilde(rf.sigma, rf.P_calm, rf.P_stressed, rf.P_panic, refits, h, A_live, s_live)
        k, ke = kappa[str(h)] if str(h) in kappa else kappa[h], kappa_ewma[str(h)] if str(h) in kappa_ewma else kappa_ewma[h]
        f_reg = (k * st)[rows].where(el[rows])
        f_ew = (ke * ew * np.sqrt(h))[rows].where(el[rows])
        y = rv[rows]
        m = f_reg.notna() & f_ew.notna() & y.notna() & (y > 0)
        lr = qlike(f_reg ** 2, y).where(m)
        le = qlike(f_ew ** 2, y).where(m)
        d = (lr - le).mean(axis=1).dropna()
        se = float(np.sqrt(spa.lrv(d.to_numpy(), "sb", 20.0)[0] / len(d)))
        out[h] = {"regime": float(lr.stack().mean()), "ewma": float(le.stack().mean()),
                  "diff_mean": float(d.mean()), "diff_se": se, "t": float(d.mean() / se), "n_days": int(len(d))}
    return out
