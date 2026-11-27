"""The declared candidate families, implemented exactly as declared.

The source is research/trading/declaration.json (the judge's output,
verbatim), summarised in research/trading/candidates.md. Every number below is
declared there. Nothing here was set after looking at any result.

SR  systemic-shock liquidity-provision reversal (long).            Holm slot 1 of 2
ND  idiosyncratic news-shock drift (bearish information).          Holm slot 2 of 2
RF  fear-gated de-risking overlay (risk forecaster; no Holm slot,
    adopted on variance, tails and non-inferiority).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from . import engine, gates, registry, signals
from .registry import Family, Variant

DEV = gates.DEV


def _days(day: pd.Series, like: pd.DataFrame) -> pd.DataFrame:
    """Broadcast a per-date boolean to dates x tickers."""
    d = day.reindex(like.index).fillna(False).to_numpy(bool)
    return pd.DataFrame(np.repeat(d[:, None], like.shape[1], axis=1), index=like.index, columns=like.columns)


def _names(rf: signals.RuleFeatures) -> pd.DataFrame:
    """Condition (2), shared: eligible, raw close >= $5, 20-session median dollar volume >= $20M."""
    return rf.eligible & rf.price_ok & rf.liquid20


# ----------------------------------------------------------------------------- SR

def sr_select(rf: signals.RuleFeatures, v: Variant) -> pd.DataFrame:
    """SR conditions (1)-(5); (6) no re-entry and (7) the cap are applied by registry.entries.

    params: min_mpct (SR3), beta_only (SR4), day = "systemic" (default) or
    "not_systemic" (the S2 shock-type contrast; the half-day exclusion and any
    M_pct condition are kept, so only the shock type changes).
    """
    p = v.params
    day = rf.systemic if p.get("day", "systemic") == "systemic" else ~rf.systemic
    day = day & ~rf.half_day
    if p.get("min_mpct") is not None:
        day = day & (rf.M_pct >= p["min_mpct"])
    resid = rf.u if p.get("beta_only") else rf.v
    news = rf.g_all if p.get("beta_only") else rf.g_seg
    news_ok = (news > -2) if p.get("news", "clean") == "clean" else (news <= -2)
    return _names(rf) & (rf.x >= 2) & (resid <= -1.5) & news_ok & _days(day, rf.x)


def _sr(name: str, H: int, priority: str = "v", **params) -> Variant:
    return Variant("SR", name, H=H, select=sr_select, side="long", beta_benchmark=True,
                   series="ct_excess_net", cap=10, priority=priority, params=params)


SR = Family("SR", "Systemic-shock liquidity-provision reversal (long)", [
    _sr("SR1", 5),
    _sr("SR2", 10),
    _sr("SR3", 5, min_mpct=1.0),
    _sr("SR4", 5, priority="u", beta_only=True),
], holm=True)


# ----------------------------------------------------------------------------- ND

def nd_select(rf: signals.RuleFeatures, v: Variant) -> pd.DataFrame:
    """ND conditions (1)-(5); (6)-(7) via registry.entries.

    params: day = "not_systemic" (default) or "systemic" (the reported typed
    contrast); news = "gap" (default: g_seg <= -2) or "no_gap" (the N3
    information-split contrast: g_seg > -1).
    """
    p = v.params
    day = ~rf.systemic if p.get("day", "not_systemic") == "not_systemic" else rf.systemic
    day = day & ~rf.half_day
    news = (rf.g_seg <= -2) if p.get("news", "gap") == "gap" else (rf.g_seg > -1)
    return _names(rf) & (rf.x >= 3) & (rf.v <= -2) & news & _days(day, rf.x)


def _nd(name: str, H: int, **params) -> Variant:
    return Variant("ND", name, H=H, select=nd_select, side="short_info", beta_benchmark=True,
                   series="ct_excess_net", cap=5, priority="v", params=params)


ND = Family("ND", "Idiosyncratic news-shock drift (bearish information)", [
    _nd("ND1", 20),
    _nd("ND2", 40),
    _nd("ND3", 10),
], holm=True)

HOLM_FAMILIES = [SR, ND]


# --------------------------------------------------------------- event helpers

def event_frame(res: engine.Result, rf: signals.RuleFeatures) -> pd.DataFrame:
    """One row per event, with the signal-day inputs the signatures need."""
    tr = res.trades.copy()
    if tr.empty:
        return tr
    d = pd.to_datetime(tr["signal_date"])
    tk = tr["ticker"].to_numpy()

    def at(frame):
        return frame.reindex(index=d, columns=None).to_numpy()[np.arange(len(tr)),
                                                                [frame.columns.get_loc(t) for t in tk]]
    tr["M_pct"] = rf.M_pct.reindex(d).to_numpy()
    for f in ("v", "u", "x", "g_seg", "dv_med20"):
        tr[f] = at(getattr(rf, f))
    return tr


def partial_excess(tr: pd.DataFrame, prices: engine.Prices, k: int, side: str, delay: int = 1,
                   end=None) -> pd.Series:
    """Per-event beta-adjusted excess from the entry open to the CLOSE of session k
    (session 1 = the entry day), against the same basket universe as the engine.
    Gross of costs. Events whose session k falls after ``end`` are NaN."""
    dates = prices.dates
    ao = prices.adj_open.to_numpy(float)
    ac = prices.adj_close.to_numpy(float)
    ok = engine.tradable(prices)
    pos = {dd: i for i, dd in enumerate(dates)}
    col = {c: j for j, c in enumerate(prices.adj_close.columns)}
    sgn = 1.0 if side == "long" else -1.0
    out = np.full(len(tr), np.nan)
    basket = {}
    for n, (sd, tk, b) in enumerate(zip(pd.to_datetime(tr["signal_date"]), tr["ticker"], tr["beta"])):
        t = pos[sd]
        e = t + delay
        c = e + k - 1
        if c >= len(dates) or (end is not None and dates[c] > pd.Timestamp(end)):
            continue
        if (t, c) not in basket:
            univ = ok[t] & np.isfinite(ao[e])
            g = ac[c] / ao[e]
            basket[(t, c)] = np.nanmean(np.where(univ & np.isfinite(g), g, np.nan)) - 1
        j = col[tk]
        Ri = ac[c, j] / ao[e, j] - 1
        out[n] = sgn * (Ri - b * basket[(t, c)])
    return pd.Series(out, index=tr.index)


def _cluster_ids(dates: pd.Series, gap: int, sessions: pd.DatetimeIndex) -> np.ndarray:
    """Episode id per signal date: a new episode starts after more than ``gap`` sessions."""
    pos = {d: i for i, d in enumerate(sessions)}
    u = sorted(set(pd.to_datetime(dates)))
    ids, cur, last = {}, -1, None
    for d in u:
        i = pos[d]
        if last is None or i - last > gap:
            cur += 1
        ids[d] = cur
        last = i
    return np.array([ids[d] for d in pd.to_datetime(dates)])


def _ols_intercept(tr: pd.DataFrame) -> dict:
    X = np.column_stack([np.ones(len(tr)), tr["basket_ret"].to_numpy(float)])
    r = gates.ols_cluster(tr["excess_net"].to_numpy(float), X, pd.to_datetime(tr["signal_date"]).to_numpy())
    return {"a": float(r["b"][0]), "a_se": float(r["se"][0]), "b": float(r["b"][1]),
            "clusters": r["clusters"], "n": r["n"]}


def _slope(tr: pd.DataFrame, col: str) -> dict:
    t = tr[np.isfinite(tr[col].astype(float))]
    X = np.column_stack([np.ones(len(t)), t[col].to_numpy(float)])
    r = gates.ols_cluster(t["excess_net"].to_numpy(float), X, pd.to_datetime(t["signal_date"]).to_numpy())
    return {"slope": float(r["b"][1]), "se": float(r["se"][1]), "t": float(r["t"][1]), "n": r["n"]}


def _mean(x) -> float | None:
    x = pd.Series(x, dtype=float).dropna()
    return float(x.mean()) if len(x) else None


# ------------------------------------------------------------------ signatures

def sr_signature(fam, v, rf, prices, start, end, cache, runs) -> dict:
    """REQUIRED: S1 intercept > 0; S2 shock-type contrast; S3 same-stock guard. Others reported."""
    from . import backtest
    res = runs[v.name]
    tr = event_frame(res, rf)
    out = {"n_events": int(len(tr))}
    if tr.empty:
        return {**out, "pass": False}
    s1 = _ols_intercept(tr)
    vc = dataclasses.replace(v, params={**v.params, "day": "not_systemic"})
    rc, _ = backtest.run_window(vc, rf, prices, start, end, cache=cache)
    contrast = _mean(rc.trades["excess_net"]) if len(rc.trades) else None
    base = backtest.same_stock_baseline(rf, prices, v.H, start, end)
    adj = tr["excess_net"] - tr["ticker"].map(base)
    e_mean = float(tr["excess_net"].mean())
    req = {"S1_intercept_gt_0": {**s1, "pass": bool(s1["a"] > 0)},
           "S2_shock_type_contrast": {"systemic_mean_e": e_mean, "not_systemic_mean_e": contrast,
                                      "n_contrast": int(len(rc.trades)),
                                      "pass": bool(contrast is not None and e_mean > contrast)},
           "S3_same_stock_guard": {"mean_e_minus_base": _mean(adj), "pass": bool(_mean(adj) is not None
                                                                                  and _mean(adj) > 0)}}
    rep = {}
    rep["slope_e_on_M_pct"] = _slope(tr, "M_pct")
    if v.H == 5:
        p3 = partial_excess(tr, prices, 3, "long", end=end)
        p5 = partial_excess(tr, prices, 5, "long", end=end)
        rep["front_loading_s1_3_over_s1_5"] = (float(p3.mean() / p5.mean())
                                               if p5.notna().any() and p5.mean() != 0 else None)
    p20 = partial_excess(tr, prices, 20, "long", end=end)
    pH = partial_excess(tr, prices, v.H, "long", end=end)
    rep["permanence_s20_over_sH"] = float(p20.mean() / pH.mean()) if p20.notna().any() and pH.mean() else None
    vn = dataclasses.replace(v, params={**v.params, "news": "news"})
    rn, _ = backtest.run_window(vn, rf, prices, start, end, cache=cache)
    rep["news_excluded_set_mean_e"] = _mean(rn.trades["excess_net"]) if len(rn.trades) else None
    rep["news_excluded_set_n"] = int(len(rn.trades))
    score = tr["u" if v.params.get("beta_only") else "v"]
    q1, q2 = np.nanquantile(score, [1 / 3, 2 / 3])
    rep["most_displaced_tercile_mean_e"] = _mean(tr.loc[score <= q1, "excess_net"])
    rep["least_displaced_tercile_mean_e"] = _mean(tr.loc[score > q2, "excess_net"])
    rep["tercile_cut_33pct"] = float(q1)
    liq = tr["dv_med20"].to_numpy(float)
    med = np.nanmedian(liq)
    rep["less_liquid_half_mean_e"] = _mean(tr.loc[liq <= med, "excess_net"])
    rep["more_liquid_half_mean_e"] = _mean(tr.loc[liq > med, "excess_net"])
    cl = _cluster_ids(tr["signal_date"], 5, prices.dates)
    loo = [float(tr.loc[cl != c, "excess_net"].mean()) for c in np.unique(cl)]
    rep["leave_one_cluster_out_min_mean_e"] = float(min(loo))
    rep["n_clusters"] = int(len(np.unique(cl)))
    rep["SR4_by_segment_mean_e"] = None
    if v.params.get("beta_only"):
        seg = signals.segments(prices.adj_close.columns)
        rep["SR4_by_segment_mean_e"] = tr.groupby(tr["ticker"].map(seg))["excess_net"].mean().to_dict()
    return {**out, "required": req, "reported": rep,
            "pass": all(x["pass"] for x in req.values())}


def nd_signature(fam, v, rf, prices, start, end, cache, runs) -> dict:
    """REQUIRED: N1 intercept > 0; N2 gradual; N3 information split. Others reported."""
    from . import backtest
    res = runs[v.name]
    tr = event_frame(res, rf)
    out = {"n_events": int(len(tr))}
    if tr.empty:
        return {**out, "pass": False}
    n1 = _ols_intercept(tr)
    e_mean = float(tr["excess_net"].mean())
    p2 = partial_excess(tr, prices, 2, "short_info", end=end)
    n2_ok = bool(e_mean > 0 and p2.notna().any() and p2.mean() <= e_mean / 3)
    vc = dataclasses.replace(v, params={**v.params, "news": "no_gap"})
    rc, _ = backtest.run_window(vc, rf, prices, start, end, cache=cache)
    contrast = _mean(rc.trades["excess_net"]) if len(rc.trades) else None
    req = {"N1_intercept_gt_0": {**n1, "pass": bool(n1["a"] > 0)},
           "N2_gradual": {"mean_e_H": e_mean, "mean_e_to_close_s2_gross": _mean(p2), "pass": n2_ok},
           "N3_information_split": {"news_mean_e": e_mean, "no_news_mean_e": contrast,
                                    "n_contrast": int(len(rc.trades)),
                                    "pass": bool(contrast is not None and e_mean > contrast)}}
    rep = {}
    vs = dataclasses.replace(v, params={**v.params, "day": "systemic"})
    rs, _ = backtest.run_window(vs, rf, prices, start, end, cache=cache)
    rep["systemic_day_contrast_mean_e"] = _mean(rs.trades["excess_net"]) if len(rs.trades) else None
    rep["systemic_day_contrast_n"] = int(len(rs.trades))
    half = partial_excess(tr, prices, v.H // 2, "short_info", end=end)
    full = partial_excess(tr, prices, v.H, "short_info", end=end)
    rep["second_half_share"] = float(1 - half.mean() / full.mean()) if full.notna().any() and full.mean() else None
    tr2 = tr.assign(log_dv=np.log(tr["dv_med20"].astype(float)))
    rep["slope_e_on_log_dollar_volume"] = _slope(tr2, "log_dv")
    q1 = float(np.nanquantile(tr["v"], 1 / 3))
    rep["v_tercile_cut_33pct"] = q1
    rep["most_negative_v_tercile_mean_e"] = _mean(tr.loc[tr["v"] <= q1, "excess_net"])
    rep["rest_mean_e"] = _mean(tr.loc[tr["v"] > q1, "excess_net"])
    rep["x_ge_4_mean_e"] = _mean(tr.loc[tr["x"] >= 4, "excess_net"])
    rep["x_3_to_4_mean_e"] = _mean(tr.loc[tr["x"] < 4, "excess_net"])
    base = backtest.same_stock_baseline(rf, prices, v.H, start, end)
    rep["same_stock_adjusted_mean_e"] = _mean(tr["excess_net"] + tr["ticker"].map(base))
    return {**out, "required": req, "reported": rep, "pass": all(x["pass"] for x in req.values())}


SIGNATURES = {"SR": sr_signature, "ND": nd_signature}


# ----------------------------------------------------------------------------- RF

RF_VARIANTS = ("RF1", "RF2", "RF3")
GAMMA = 5.0


def har_forecast(cc: pd.Series, min_rows: int = 60) -> pd.Series:
    """Recursive HAR forecast of the basket's next-5-session daily volatility.

    At each t with complete HAR lags, OLS of y_s = log forward 5-session daily vol
    on [1, HAR lags at s] over rows s <= t-5 (their targets end by t) from the
    first complete row, with at least ``min_rows`` rows. sigma_hat_t = exp(fit).
    """
    from .. import validate
    R = cc.to_numpy(float)
    y, X = validate.har_design(R)
    good = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    xok = np.all(np.isfinite(X), axis=1)
    first = int(np.argmax(good))
    out = np.full(len(R), np.nan)
    for t in np.flatnonzero(xok):
        rows = np.flatnonzero(good[: max(t - 4, 0)])
        rows = rows[rows >= first]
        if len(rows) < min_rows:
            continue
        b = np.linalg.lstsq(X[rows], y[rows], rcond=None)[0]
        out[t] = np.exp(X[t] @ b)
    return pd.Series(out, index=cc.index)


def rf_weights(name: str, M_pct: pd.Series, sigma_hat: pd.Series, half_day: pd.Series) -> pd.Series:
    """Exposure E_t in [0.25, 1] decided at each close, per the declared state machines."""
    idx = sigma_hat.index
    m = M_pct.reindex(idx)
    sh = sigma_hat.to_numpy(float)
    star = pd.Series(sh, index=idx).expanding().median().to_numpy()     # causal target: median of s <= t
    hd = half_day.reindex(idx).fillna(False).to_numpy(bool)
    E = np.ones(len(idx))
    e, state = 1.0, False
    for t in range(len(idx)):
        mp = m.iloc[t]
        tgt = np.clip(star[t] / sh[t], 0.25, 1.0) if np.isfinite(sh[t]) and np.isfinite(star[t]) else np.nan
        if hd[t]:
            E[t] = e
            continue
        if name == "RF1":
            if np.isfinite(mp):
                if mp >= 1.5:
                    e = 0.5
                elif mp < 1.0 and e < 1.0:
                    e = min(1.0, e + 0.25)
        elif name == "RF2":
            if np.isfinite(mp):
                if mp >= 1.5:
                    state = True
                elif state and mp < 1.0:
                    state = False
                if state:
                    if np.isfinite(tgt):
                        e = float(tgt)
                elif mp < 1.0 and e < 1.0:
                    e = min(1.0, e + 0.25)
        elif name == "RF3":
            if np.isfinite(tgt) and abs(tgt - e) >= 0.10:
                e = float(tgt)
        else:
            raise ValueError(name)
        E[t] = e
    return pd.Series(E, index=idx)


def _u(r: np.ndarray, gamma: float) -> np.ndarray:
    return r - 0.5 * gamma * r ** 2


def _cvar95(r: np.ndarray) -> float:
    q = np.quantile(r, 0.05)
    return float(r[r <= q].mean())


def _maxdd(r: np.ndarray) -> float:
    eq = np.cumprod(1 + r)
    return float((eq / np.maximum.accumulate(eq) - 1).min())


def rf_stats(E: pd.Series, legs: pd.DataFrame, start, end, cost_bps: float = 10.0, delay: int = 1,
             gamma: float = GAMMA) -> dict:
    out = engine.run_exposure(E, legs, cost_bps=cost_bps, delay=delay)
    w = gates.window(out["net"], start, end)
    win = out.loc[w.index]
    Ebar = float(win["w_intraday"].mean())
    static = (Ebar * win["basket"]).to_numpy(float)
    net = win["net"].to_numpy(float)
    d = pd.Series(_u(net, gamma) - _u(static, gamma), index=win.index)
    st = gates.variant_stats(d, start, end)
    timing = float(((win["w_intraday"] - Ebar) * win["basket"]).mean())
    return {"E_bar": Ebar, "mean_d": st["mean_daily"], "se_d": st["se_daily"], "t_d": st["t"],
            "var_ratio": float(np.var(net, ddof=1) / np.var(static, ddof=1)),
            "cvar95_net": _cvar95(net), "cvar95_static": _cvar95(static),
            "maxdd_net": _maxdd(net), "maxdd_static": _maxdd(static),
            "timing_of_mean": timing,
            "mean_d_gamma2": float(np.mean(_u(net, 2.0) - _u(static, 2.0))),
            "mean_d_gamma10": float(np.mean(_u(net, 10.0) - _u(static, 10.0))),
            "sharpe_net": float(net.mean() / net.std(ddof=1) * np.sqrt(252)),
            "sharpe_basket_100": float(win["basket"].mean() / win["basket"].std(ddof=1) * np.sqrt(252)),
            "reduced_sessions": int((win["w_intraday"] < 1).sum()), "n_days": int(len(win)),
            "d": d}


def rf_evaluate(rf: signals.RuleFeatures, prices: engine.Prices, start, end) -> dict:
    """RF's declared adoption path: 1-SE on mean d (RF1, RF2, RF3 order), then A1-A3."""
    legs = engine.basket_legs(prices)
    sh = har_forecast(legs["cc"])
    res = {}
    for name in RF_VARIANTS:
        E = rf_weights(name, rf.M_pct, sh, rf.half_day)
        res[name] = {"base": rf_stats(E, legs, start, end),
                     "cost25": rf_stats(E, legs, start, end, cost_bps=25.0),
                     "delay2": rf_stats(E, legs, start, end, delay=2), "E": E}
    stats = [{"mean_daily": res[n]["base"]["mean_d"], "se_daily": res[n]["base"]["se_d"]} for n in RF_VARIANTS]
    k = gates.one_se_choice(stats)
    ch = RF_VARIANTS[k]
    b = res[ch]
    a1 = b["base"]["var_ratio"] <= 0.85
    a2 = all(r["mean_d"] >= -0.5 * r["se_d"] for r in (b["base"], b["cost25"], b["delay2"]))
    a3 = (b["base"]["cvar95_net"] >= b["base"]["cvar95_static"]) and (b["base"]["maxdd_net"] >= b["base"]["maxdd_static"])
    reported = {"RF2_minus_RF3_mean_d": res["RF2"]["base"]["mean_d"] - res["RF3"]["base"]["mean_d"]}
    return {"variants": res, "chosen": ch,
            "criteria": {"A1_var_ratio_le_0_85": bool(a1), "A2_non_inferior_10_25_delay": bool(a2),
                         "A3_cvar_and_maxdd_no_worse": bool(a3)},
            "adopted": bool(a1 and a2 and a3), "reported": reported, "sigma_hat": sh}
