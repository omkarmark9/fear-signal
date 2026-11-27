"""Both models on recent sessions, and what the trading model does next.

Model 1 is the frozen v3 fear model on the latest bars (pipeline._live_state).
Model 2 runs each family's chosen variant on those readings, using
signals.from_live. Its tickets are dated on the NYSE calendar, and open
tickets are marked to the latest close:
- gross return;
- excess over beta x the equal-weight basket bought at the same open.

A ticket's tier comes from the dev selection and the single holdout read
(research/trading/candidates.md, "Tiers"). Before those exist, every ticket
is Tier 0 (display only).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import calendar, engine, families as F, protocol, registry, signals

LIVE_OUT = protocol.OUT / "live"


def state(source: str = "yahoo"):
    from .. import pipeline
    mk, st, sig, feat, report = pipeline._live_state(source)
    prices = engine.Prices.load(live=True)
    if prices.dates[-1] != feat.obs.dates[-1]:
        raise RuntimeError(f"price cache ends {prices.dates[-1].date()} but the fear panel ends "
                           f"{feat.obs.dates[-1].date()}: refresh both (fearsignal live; fearsignal trading-data)")
    inp = signals.from_live(mk, st, feat, prices)
    rf = signals.rule_features(inp, feat)
    return mk, st, rf, prices, report


def family_status() -> dict:
    """Tier per family from config/trading_selected.json and the holdout results."""
    out = {"SR": {"chosen": "SR1", "tier": 0, "why": "dev selection not run"},
           "ND": {"chosen": "ND1", "tier": 0, "why": "dev selection not run"},
           "RF": {"chosen": "RF1", "state": "OFF", "why": "dev selection not run"}}
    if not protocol.SELECTED.exists():
        return out
    sel = json.loads(protocol.SELECTED.read_text())["families"]
    hpath = protocol.OUT / "holdout" / "holdout_results.json"
    # a read counts only if it was made under the committed selection (audit SP-6)
    hold = json.loads(hpath.read_text()) if protocol.holdout_matches_selection() else {}
    for k in ("SR", "ND"):
        s = sel[k]
        o = {"chosen": s["chosen"], "adopted": s["adopted"]}
        h = hold.get(f"{k}.{s['chosen']}")
        if not s["adopted"]:
            o.update(tier=0, why="not adopted on dev")
        elif h is None:
            o.update(tier=0, why="holdout not read under the committed selection")
        elif h.get("status") != "confirmation read":
            o.update(tier=0, why="no confirmation read")
        elif h["stats"]["mean_daily"] <= 0:
            o.update(tier=0, why="holdout disagrees")
        elif h["stats"]["t"] < 1:
            o.update(tier=1, why="adopted; holdout agrees with t < 1")
        else:
            o.update(tier=2, why="adopted; holdout agrees with t >= 1")
        out[k] = o
    r = sel["RF"]
    h = hold.get(f"RF.{r['chosen']}")
    confirmed = bool(h and h.get("confirmed") and h.get("status") == "confirmation read")
    out["RF"] = {"chosen": r["chosen"], "adopted": r["adopted"], "state": "ON" if confirmed else "OFF",
                 "why": "adopted and confirmed on 2026" if confirmed else
                 ("not adopted on dev" if not r["adopted"] else "not confirmed on 2026")}
    return out


def model1_table(mk: pd.DataFrame, st: pd.DataFrame, rf: signals.RuleFeatures, days) -> pd.DataFrame:
    rows = []
    for d in days:
        s = st[st["date"] == d]
        big = s[s["x"] >= 3].sort_values("x", ascending=False)
        top = s.sort_values("Z", ascending=False).head(5)
        rows.append({"date": d.date(), "M_pct": mk.loc[d, "M_pct"], "M_pct_lo": mk.loc[d, "M_pct_lo"],
                     "band": mk.loc[d, "band"], "high_conviction_fear": bool(mk.loc[d, "high_conviction"]),
                     "breadth": mk.loc[d, "breadth"], "share_x_ge_2": float(rf.share_hit2.get(d, np.nan)),
                     "systemic": bool(rf.systemic.get(d, False)), "half_day": bool(rf.half_day.get(d, False)),
                     "n_x_ge_3": int(len(big)), "x_ge_3": ", ".join(big["ticker"].head(8)),
                     "most_afraid": ", ".join(f"{r.ticker} {r.Z:.2f}" for r in top.itertuples())})
    return pd.DataFrame(rows).set_index("date")


def _mark(v: registry.Variant, rows: pd.DataFrame, rf, prices: engine.Prices, cal: pd.DatetimeIndex) -> pd.DataFrame:
    """Date and mark each ticket: entry next open, exit H sessions later, P/L to the last close."""
    dates = prices.dates
    last = len(dates) - 1
    ao = prices.adj_open.to_numpy(float)
    ac = prices.adj_close.to_numpy(float)
    ok = engine.tradable(prices)
    pos = {d: i for i, d in enumerate(dates)}
    col = {c: j for j, c in enumerate(prices.adj_close.columns)}
    out = []
    sgn = 1.0 if v.side == "long" else -1.0
    # audit SP-3: realised outcomes inside the 2026 holdout stay hidden until the holdout has
    # been read under the committed selection (the read is the only sanctioned look)
    from . import gates
    hide_until = None if protocol.holdout_matches_selection() else gates.HOLDOUT[1]
    for r in rows.itertuples():
        t = pos[pd.Timestamp(r.date)]
        entry = calendar.offset(cal, dates[t], 1)
        exit_ = calendar.offset(cal, dates[t], 1 + v.H)
        j = col[r.ticker]
        beta = float(rf.beta.iloc[t, j]) if np.isfinite(rf.beta.iloc[t, j]) else 1.0
        rec = {"signal": dates[t].date(), "ticker": r.ticker, "family": v.family, "variant": v.name,
               "side": "long" if v.side == "long" else "bearish info (not shortable)",
               "entry_open": entry.date(), "exit_open": exit_.date(), "beta": round(beta, 3),
               "x": round(float(rf.x.iloc[t, j]), 2), "v": round(float(rf.v.iloc[t, j]), 2),
               "u": round(float(rf.u.iloc[t, j]), 2), "g_seg": round(float(rf.g_seg.iloc[t, j]), 2),
               "M_pct": round(float(rf.M_pct.iloc[t]), 2), "status": "pending", "ret": None, "excess": None}
        e = t + 1
        if hide_until is not None and dates[t] <= hide_until:
            rec["status"] = "outcome hidden until the holdout read"
        elif e <= last and np.isfinite(ao[e, j]):
            x_end = min(t + 1 + v.H, last)
            closed = t + 1 + v.H <= last
            px_end = ao[t + 1 + v.H, j] if closed else ac[x_end, j]
            univ = ok[t] & np.isfinite(ao[e])
            b_end = (ao[t + 1 + v.H] if closed else ac[x_end])
            g = b_end / ao[e]
            RB = float(np.nanmean(np.where(univ & np.isfinite(g), g, np.nan)) - 1)
            R = float(px_end / ao[e, j] - 1)
            rec.update(status="closed" if closed else "open", ret=R, excess=sgn * (R - beta * RB),
                       marked_to=(dates[t + 1 + v.H] if closed else dates[x_end]).date())
        out.append(rec)
    return pd.DataFrame(out)


def tickets(rf, prices, status: dict, days: pd.DatetimeIndex) -> pd.DataFrame:
    byname = {v.name: v for fam in F.HOLM_FAMILIES for v in fam.variants}
    cal = calendar.sessions(prices.dates)
    frames = []
    for fam in ("SR", "ND"):
        v = byname[status[fam]["chosen"]]
        ent = registry.entries(v, rf)
        ent["date"] = pd.to_datetime(ent["date"])
        lookback = days.min() - pd.Timedelta(days=int(v.H * 1.6) + 7)
        rows = ent[ent["date"] >= lookback]
        if len(rows):
            m = _mark(v, rows, rf, prices, cal)
            m["tier"] = status[fam]["tier"]
            frames.append(m)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def rf_exposure(rf, prices, status: dict, days) -> pd.Series:
    sh = F.har_forecast(engine.basket_legs(prices)["cc"])
    E = F.rf_weights(status["RF"]["chosen"], rf.M_pct, sh, rf.half_day)
    return E.reindex(days)


def main(n_days: int = 6) -> dict:
    mk, st, rf, prices, report = state()
    days = mk.index[-n_days:]
    status = family_status()
    m1 = model1_table(mk, st, rf, days)
    tk = tickets(rf, prices, status, days)
    E = rf_exposure(rf, prices, status, days)
    cal = calendar.sessions(prices.dates)
    nxt = calendar.offset(cal, prices.dates[-1], 1)
    pd.set_option("display.width", 220)
    print("MODEL 1 (fear, frozen v3):")
    print(m1.round(2).to_string())
    print("\nMODEL 2 family status:", json.dumps(status, indent=1, default=str))
    if len(tk):
        print("\nMODEL 2 tickets (signal at the close; entry at the next open):")
        print(tk.to_string(index=False))
    print("\nRF exposure E (decided at each close):", E.round(2).to_dict())
    print("\nnext session:", nxt.date())
    LIVE_OUT.mkdir(parents=True, exist_ok=True)
    rec = {"as_of_close": str(prices.dates[-1].date()), "next_session": str(nxt.date()), "status": status,
           "model1": m1.reset_index().astype(str).to_dict(orient="records"),
           "tickets": tk.astype(str).to_dict(orient="records") if len(tk) else [],
           "rf_exposure": {str(k.date()): float(v) for k, v in E.items()}, "data": report}
    return rec
