"""v2 exploration on dev: rule templates, the trial runner and the append-only registry.

research/trading/v2/PLAN.md declares the permutation space and the
selection rule. Each trial runs on dev only (2024-01-02..2025-12-31, with
every cohort exiting by 2025-12-31). Its daily performance series, where
positive means better than its benchmark, goes into the registry. The
registry-wide multiple-testing statistics (multitest.py) are computed over
every series in the registry.

Performance series by kind:
- **event:** the 1/H constant-capital excess series (each of the last H
  cohorts holds 1/H of the sleeve, cash otherwise), against beta x the
  basket.
- **portfolio:** the vol-matched differential r_s - (sd_s / sd_b) r_b
  against the equal-weight portfolio of the same universe on the same
  rebalance schedule. Its mean > 0 if and only if Sharpe_s > Sharpe_b.
- **timing:** the utility differential u(net) - u(E_bar x basket), with
  gamma = 5, as declared for RF.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import backtest, engine, families as F, gates, protocol, signals, sizing

REG = protocol.OUT / "v2"
TRIALS = REG / "trials.jsonl"
SERIES = REG / "trial_series.parquet"
START, END = gates.DEV


@dataclass
class Trial:
    family: str                      # V1, P, T, E1..E5
    kind: str                        # event | portfolio | timing
    params: dict = field(default_factory=dict)
    round: int = 1

    @property
    def id(self) -> str:
        return self.family + "|" + json.dumps(self.params, sort_keys=True)


class Ctx:
    """Inputs shared by every trial (point-in-time, hash-locked)."""

    def __init__(self, rf=None, prices=None):
        if rf is None:
            rf, prices = backtest.load_inputs()
        self.rf, self.prices = rf, prices
        self.cache: dict = {}
        self.legs = engine.basket_legs(prices)
        self._st: dict = {}
        self._ew = None
        self._rho: dict = {}
        self.refits = sizing.load_refits() if (protocol.RAW / "pit_refits.json").exists() else []
        self.earnings = None

    def stilde(self, h: int) -> pd.DataFrame:
        if h not in self._st:
            rf = self.rf
            self._st[h] = sizing.unscaled_sigma_tilde(rf.sigma, rf.P_calm, rf.P_stressed, rf.P_panic,
                                                      self.refits, h)
        return self._st[h]

    def ewma(self) -> pd.DataFrame:
        if self._ew is None:
            self._ew = sizing.ewma_sigma(self.rf.r.where(self.rf.eligible))
        return self._ew

    def flags(self, rule: str, H: int) -> pd.DataFrame:
        """True where a name had an idiosyncratic-shock signal in the last H sessions (known at t)."""
        key = ("flags", rule, H)
        if key not in self._st:
            rf = self.rf
            names = F._names(rf)
            calm = _days(_day(rf, "calm"), rf.x)
            down = (rf.x >= 2) & (rf.v <= -2) & (rf.g_seg <= -1.5)
            up = (rf.g_seg >= 1.5) & (rf.r >= rf.sigma)
            if rule in ("down", "up", "both"):
                ev = {"down": down, "up": up, "both": down | up}[rule] & names & calm
            else:                                     # earnings reactions: "earn1.5", "earn3.0", "union1.5"
                thr = float(rule.replace("earn", "").replace("union", ""))
                earn = earnings_sessions(self) & (rf.v.abs() >= thr) & names
                ev = earn | ((down | up) & names & calm) if rule.startswith("union") else earn
            self._st[key] = ev.astype(float).rolling(H, min_periods=1).max().fillna(0).astype(bool)
        return self._st[key]

    def rho(self, t: int) -> float:
        if t not in self._rho:
            self._rho[t] = sizing.equicorrelation(self.rf.r, self.rf.eligible, t)
        return self._rho[t]


# ------------------------------------------------------------------ event rules

def _days(day: pd.Series, like: pd.DataFrame) -> pd.DataFrame:
    return F._days(day, like)


def _day(rf, kind: str) -> pd.Series:
    full = ~rf.half_day
    if kind == "calm":
        return ~rf.systemic & full
    if kind == "systemic":
        return rf.systemic & full
    if kind == "any":
        return full
    raise ValueError(kind)


def event_rule(tr: Trial, ctx: Ctx):
    """(entry mask, priority frame, H, cap, side, delay) for an event trial."""
    rf, p = ctx.rf, tr.params
    names = F._names(rf)
    if tr.family == "V1":
        byname = {v.name: v for fam in F.HOLM_FAMILIES for v in fam.variants}
        v = byname[p["variant"]]
        return v.select(rf, v), getattr(rf, v.priority), v.H, v.cap, v.side, 1
    if tr.family == "E1":            # news drift, bearish information
        m = names & (rf.x >= p["x"]) & (rf.v <= -2) & (rf.g_seg <= p["g"]) & _days(_day(rf, p["day"]), rf.x)
        return m, rf.v, p["H"], 5, "short_info", 1
    if tr.family == "E2":            # good-news drift, long
        m = (names & (rf.g_seg >= p["g"]) & (rf.r >= rf.sigma) & (rf.relvol >= p["rv"])
             & _days(_day(rf, "calm"), rf.x))
        return m, -rf.g_seg, p["H"], 5, "long", 1
    if tr.family == "E3":            # systemic reversal, long
        day = _day(rf, "systemic")
        if p.get("m") is not None:
            day = day & (rf.M_pct >= p["m"])
        m = names & (rf.x >= 2) & (rf.v <= p["v"]) & (rf.g_seg > -2) & _days(day, rf.x)
        return m, rf.v, p["H"], 10, "long", p["delay"]
    if tr.family == "E4":            # earnings reaction drift
        rs = earnings_sessions(ctx)
        good = p["side"] == "good"
        m = names & rs & ((rf.v >= p["thr"]) if good else (rf.v <= -p["thr"]))
        prio = -rf.v if good else rf.v
        return m, prio, p["H"], 10, "long" if good else "short_info", 1
    if tr.family == "E2F":           # good-news gap-up FADED: bearish information (Round 2; counts the sign flip)
        m = (names & (rf.g_seg >= p["g"]) & (rf.r >= rf.sigma) & (rf.relvol >= p["rv"])
             & _days(_day(rf, "calm"), rf.x))
        return m, -rf.g_seg, p["H"], 5, "short_info", 1
    if tr.family == "E6":            # any idiosyncratic shock on a calm day -> underperformance (MAX/IVOL-like)
        down = (rf.x >= 2) & (rf.v <= -2) & (rf.g_seg <= -1.5)
        up = (rf.g_seg >= 1.5) & (rf.r >= rf.sigma)
        m = names & (down | up) & _days(_day(rf, "calm"), rf.x)
        return m, -rf.v.abs(), p["H"], 10, "short_info", 1
    if tr.family in ("MAXA", "IVOLA"):  # periodic top-10 by MAX or IVOL: bearish information
        score = max_return(ctx) if tr.family == "MAXA" else ivol(ctx)
        score = score.where(names)
        rank = score.rank(axis=1, ascending=False, method="first")
        every = np.zeros(len(score), bool)
        every[::5] = True
        m = (rank <= 10) & _days(pd.Series(every, index=score.index) & ~rf.half_day, rf.x)
        return m, -score, p["H"], 10, "short_info", 1
    if tr.family == "E7":            # earnings reaction of EITHER sign as bearish information (Round 3)
        m = names & earnings_sessions(ctx) & (rf.v.abs() >= p["thr"])
        mc, zc = p.get("m_cond"), p.get("z_cond")      # Round 4: does model 1 modulate it?
        if mc == "hi":
            m = m & _days(rf.M_pct >= 1.0, rf.x)
        elif mc == "lo":
            m = m & _days(rf.M_pct < 1.0, rf.x)
        if zc == "hi":
            m = m & (rf.Z >= 2.0)
        elif zc == "lo":
            m = m & (rf.Z < 2.0)
        return m, -rf.v.abs(), p["H"], 10, "short_info", 1
    if tr.family == "E8":            # EPS surprise drift (Round 5)
        sp = earnings_surprise(ctx)
        good = p["side"] == "good"
        m = names & ((sp >= p["thr"]) if good else (sp <= -p["thr"]))
        return m, (-sp if good else sp), p["H"], 10, "long" if good else "short_info", 1
    if tr.family == "E9":            # reaction against surprise (Round 5)
        sp = earnings_surprise(ctx)
        if p["case"] == "beat_drop":
            m = names & (sp > 0) & (rf.v <= -2)
            return m, rf.v, p["H"], 10, "short_info", 1
        m = names & (sp < 0) & (rf.v >= 2)
        return m, -rf.v, p["H"], 10, "long", 1
    if tr.family == "E10":           # earnings-announcement premium: buy ahead of the report (Round 5)
        es = earnings_sessions(ctx).to_numpy(bool)
        m = np.zeros_like(es)
        lead = 6                                            # signal 6 sessions before the reaction session
        m[:-lead] = es[lead:]
        m = pd.DataFrame(m, index=rf.x.index, columns=rf.x.columns) & names
        return m, -rf.x.fillna(0), p["H"], 10, "long", 1
    if tr.family == "E5":            # cross-sectional fear factor
        score = (rf.Z if p["signal"] == "Z" else inputs_I(ctx)).where(names)
        asc = p["end"] == "low"
        rank = score.rank(axis=1, ascending=asc, method="first")
        m = (rank <= 10) & _days(~rf.half_day, rf.x)
        return m, (score if asc else -score), p["H"], 10, "long", 1
    raise ValueError(tr.family)


def max_return(ctx: Ctx, window: int = 21) -> pd.DataFrame:
    """MAX (Bali, Cakici & Whitelaw 2011): the largest daily log return over the last
    ``window`` sessions, in own sigma, known at the close of t."""
    rf = ctx.rf
    return (rf.r / rf.sigma).rolling(window, min_periods=15).max()


def ivol(ctx: Ctx, window: int = 63) -> pd.DataFrame:
    """Idiosyncratic volatility (Ang, Hodrick, Xing & Zhang 2006): the s.d. of the beta
    residual u over the last ``window`` sessions, known at the close of t."""
    return ctx.rf.u.rolling(window, min_periods=40).std()


def inputs_I(ctx: Ctx) -> pd.DataFrame:
    """The fear model's relative fear I, point-in-time, on the price grid."""
    if not hasattr(ctx, "_I"):
        inp = signals.build()
        ctx._I = inp.wide("I").where(ctx.rf.eligible)
    return ctx._I


def earnings_surprise(ctx: Ctx) -> pd.DataFrame:
    """EPS surprise on each reaction session, (actual - estimate) / max(|estimate|, 0.05), clipped
    to [-2, 2]. Scale-free, so the vendor's mixed split bases within a few names cancel. NaN
    elsewhere or when either number is missing."""
    if getattr(ctx, "_surprise", None) is None:
        es = earnings_sessions(ctx)                       # also builds ctx._earn_rows
        out = pd.DataFrame(np.nan, index=es.index, columns=es.columns)
        for (i, tk), (est, act) in ctx._earn_eps.items():
            if est is None or act is None:
                continue
            out.iloc[i, out.columns.get_loc(tk)] = float(np.clip((act - est) / max(abs(est), 0.05), -2, 2))
        ctx._surprise = out
    return ctx._surprise


def earnings_sessions(ctx: Ctx) -> pd.DataFrame:
    """True on each name's earnings REACTION session: the report date for 'am'
    reports, the next session for 'pm' (after-close) reports. Unknown timing
    counts as the report date, so the reaction session is never too early."""
    if ctx.earnings is None:
        path = protocol.RAW / "earnings_calendar.json"
        rows = json.loads(path.read_text())["rows"]
        # an unverified (tentative) date is dropped when the same fiscal quarter has a verified date
        verified = {(r["symbol"], r.get("year"), r.get("quarter")) for r in rows if r["report"].get("verified")}
        rows = [r for r in rows if r["report"].get("verified")
                or (r["symbol"], r.get("year"), r.get("quarter")) not in verified]
        dates = ctx.prices.dates
        cols = ctx.prices.adj_close.columns
        m = pd.DataFrame(False, index=dates, columns=cols)
        ctx._earn_eps = {}
        for r in rows:
            tk = r["symbol"]
            if tk not in m.columns:
                continue
            d = pd.Timestamp(r["report"]["date"])
            i = int(np.searchsorted(dates, d))            # first session on or after the report date
            if i >= len(dates):
                continue
            if r["report"].get("timing") == "pm":
                i += 1 if dates[i] == d else 0            # after the close -> reacts next session
            if i < len(dates):
                m.iloc[i, m.columns.get_loc(tk)] = True
                eps = r.get("eps") or {}
                num = lambda v: float(v) if v not in (None, "") else None
                ctx._earn_eps[(i, tk)] = (num(eps.get("estimate")), num(eps.get("actual")))
        ctx.earnings = m
    return ctx.earnings


def _episodes(dates: pd.Series, sessions: pd.DatetimeIndex) -> np.ndarray:
    return F._cluster_ids(dates, 5, sessions)


def _event_stats(tr_df: pd.DataFrame, prices) -> dict:
    if len(tr_df) < 3:
        return {"n_events": int(len(tr_df))}
    e = tr_df["excess_net"].to_numpy(float)
    cl = _episodes(tr_df["signal_date"], prices.dates)
    r = gates.ols_cluster(e, np.ones((len(e), 1)), cl)
    return {"n_events": int(len(e)), "n_signal_days": int(tr_df["signal_date"].nunique()),
            "event_mean": float(e.mean()), "event_t_episode": float(r["t"][0]),
            "n_episodes": int(r["clusters"]), "event_median": float(np.median(e))}


def run_event(tr: Trial, ctx: Ctx) -> tuple[dict, pd.Series]:
    rf, prices = ctx.rf, ctx.prices
    mask, prio, H, cap, side, delay = event_rule(tr, ctx)
    sel = signals.suppress_reentry(mask, H, priority=prio, cap=cap)
    w = None
    if tr.params.get("weight") == "invvol":                 # risk parity across a cohort's names
        w = 1.0 / ctx.stilde(H)
    sig = signals.to_signals(sel, w)
    if w is not None:
        sig = sig[np.isfinite(sig["weight"]) & (sig["weight"] > 0)]

    def go(cost=10.0, dly=delay):
        last = gates.last_signal_date(prices.dates, H, dly, END)
        res = engine.run_cohorts(sig, prices, H=H, cost_bps=cost, delay=dly, side=side, start=START,
                                 end=last, cache=ctx.cache, beta=rf.beta)
        return res, gates.window(res.daily["excess_net"], START, END)

    res, x = go()
    st = gates.variant_stats(x, START, END)
    rec = {"mean": st["mean_daily"], "se": st["se_daily"], "t": st["t"],
           "ct_mean": float(gates.window(res.daily["ct_excess_net"], START, END).mean())}
    rec.update(_event_stats(res.trades, prices))
    if len(res.trades) >= 3:
        base = backtest.same_stock_baseline(rf, prices, H, START, END, delay=delay)
        adj = res.trades["excess_net"] + (-1 if side == "long" else 1) * res.trades["ticker"].map(base)
        rec["same_stock_adj_mean"] = float(adj.mean())
    rec["mean_25bps"] = float(go(cost=25.0)[1].mean())
    rec["mean_delay_plus1"] = float(go(dly=delay + 1)[1].mean())
    rec.update(H=H, side=side, delay=delay)
    return rec, x


# ------------------------------------------------------------- portfolio rules

def _rebalance_rows(prices, every: int = 5) -> list[int]:
    i0 = int(np.searchsorted(prices.dates, START))
    i1 = int(np.searchsorted(prices.dates, END, side="right")) - 1
    return list(range(i0, i1, every))


def portfolio_weights(tr: Trial, ctx: Ctx) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(strategy target weights, equal-weight target) on the rebalance rows."""
    rf, prices, p = ctx.rf, ctx.prices, tr.params
    ok_t = engine.tradable(prices)
    st20, ew = ctx.stilde(20), ctx.ewma()
    rows = _rebalance_rows(prices)
    S, B = [], []
    idx = prices.dates[rows]
    for t in rows:
        u = (ok_t[t] & rf.eligible.iloc[t].to_numpy(bool) & np.isfinite(st20.iloc[t].to_numpy(float))
             & np.isfinite(ew.iloc[t].to_numpy(float)))
        b = u / max(u.sum(), 1)
        sg_f, sg_e = st20.iloc[t].to_numpy(float), ew.iloc[t].to_numpy(float)
        w = np.zeros_like(b)
        kind = p["w"]
        if kind == "invvol_fear":
            w = np.where(u, sg_f ** -p["k"], 0.0)
        elif kind == "invvol_ewma":
            w = np.where(u, sg_e ** -p["k"], 0.0)
        elif kind == "ex_qnext":
            q = rf.q_next.iloc[t].to_numpy(float)
            cut = np.nanquantile(np.where(u, q, np.nan), 1 - p["q"])
            w = np.where(u & (q < cut), 1.0, 0.0)
        elif kind == "minvar_fear":
            rho = min(max(ctx.rho(t), 0.0), 0.95)
            a = np.where(u, 1 / sg_f, 0.0)
            n = u.sum()
            w = np.where(u, a * (a - rho * a.sum() / (1 + (n - 1) * rho)) / (1 - rho), 0.0)
            w = np.clip(w, 0, None)
        elif kind == "clean_basket":
            flag = ctx.flags(p["rule"], p["H"])
            w = np.where(u & ~flag.iloc[t].to_numpy(bool), 1.0, 0.0)
        elif kind == "invvol_fear_ex_qnext":
            q = rf.q_next.iloc[t].to_numpy(float)
            cut = np.nanquantile(np.where(u, q, np.nan), 1 - p["q"])
            w = np.where(u & (q < cut), sg_f ** -p["k"], 0.0)
        else:
            raise ValueError(kind)
        S.append(w / w.sum() if w.sum() > 0 else w)
        B.append(b)
    cols = prices.adj_close.columns
    return pd.DataFrame(S, index=idx, columns=cols), pd.DataFrame(B, index=idx, columns=cols)


def _maxdd(r) -> float:
    eq = np.cumprod(1 + np.asarray(r, float))
    return float((eq / np.maximum.accumulate(eq) - 1).min())


def run_portfolio(tr: Trial, ctx: Ctx) -> tuple[dict, pd.Series]:
    prices = ctx.prices
    S, B = portfolio_weights(tr, ctx)

    def go(cost=10.0, delay=1):
        rs = gates.window(engine.run_weights(S, prices, cost, delay)["ret"], START, END)
        rb = gates.window(engine.run_weights(B, prices, cost, delay)["ret"], START, END)
        d = rs - (rs.std(ddof=1) / rb.std(ddof=1)) * rb
        return rs, rb, d

    rs, rb, d = go()
    st = gates.variant_stats(d, START, END)
    sh = lambda x: float(x.mean() / x.std(ddof=1) * np.sqrt(252))
    rec = {"mean": st["mean_daily"], "se": st["se_daily"], "t": st["t"],
           "sharpe_s": sh(rs), "sharpe_b": sh(rb), "ann_ret_s": float((1 + rs).prod() ** (252 / len(rs)) - 1),
           "ann_ret_b": float((1 + rb).prod() ** (252 / len(rb)) - 1), "vol_s": float(rs.std() * np.sqrt(252)),
           "vol_b": float(rb.std() * np.sqrt(252)), "maxdd_s": _maxdd(rs), "maxdd_b": _maxdd(rb),
           "sharpe_diff": F._sharpe_diff_boot(rs.to_numpy(), rb.to_numpy(), B=1000)}
    rec["mean_25bps"] = float(go(cost=25.0)[2].mean())
    rec["mean_delay_plus1"] = float(go(delay=2)[2].mean())
    return rec, d


# ---------------------------------------------------------------- timing rules

def staircase(M_pct: pd.Series, half_day: pd.Series, on: float, cut: float, off: float = 1.0,
              step: float = 0.25) -> pd.Series:
    e = 1.0
    out = []
    hd = half_day.reindex(M_pct.index).fillna(False).to_numpy(bool)
    for mp, h in zip(M_pct.to_numpy(float), hd):
        if not h and np.isfinite(mp):
            if mp >= on:
                e = cut
            elif mp < off and e < 1.0:
                e = min(1.0, e + step)
        out.append(e)
    return pd.Series(out, index=M_pct.index)


def fear_basket_vol(ctx: Ctx) -> pd.Series:
    """Equal-weight basket daily vol from the fear yardstick and equicorrelation."""
    st5 = ctx.stilde(5) / np.sqrt(5)
    out = np.full(len(st5), np.nan)
    for t in range(len(st5)):
        s = st5.iloc[t].to_numpy(float)
        s = s[np.isfinite(s)]
        if len(s) < 10:
            continue
        rho = min(max(ctx.rho(t), 0.0), 0.99)
        out[t] = np.sqrt((1 - rho) * np.mean(s ** 2) / len(s) + rho * np.mean(s) ** 2)
    return pd.Series(out, index=st5.index)


def run_timing(tr: Trial, ctx: Ctx) -> tuple[dict, pd.Series]:
    rf, p = ctx.rf, tr.params
    if tr.family == "V1":
        E = F.rf_weights(p["variant"], rf.M_pct, F.har_forecast(ctx.legs["cc"]), rf.half_day)
    elif p["rule"] == "staircase":
        E = staircase(rf.M_pct.reindex(ctx.legs.index), rf.half_day, p["on"], p["cut"])
    elif p["rule"] == "voltarget_fear":
        vb = fear_basket_vol(ctx)
        star = vb.expanding().median()
        E = (star / vb).clip(0.25, 1.0).ffill().fillna(1.0)
    else:
        raise ValueError(p)
    b = F.rf_stats(E, ctx.legs, START, END)
    c25 = F.rf_stats(E, ctx.legs, START, END, cost_bps=25.0)
    d2 = F.rf_stats(E, ctx.legs, START, END, delay=2)
    rec = {"mean": b["mean_d"], "se": b["se_d"], "t": b["t_d"], "var_ratio": b["var_ratio"],
           "cvar95_net": b["cvar95_net"], "cvar95_static": b["cvar95_static"], "maxdd_net": b["maxdd_net"],
           "maxdd_static": b["maxdd_static"], "E_bar": b["E_bar"],
           "mean_25bps": c25["mean_d"], "mean_delay_plus1": d2["mean_d"]}
    return rec, b["d"]


# ---------------------------------------------------------------- registry

def registered() -> set[str]:
    if not TRIALS.exists():
        return set()
    return {json.loads(l)["id"] for l in TRIALS.read_text().splitlines() if l.strip()}


def register(tr: Trial, rec: dict, x: pd.Series) -> None:
    REG.mkdir(parents=True, exist_ok=True)
    row = {"id": tr.id, "family": tr.family, "kind": tr.kind, "params": tr.params, "round": tr.round,
           **{k: v for k, v in rec.items()}}
    with open(TRIALS, "a") as fh:
        fh.write(json.dumps(protocol.jsonable(row)) + "\n")
    col = pd.DataFrame({tr.id: x.astype(float)})
    if SERIES.exists():
        cur = pd.read_parquet(SERIES)
        cur[tr.id] = col[tr.id].reindex(cur.index)
        cur.to_parquet(SERIES)
    else:
        col.to_parquet(SERIES)


def run_ensemble(tr: Trial, ctx: Ctx) -> tuple[dict, pd.Series]:
    """An equal split of capital across member event trials: the daily series is the
    members' mean, at base settings, at 25 bps and with a one-day delay."""
    members = [Trial(m["family"], "event", m["params"]) for m in tr.params["members"]]
    rf, prices = ctx.rf, ctx.prices
    base, c25, dl = [], [], []
    for m in members:
        mask, prio, H, cap, side, delay = event_rule(m, ctx)
        sig = signals.to_signals(signals.suppress_reentry(mask, H, priority=prio, cap=cap))
        for out, (cost, dly) in ((base, (10.0, delay)), (c25, (25.0, delay)), (dl, (10.0, delay + 1))):
            last = gates.last_signal_date(prices.dates, H, dly, END)
            res = engine.run_cohorts(sig, prices, H=H, cost_bps=cost, delay=dly, side=side, start=START,
                                     end=last, cache=ctx.cache, beta=rf.beta)
            out.append(gates.window(res.daily["excess_net"], START, END))
    x = pd.concat(base, axis=1).mean(axis=1)
    st = gates.variant_stats(x, START, END)
    rec = {"mean": st["mean_daily"], "se": st["se_daily"], "t": st["t"], "n_members": len(members),
           "mean_25bps": float(pd.concat(c25, axis=1).mean(axis=1).mean()),
           "mean_delay_plus1": float(pd.concat(dl, axis=1).mean(axis=1).mean())}
    return rec, x


def run(trials: list[Trial], ctx: Ctx, verbose: bool = True) -> list[dict]:
    """Run every trial not yet in the registry, append it, and return the new records."""
    done = registered()
    out = []
    for tr in trials:
        if tr.id in done:
            continue
        fn = {"event": run_event, "portfolio": run_portfolio, "timing": run_timing,
              "ensemble": run_ensemble}[tr.kind]
        rec, x = fn(tr, ctx)
        register(tr, rec, x)
        done.add(tr.id)
        out.append({"id": tr.id, **rec})
        if verbose:
            print(f"{tr.id:70s} t {rec['t']:+.2f}", flush=True)
    return out


def load_registry() -> tuple[pd.DataFrame, pd.DataFrame]:
    recs = [json.loads(l) for l in TRIALS.read_text().splitlines() if l.strip()]
    return pd.DataFrame(recs), pd.read_parquet(SERIES)


# --------------------------------------------------------- the declared spaces

def v1_trials() -> list[Trial]:
    out = [Trial("V1", "event", {"variant": v.name}, 0) for fam in F.HOLM_FAMILIES for v in fam.variants]
    out += [Trial("V1", "timing", {"variant": n}, 0) for n in F.RF_VARIANTS]
    return out


def round1_trials(with_earnings: bool) -> list[Trial]:
    T = []
    for k in (1, 2, 4):
        T.append(Trial("P", "portfolio", {"w": "invvol_fear", "k": k}))
    for k in (1, 2):
        T.append(Trial("P", "portfolio", {"w": "invvol_ewma", "k": k}))
    for q in (0.10, 0.20):
        T.append(Trial("P", "portfolio", {"w": "ex_qnext", "q": q}))
    T.append(Trial("P", "portfolio", {"w": "minvar_fear"}))
    T.append(Trial("P", "portfolio", {"w": "invvol_fear_ex_qnext", "k": 1, "q": 0.10}))
    for on in (1.0, 1.5, 2.0):
        for cut in (0.0, 0.5):
            T.append(Trial("T", "timing", {"rule": "staircase", "on": on, "cut": cut}))
    T.append(Trial("T", "timing", {"rule": "voltarget_fear"}))
    for day in ("calm", "any"):
        for x in (2, 3):
            for g in (-1.5, -2.5):
                for H in (5, 10, 20, 40):
                    T.append(Trial("E1", "event", {"day": day, "x": x, "g": g, "H": H}))
    for g in (1.5, 2.5):
        for rv in (1.0, 2.0):
            for H in (5, 10, 20, 40):
                T.append(Trial("E2", "event", {"g": g, "rv": rv, "H": H}))
    for delay in (1, 2):
        for v in (-1.5, -2.5):
            for m in (None, 1.0):
                for H in (3, 5, 10):
                    T.append(Trial("E3", "event", {"delay": delay, "v": v, "m": m, "H": H}))
    if with_earnings:
        for side in ("good", "bad"):
            for thr in (1.5, 3.0):
                for H in (5, 20, 40):
                    T.append(Trial("E4", "event", {"side": side, "thr": thr, "H": H}))
    for signal in ("Z", "I"):
        for end in ("low", "high"):
            for H in (5, 20):
                T.append(Trial("E5", "event", {"signal": signal, "end": end, "H": H}))
    return T


def round2_trials() -> list[Trial]:
    """Adaptive follow-ups to Round 1 (LOG.md). Logged before their results were viewed."""
    T = []
    for g in (1.5, 2.5):                         # E2 faded: the sign flip is a new set of trials
        for rv in (1.0, 2.0):
            for H in (5, 10, 20, 40):
                T.append(Trial("E2F", "event", {"g": g, "rv": rv, "H": H}, 2))
    for H in (5, 10, 20, 40):                     # either-sign idiosyncratic shock
        T.append(Trial("E6", "event", {"H": H}, 2))
    for fam in ("MAXA", "IVOLA"):
        T.append(Trial(fam, "event", {"H": 20}, 2))
    for v in (-3.0, -3.5):                        # more displaced systemic reversal
        for H in (10, 20):
            T.append(Trial("E3", "event", {"delay": 1, "v": v, "m": None, "H": H}, 2))
    e1 = [{"family": "E1", "params": {"day": d, "x": x, "g": g, "H": H}}
          for d in ("calm", "any") for x in (2, 3) for g in (-1.5, -2.5) for H in (5, 10, 20, 40)]
    e2f = [{"family": "E2F", "params": {"g": g, "rv": rv, "H": H}}
           for g in (1.5, 2.5) for rv in (1.0, 2.0) for H in (5, 10, 20, 40)]
    e6 = [{"family": "E6", "params": {"H": H}} for H in (5, 10, 20, 40)]
    T.append(Trial("ENS", "ensemble", {"name": "E1_all", "members": e1}, 2))
    T.append(Trial("ENS", "ensemble", {"name": "E2F_all", "members": e2f}, 2))
    T.append(Trial("ENS", "ensemble", {"name": "E6_all", "members": e6}, 2))
    T.append(Trial("ENS", "ensemble", {"name": "E1_E2F_all", "members": e1 + e2f}, 2))
    for rule in ("down", "up", "both"):           # long-only expression: the basket minus shocked names
        T.append(Trial("P", "portfolio", {"w": "clean_basket", "rule": rule, "H": 20}, 2))
    return T


def round3_trials() -> list[Trial]:
    """Round 3: the post-news-underperformance theme (LOG.md), and its long-only expression."""
    T = []
    for rule in ("earn1.5", "earn3.0"):
        for H in (20, 40):
            T.append(Trial("P", "portfolio", {"w": "clean_basket", "rule": rule, "H": H}, 3))
    for H in (20, 40):
        T.append(Trial("P", "portfolio", {"w": "clean_basket", "rule": "union1.5", "H": H}, 3))
    for H in (10, 40):
        T.append(Trial("P", "portfolio", {"w": "clean_basket", "rule": "both", "H": H}, 3))
    for thr in (1.5, 3.0):
        for H in (20, 40):
            T.append(Trial("E7", "event", {"thr": thr, "H": H}, 3))
    e1 = [{"family": "E1", "params": {"day": d, "x": x, "g": g, "H": H}}
          for d in ("calm", "any") for x in (2, 3) for g in (-1.5, -2.5) for H in (5, 10, 20, 40)]
    e2f = [{"family": "E2F", "params": {"g": g, "rv": rv, "H": H}}
           for g in (1.5, 2.5) for rv in (1.0, 2.0) for H in (5, 10, 20, 40)]
    e4b = [{"family": "E4", "params": {"side": "bad", "thr": thr, "H": H}} for thr in (1.5, 3.0) for H in (5, 20, 40)]
    e6 = [{"family": "E6", "params": {"H": H}} for H in (5, 10, 20, 40)]
    e7 = [{"family": "E7", "params": {"thr": thr, "H": H}} for thr in (1.5, 3.0) for H in (20, 40)]
    T.append(Trial("ENS", "ensemble", {"name": "news_theme", "members": e1 + e2f + e4b + e6}, 3))
    T.append(Trial("ENS", "ensemble", {"name": "news_theme_with_E7", "members": e1 + e2f + e4b + e6 + e7}, 3))
    return T


def round4_trials() -> list[Trial]:
    """Round 4: does model 1 add value to the news rule (E7)? Plus two dose checks."""
    T = []
    for H in (20, 40):
        for mc in ("hi", "lo"):
            T.append(Trial("E7", "event", {"thr": 3.0, "H": H, "m_cond": mc}, 4))
    for zc in ("hi", "lo"):
        T.append(Trial("E7", "event", {"thr": 3.0, "H": 40, "z_cond": zc}, 4))
    for H in (20, 40):
        T.append(Trial("E7", "event", {"thr": 3.0, "H": H, "weight": "invvol"}, 4))
    for H in (20, 40):
        T.append(Trial("E7", "event", {"thr": 4.0, "H": H}, 4))
    T.append(Trial("E7", "event", {"thr": 3.0, "H": 60}, 4))
    T.append(Trial("P", "portfolio", {"w": "clean_basket", "rule": "earn3.0", "H": 60}, 4))
    return T


def round5_trials() -> list[Trial]:
    """Round 5: new earnings information. EPS surprise (E8), reaction against surprise (E9),
    and the pre-announcement premium (E10). E10 uses report dates that are scheduled in
    advance: a small look-ahead risk for dates confirmed late, disclosed in LOG.md."""
    T = []
    for side in ("good", "bad"):
        for thr in (0.10, 0.25):
            for H in (20, 40):
                T.append(Trial("E8", "event", {"side": side, "thr": thr, "H": H}, 5))
    for case in ("beat_drop", "miss_pop"):
        for H in (20, 40):
            T.append(Trial("E9", "event", {"case": case, "H": H}, 5))
    T.append(Trial("E10", "event", {"H": 5, "through": True}, 5))     # exit at the reaction session's open
    T.append(Trial("E10", "event", {"H": 4, "through": False}, 5))    # exit the session before it
    return T
