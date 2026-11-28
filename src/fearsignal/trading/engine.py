"""Backtest engine for the trading model.

Execution model (fixed before any strategy was tested):

* A signal is computed at the close of session t from point-in-time fear
  outputs and prices up to t. It is executed at the OPEN of t+delay
  (``delay=1``; ``delay=2`` is the robustness check).
* A signal day opens a COHORT: the names it selects, weighted at entry
  (equal unless the signal gives weights), bought and held for H sessions,
  sold at the open H sessions after entry. Cohorts overlap. Each of the last
  H cohorts holds 1/H of the sleeve's capital (Jegadeesh and Titman 1993),
  and a day with no signal leaves its 1/H in cash.
* Costs: ``cost_bps`` per side on every entry and exit.
* Tradability at the signal requires a close at t, an open at entry, and a
  20-session median dollar volume of at least $5M as of t (no future data).
* Benchmark: the equal-weight basket of every tradable name, bought at the
  same open and sold at the same open. A cohort's excess return is its
  return minus that basket cohort's return. That is the stock-selection
  edge, net of the AI-boom tailwind every long position in this hindsight-
  chosen universe enjoys.
* Bearish families (``side="short_info"``) are not traded: the account
  cannot short. Their excess is basket minus names, after the same costs, as
  the information a put buyer would be paying for.

Accounting is exact for buy-and-hold cohorts. On the entry day a name earns
open-to-close, then close-to-close, and on the exit day close-to-open. A
cohort's daily return is the change in the weighted average of its names'
value paths.

The exposure engine (``run_exposure``) scales a daily-rebalanced position in
the equal-weight basket. It uses the same timing: decided at close t, traded
at the open of t+delay, with each day split into overnight and intraday parts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

MIN_DOLLAR_VOLUME = 5e6
ANN = 252


@dataclass
class Prices:
    adj_open: pd.DataFrame
    adj_close: pd.DataFrame
    dollar_volume: pd.DataFrame
    close: pd.DataFrame | None = None       # raw (unadjusted) close, for price floors and options
    volume: pd.DataFrame | None = None      # share volume

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.adj_close.index

    def liquid(self) -> pd.DataFrame:
        med = self.dollar_volume.rolling(20, min_periods=10).median()
        return med >= MIN_DOLLAR_VOLUME

    @classmethod
    def load(cls, live: bool = False) -> "Prices":
        """The declared backtest prices; ``live=True`` reads the live refresh if one exists."""
        from .market_data import LIVE_PATH, load
        f = load(LIVE_PATH if live and LIVE_PATH.exists() else None)
        return cls(f["adj_open"], f["adj_close"], f["dollar_volume"], f["close"], f["volume"])


@dataclass
class Result:
    daily: pd.DataFrame      # gross, net, benchmark, excess_net, n_cohorts_active (sleeve-level)
    trades: pd.DataFrame     # one row per (cohort, name)
    cohorts: pd.DataFrame    # one row per cohort
    params: dict = field(default_factory=dict)


def _value_path(ao: np.ndarray, ac: np.ndarray, e: int, x: int) -> np.ndarray:
    """Value of $1 bought at the open of e, at the closes of e..x-1 and the open of x.

    Length x - e + 1. A missing close carries the last value forward (a halted
    or delisted name is held at its last price), and a missing exit open falls
    back to the last value.
    """
    v = np.empty(x - e + 1)
    entry = ao[e]
    last = entry
    for k, d in enumerate(range(e, x)):
        c = ac[d]
        if np.isfinite(c):
            last = c
        v[k] = last / entry
    o = ao[x] if np.isfinite(ao[x]) else last
    v[-1] = o / entry
    return v


def _paths(ao: np.ndarray, ac: np.ndarray, e: int, x: int, J: np.ndarray) -> np.ndarray:
    """``_value_path`` for several columns at once: an (len(J), x - e + 1) array."""
    entry = ao[e, J]
    c = ac[e:x][:, J]
    filled = pd.DataFrame(c).ffill().to_numpy()
    filled = np.where(np.isfinite(filled), filled, entry[None, :])
    o = ao[x, J]
    o = np.where(np.isfinite(o), o, filled[-1])
    return (np.vstack([filled, o[None, :]]) / entry[None, :]).T


def _half_paths(ao: np.ndarray, ac: np.ndarray, e: int, x: int, J: np.ndarray) -> np.ndarray:
    """Values of $1 bought at the open of e, at every open AND close of the hold:
    O_e, C_e, O_e+1, C_e+1, ..., O_x-1, C_x-1, O_x. Shape (len(J), 2(x-e)+1).
    A missing price carries the last value forward (a halt), as in ``_paths``."""
    H = x - e
    seq = np.empty((len(J), 2 * H + 1))
    seq[:, 0::2] = ao[e:x + 1][:, J].T
    seq[:, 1::2] = ac[e:x][:, J].T
    filled = pd.DataFrame(seq.T).ffill().to_numpy().T
    return filled / filled[:, :1]


def tradable(prices: Prices) -> np.ndarray:
    """T x N: has a close and is liquid at t (both known at the close of t)."""
    return np.isfinite(prices.adj_close.to_numpy(float)) & prices.liquid().to_numpy(bool)


def run_cohorts(signals: pd.DataFrame, prices: Prices, H: int, cost_bps: float = 10.0,
                delay: int = 1, side: str = "long", start=None, end=None,
                cache: dict | None = None, beta: pd.DataFrame | None = None) -> Result:
    """Backtest cohort signals.

    signals: rows (date, ticker[, weight]); ``date`` is the signal session t.
    Entry is the open of t+delay and exit the open of t+delay+H. A cohort runs
    only if its exit session exists and its signal date is within [start, end].
    ``cache`` (a dict, shared across calls on the same prices) memoises the
    basket cohort, which does not depend on the signal.

    Two daily excess series are returned. ``excess_net`` gives each of the
    last H cohorts 1/H of the capital (Jegadeesh-Titman). ``ct_excess_net``
    is the calendar-time portfolio (Fama 1998), fully invested in the open
    positions at equal weight. Each session is two legs:
    - the overnight leg, close d-1 to open d, averages over the positions held
      at the close of d-1, net of exit costs for those sold at d's open;
    - the intraday leg, open d to close d, averages over the positions held
      after d's open, net of entry costs for those bought then.
    The two legs compound. A leg with no position is 0 (cash).

    ``beta`` (dates x tickers, known at t) makes the benchmark beta x basket.
    Each name's event excess is then R_i - beta_i,t R_B. On a day the
    cohort's benchmark return is the value-weighted mean beta of its names
    times the basket's return. Without it, beta is 1.
    """
    if side not in ("long", "short_info"):
        raise ValueError(side)
    dates = prices.dates
    ao = prices.adj_open.to_numpy(float)
    ac = prices.adj_close.to_numpy(float)
    ok_t = tradable(prices)
    cols = prices.adj_close.columns
    tick_ix = {t: j for j, t in enumerate(cols)}
    T = len(dates)
    pos = {d: i for i, d in enumerate(dates)}
    sig = signals.copy()
    sig["date"] = pd.to_datetime(sig["date"])
    if start is not None:
        sig = sig[sig["date"] >= pd.Timestamp(start)]
    if end is not None:
        sig = sig[sig["date"] <= pd.Timestamp(end)]
    if "weight" not in sig:
        sig = sig.assign(weight=1.0)
    c = cost_bps / 1e4
    sgn = 1.0 if side == "long" else -1.0
    cols_out = {k: np.zeros(T) for k in ("gross", "net", "benchmark", "excess_net", "n_cohorts_active")}
    # calendar-time portfolio, per leg: overnight (close d-1 -> open d) over positions held at
    # the close of d-1, intraday (open d -> close d) over positions held after d's open
    on_sum, on_n, id_sum, id_n = np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T)
    BETA = beta.reindex(index=dates, columns=cols).to_numpy(float) if beta is not None else None
    trades, cohorts = [], []
    for d, grp in sig.groupby("date"):
        t = pos.get(d)
        if t is None:
            continue
        e, x = t + delay, t + delay + H
        if x >= T:
            continue                                     # exit not yet observable
        can = ok_t[t] & np.isfinite(ao[e])
        univ = np.flatnonzero(can)
        if univ.size == 0:
            continue
        names, w = [], []
        for tk, wt in zip(grp["ticker"], grp["weight"]):
            j = tick_ix.get(tk)
            if j is not None and can[j] and wt > 0:
                names.append(j)
                w.append(float(wt))
        if not names:
            continue
        w = np.asarray(w) / np.sum(w)
        key = (t, delay, H)
        if cache is not None and key in cache:
            bval = cache[key]
        else:
            bval = np.concatenate([[1.0], _paths(ao, ac, e, x, univ).mean(axis=0)])
            if cache is not None:
                cache[key] = bval
        paths = _paths(ao, ac, e, x, np.asarray(names))
        gval = np.concatenate([[1.0], w @ paths])
        nval = gval.copy()
        nval[1:] *= (1 - c)                              # entry cost, carried
        nval[-1] *= (1 - c)                              # exit cost
        rg = gval[1:] / gval[:-1] - 1
        rn = nval[1:] / nval[:-1] - 1
        rb = bval[1:] / bval[:-1] - 1
        if BETA is None:
            bj = np.ones(len(names))
            beta_d = np.ones_like(rb)
        else:
            bj = np.nan_to_num(BETA[t, names], nan=1.0)
            start_val = np.hstack([np.ones((len(names), 1)), paths[:, :-1]])   # value at the start of each day
            wd = w[:, None] * start_val
            beta_d = (wd * bj[:, None]).sum(axis=0) / wd.sum(axis=0)
        bench = beta_d * rb
        drag = rg - rn                                   # cost per day (entry and exit days only)
        ex = sgn * (rg - bench) - drag
        days = np.arange(e, x + 1)
        cols_out["gross"][days] += rg / H
        cols_out["net"][days] += rn / H
        cols_out["benchmark"][days] += bench / H
        cols_out["excess_net"][days] += ex / H
        cols_out["n_cohorts_active"][days] += 1.0
        # calendar-time accounting, per leg. Legs alternate: intraday of e, overnight of e+1,
        # intraday of e+1, ..., intraday of x-1, overnight of x. Each position is measured
        # against beta x its OWN cohort's buy-and-hold basket over the same leg.
        hkey = (t, delay, H, "half")
        if cache is not None and hkey in cache:
            vb = cache[hkey]
        else:
            vb = _half_paths(ao, ac, e, x, univ).mean(axis=0)
            if cache is not None:
                cache[hkey] = vb
        vp = _half_paths(ao, ac, e, x, np.asarray(names))
        lp = vp[:, 1:] / vp[:, :-1] - 1                             # (n, 2H) position legs
        lb = vb[1:] / vb[:-1] - 1                                   # (2H,) own-basket legs
        lex = sgn * (lp - bj[:, None] * lb[None, :])
        lex[:, 0] -= c * (1 + lp[:, 0])                             # bought at the open of e
        lex[:, -1] -= c * (1 + lp[:, -1])                           # sold at the open of x
        n_pos = len(names)
        id_days = e + np.arange(H)                                  # legs 0, 2, 4, ...
        on_days = e + 1 + np.arange(H)                              # legs 1, 3, 5, ...
        id_sum[id_days] += lex[:, 0::2].sum(axis=0)
        id_n[id_days] += n_pos
        on_sum[on_days] += lex[:, 1::2].sum(axis=0)
        on_n[on_days] += n_pos
        rt_cost = 1 - (1 - c) ** 2
        RB = bval[-1] - 1
        for j, wt, p, b in zip(names, w, paths, bj):
            trades.append({"signal_date": d, "ticker": cols[j], "weight": wt, "entry": dates[e],
                           "exit": dates[x], "ret": p[-1] - 1, "basket_ret": RB, "beta": b,
                           "excess_net": sgn * (p[-1] - 1 - b * RB) - rt_cost})
        cohorts.append({"signal_date": d, "entry": dates[e], "exit": dates[x], "n": len(names),
                        "ret_gross": gval[-1] - 1, "ret_net": nval[-1] - 1, "basket_ret": RB,
                        "beta": float(w @ bj),
                        "excess_net": sgn * (float(w @ (paths[:, -1] - 1 - bj * RB))) - rt_cost})
    daily = pd.DataFrame(cols_out, index=dates)
    on_leg = np.where(on_n > 0, on_sum / np.maximum(on_n, 1), 0.0)
    id_leg = np.where(id_n > 0, id_sum / np.maximum(id_n, 1), 0.0)
    daily["ct_excess_net"] = (1 + on_leg) * (1 + id_leg) - 1
    daily["ct_n_overnight"] = on_n
    daily["ct_n_intraday"] = id_n
    daily["ct_n_open"] = np.maximum(on_n, id_n)
    return Result(daily, pd.DataFrame(trades), pd.DataFrame(cohorts),
                  {"H": H, "cost_bps": cost_bps, "delay": delay, "side": side})


def basket_legs(prices: Prices) -> pd.DataFrame:
    """Equal-weight basket of names tradable at the previous close, split into
    overnight (close d-1 to open d) and intraday (open d to close d) returns.
    (1 + on)(1 + intraday) = 1 + the basket's close-to-close return. A session with
    no tradable name has no basket: cc is NaN there (on/intraday 0, i.e. cash), so a
    volatility model fitted on cc never sees a fabricated zero return."""
    ao = prices.adj_open.to_numpy(float)
    ac = prices.adj_close.to_numpy(float)
    ok = tradable(prices)
    T = len(prices.dates)
    on, intra, cc, n = np.zeros(T), np.zeros(T), np.full(T, np.nan), np.zeros(T)
    for d in range(1, T):
        m = ok[d - 1] & np.isfinite(ao[d]) & np.isfinite(ac[d])
        if not m.any():
            continue
        g_on = ao[d, m] / ac[d - 1, m]
        g_id = ac[d, m] / ao[d, m]
        on[d] = g_on.mean() - 1
        intra[d] = (g_on * g_id).sum() / g_on.sum() - 1
        cc[d] = (g_on * g_id).mean() - 1
        n[d] = m.sum()
    return pd.DataFrame({"on": on, "intraday": intra, "cc": cc, "n": n}, index=prices.dates)


def run_exposure(weight: pd.Series, legs: pd.DataFrame, cost_bps: float = 10.0,
                 delay: int = 1) -> pd.DataFrame:
    """A position of weight w in the basket, decided at close t and traded at
    the open of t+delay. On day d the overnight leg is held at the weight in
    force since the previous open, and the intraday leg at the weight traded at
    d's open. Costs are |change in weight| x cost_bps, charged at the open.
    Weights before the series starts are 0 (cash)."""
    w = weight.reindex(legs.index).ffill().fillna(0.0).to_numpy(float)
    T = len(w)
    w_id = np.concatenate([np.zeros(delay), w[:T - delay]])
    w_on = np.concatenate([[0.0], w_id[:-1]])
    cost = np.abs(w_id - w_on) * cost_bps / 1e4
    net = (1 + w_on * legs["on"].to_numpy()) * (1 + w_id * legs["intraday"].to_numpy()) - 1 - cost
    return pd.DataFrame({"w_overnight": w_on, "w_intraday": w_id, "cost": cost, "net": net,
                         "basket": legs["cc"].to_numpy()}, index=legs.index)


def summary(ret: pd.Series, start=None, end=None) -> dict:
    s = ret.copy()
    if start is not None:
        s = s[s.index >= pd.Timestamp(start)]
    if end is not None:
        s = s[s.index <= pd.Timestamp(end)]
    s = s.dropna()
    if len(s) < 2:
        return {}
    eq = (1 + s).cumprod()
    sd = s.std(ddof=1)
    return {"days": int(len(s)), "total_return": float(eq.iloc[-1] - 1),
            "ann_return": float(eq.iloc[-1] ** (ANN / len(s)) - 1),
            "ann_vol": float(sd * np.sqrt(ANN)),
            "sharpe": float(s.mean() / sd * np.sqrt(ANN)) if sd > 0 else 0.0,
            "max_drawdown": float((eq / eq.cummax() - 1).min()), "mean_daily": float(s.mean())}


def run_weights(target: pd.DataFrame, prices: Prices, cost_bps: float = 10.0, delay: int = 1) -> pd.DataFrame:
    """A long-only portfolio rebalanced to target weights.

    ``target`` has one row per decision date t (weights per ticker, summing to
    1; zero or NaN means not held). Each row is executed at the open of
    t+delay. Between rebalances the holdings drift with prices: overnight
    close->open, then intraday open->close. Costs are cost_bps x the traded
    notional at the open. Before the first execution the portfolio is cash.
    A name with a missing price is carried at its last value.

    Returns daily ``ret`` (close-to-close, net of costs), ``turnover`` (traded
    notional over the portfolio's value) and ``n_held``.
    """
    dates = prices.dates
    pos = {d: i for i, d in enumerate(dates)}
    ao = prices.adj_open.to_numpy(float)
    ac = prices.adj_close.to_numpy(float)
    cols = list(prices.adj_close.columns)
    W = target.reindex(columns=cols).fillna(0.0)
    execs = {}
    for d, row in W.iterrows():
        t = pos.get(pd.Timestamp(d))
        if t is not None and t + delay < len(dates):
            execs[t + delay] = row.to_numpy(float)
    T, N = ac.shape
    c = cost_bps / 1e4
    h = np.zeros(N)
    cash = 1.0
    v_prev = 1.0
    ret, turn, nheld = np.zeros(T), np.zeros(T), np.zeros(T)
    for d in range(T):
        if d > 0:
            g_on = ao[d] / ac[d - 1]
            h = h * np.where(np.isfinite(g_on), g_on, 1.0)
        if d in execs:
            w = execs[d]
            ok = np.isfinite(ao[d]) & (w > 0)
            w = np.where(ok, w, 0.0)
            if w.sum() > 0:
                w = w / w.sum()
                V = h.sum() + cash
                tgt = w * V
                traded = np.abs(tgt - h).sum()
                cost = c * traded
                h = w * (V - cost)
                cash = 0.0
                turn[d] = traded / V
        g_id = ac[d] / ao[d]
        h = h * np.where(np.isfinite(g_id), g_id, 1.0)
        v = h.sum() + cash
        ret[d] = v / v_prev - 1
        v_prev = v
        nheld[d] = int((h > 0).sum())
    return pd.DataFrame({"ret": ret, "turnover": turn, "n_held": nheld}, index=dates)
