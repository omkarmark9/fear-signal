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
    def load(cls) -> "Prices":
        from .market_data import load
        f = load()
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
    is the calendar-time portfolio (Fama 1998): each day, equal weight across
    every open position, and 0 on days with none.

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
    ct_sum, ct_n = np.zeros(T), np.zeros(T)          # calendar-time: sum and count of open positions
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
        # calendar-time accounting: every open position's own daily excess, equal weight per day
        pv = np.hstack([np.ones((len(names), 1)), paths])          # (n, H+2) value at entry open = 1
        pr = pv[:, 1:] / pv[:, :-1] - 1                             # (n, H+1) daily returns per position
        pcost = np.zeros_like(pr)
        pcost[:, 0] += c * (1 + pr[:, 0])                           # entry cost, as a return on the entry day
        pcost[:, -1] += c * (1 + pr[:, -1])                         # exit cost on the exit day
        pex = sgn * (pr - bj[:, None] * rb[None, :]) - pcost
        ct_sum[days] += pex.sum(axis=0)
        ct_n[days] += len(names)
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
    daily["ct_excess_net"] = np.where(ct_n > 0, ct_sum / np.maximum(ct_n, 1), 0.0)
    daily["ct_n_open"] = ct_n
    return Result(daily, pd.DataFrame(trades), pd.DataFrame(cohorts),
                  {"H": H, "cost_bps": cost_bps, "delay": delay, "side": side})


def basket_legs(prices: Prices) -> pd.DataFrame:
    """Equal-weight basket of names tradable at the previous close, split into
    overnight (close d-1 to open d) and intraday (open d to close d) returns.
    (1 + on)(1 + intraday) = 1 + the basket's close-to-close return."""
    ao = prices.adj_open.to_numpy(float)
    ac = prices.adj_close.to_numpy(float)
    ok = tradable(prices)
    T = len(prices.dates)
    on, intra, cc, n = np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T)
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
