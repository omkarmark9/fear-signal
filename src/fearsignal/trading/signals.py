"""Point-in-time fear inputs on the price grid, shared by every family.

``Inputs.wide(field)`` gives a dates x tickers frame aligned with the price
frames. Every value at row t was computable at the close of t.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import engine, pit as pit_mod

STOCK_FIELDS = ("x", "Z", "I", "q_next", "P_calm", "P_stressed", "P_panic", "sigma", "r")


@dataclass
class Inputs:
    prices: engine.Prices
    stock: dict           # field -> dates x tickers DataFrame
    market: pd.DataFrame  # by date: M, M_pct, breadth, share_hit2, systemic, n_stocks

    def wide(self, field: str) -> pd.DataFrame:
        return self.stock[field]

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.prices.dates


def build(p: pit_mod.PIT | None = None, prices: engine.Prices | None = None) -> Inputs:
    p = p if p is not None else pit_mod.load()
    prices = prices if prices is not None else engine.Prices.load()
    idx, cols = prices.dates, prices.adj_close.columns
    st = p.stock.copy()
    st["date"] = pd.to_datetime(st["date"])
    stock = {f: st.pivot(index="date", columns="ticker", values=f).reindex(index=idx, columns=cols)
             for f in STOCK_FIELDS}
    mk = p.market.copy()
    mk.index = pd.to_datetime(mk.index)
    return Inputs(prices, stock, mk.reindex(idx))


def to_signals(mask: pd.DataFrame, weight: pd.DataFrame | None = None) -> pd.DataFrame:
    """Long table (date, ticker, weight) from a boolean dates x tickers mask."""
    m = mask.fillna(False).astype(bool)
    tt, nn = np.nonzero(m.to_numpy())
    out = pd.DataFrame({"date": m.index[tt], "ticker": m.columns[nn]})
    out["weight"] = 1.0 if weight is None else weight.reindex_like(m).to_numpy(float)[tt, nn]
    return out


def top_k(score: pd.DataFrame, k: int, eligible: pd.DataFrame | None = None) -> pd.DataFrame:
    """Boolean mask of the k highest scores per row (ties broken by column order)."""
    s = score.where(eligible) if eligible is not None else score
    r = s.rank(axis=1, ascending=False, method="first")
    return r <= k


def from_live(mk: pd.DataFrame, st: pd.DataFrame, feat, prices: engine.Prices | None = None) -> Inputs:
    """Inputs from the frozen v3 fear model's live state (pipeline._live_state).

    Live tickets use model 1 as frozen, the model the user sees. The backtest
    uses quarterly point-in-time refits of the same structure. The two agree
    closely after 2024 (trading/pit.py), and the live report shows both.
    """
    from .. import features

    prices = prices if prices is not None else engine.Prices.load()
    idx, cols = prices.dates, prices.adj_close.columns
    obs = feat.obs
    d = obs.dates
    tk = list(obs.tickers)
    extra = {"sigma": pd.DataFrame(feat.sigma, index=d, columns=tk),
             "r": pd.DataFrame(feat.r, index=d, columns=tk)}
    s = st.copy()
    s["date"] = pd.to_datetime(s["date"])
    stock = {}
    for f in STOCK_FIELDS:
        w = extra[f] if f in extra else s.pivot(index="date", columns="ticker", values=f)
        stock[f] = w.reindex(index=idx, columns=cols)
    el = obs.eligible
    hit2 = el & (np.nan_to_num(obs.x) >= features.SYSTEMIC_X)
    share = pd.Series(hit2.sum(axis=1) / np.maximum(el.sum(axis=1), 1), index=d)
    m = mk.copy()
    m.index = pd.to_datetime(m.index)
    m = m[["M", "M_pct", "breadth", "n_stocks"]].join(share.rename("share_hit2"), how="left")
    m["systemic"] = m["share_hit2"] >= features.SYSTEMIC_SHARE
    return Inputs(prices, stock, m.reindex(idx))


# ------------------------------------------------------------ price features
# All use data up to and including the close of t, never later.

def log_returns(prices: engine.Prices) -> pd.DataFrame:
    ac = prices.adj_close
    return np.log(ac / ac.shift(1))


def basket_log_return(prices: engine.Prices) -> pd.Series:
    """Equal-weight mean of the names' log returns (a basket proxy for betas)."""
    return log_returns(prices).mean(axis=1)


def trailing_beta(prices: engine.Prices, window: int = 126, min_periods: int = 100,
                  shrink: float = 1 / 3) -> pd.DataFrame:
    """Beta on the basket over the ``window`` sessions ending at t-1, shrunk toward 1
    (Vasicek-style: (1 - shrink) x OLS + shrink x 1)."""
    r = log_returns(prices)
    b = basket_log_return(prices)
    cov = r.rolling(window, min_periods=min_periods).cov(b)
    var = b.rolling(window, min_periods=min_periods).var()
    beta = cov.div(var, axis=0).shift(1)
    return (1 - shrink) * beta + shrink


def residual_sigma(prices: engine.Prices, sigma: pd.DataFrame) -> pd.DataFrame:
    """Same-day beta residual in units of the stock's own robust sigma (from the fear model)."""
    r = log_returns(prices)
    b = basket_log_return(prices)
    return (r - trailing_beta(prices).mul(b, axis=0)) / sigma


def gap_sigma(prices: engine.Prices, sigma: pd.DataFrame) -> pd.DataFrame:
    """Overnight gap, log(adj open / previous adj close), in own sigma."""
    return np.log(prices.adj_open / prices.adj_close.shift(1)) / sigma


def relative_volume(prices: engine.Prices, window: int = 20) -> pd.DataFrame:
    """Same-day share volume over its median of the previous ``window`` sessions
    (dollar volume if share volume is not loaded)."""
    v = prices.volume if prices.volume is not None else prices.dollar_volume
    return v / v.rolling(window, min_periods=15).median().shift(1)


# ------------------------------------------------------------ rule helpers

def half_days() -> pd.DatetimeIndex:
    """NYSE early-close sessions, 2023-2026 (the fear model reads them as quiet)."""
    from ..forecast import HALF_DAYS
    return pd.DatetimeIndex(HALF_DAYS)


def segments(columns) -> pd.Series:
    from .. import universe
    return universe.table()["segment"].reindex(list(columns))


def segment_demean(frame: pd.DataFrame, eligible: pd.DataFrame, min_n: int = 3) -> pd.DataFrame:
    """frame minus the median of eligible names in the same segment that day.
    Segments with fewer than ``min_n`` eligible names that day use the median
    of all eligible names instead."""
    f = frame.where(eligible)
    seg = segments(frame.columns)
    all_med = f.median(axis=1)
    out = pd.DataFrame(np.nan, index=frame.index, columns=frame.columns)
    for s in seg.unique():
        cols = seg.index[seg == s]
        sub = f[cols]
        n = sub.notna().sum(axis=1)
        med = sub.median(axis=1).where(n >= min_n, all_med)
        out[cols] = frame[cols].sub(med, axis=0)
    return out


def suppress_reentry(mask: pd.DataFrame, H: int, priority: pd.DataFrame | None = None,
                     cap: int | None = None) -> pd.DataFrame:
    """Walk forward through signal days. A name selected on day t is held for
    the next H-1 signal days and cannot be selected again until day t+H, when
    the old position is sold at the same open the new one would be bought. If
    ``cap`` is set, at most ``cap`` new names per day are kept, taken in
    ``priority`` order (smallest first)."""
    m = mask.fillna(False).to_numpy(bool).copy()
    T, N = m.shape
    held_until = np.full(N, -1)
    pr = priority.to_numpy(float) if priority is not None else None
    for t in range(T):
        cand = np.flatnonzero(m[t] & (held_until < t))
        m[t] = False
        if cand.size == 0:
            continue
        if cap is not None and cand.size > cap:
            order = np.argsort(pr[t, cand], kind="stable") if pr is not None else np.arange(cand.size)
            cand = cand[order[:cap]]
        m[t, cand] = True
        held_until[cand] = t + H - 1
    return pd.DataFrame(m, index=mask.index, columns=mask.columns)


# ------------------------------------------------------- shared rule features

@dataclass
class RuleFeatures:
    """Everything the declared rules read, each value known at the close of t.

    r, sigma, x and eligibility are deterministic transforms of prices (the
    fear model's features), computed over the full history from 2023, so
    betas have their 126 sessions. Z, q_next and the regime probabilities
    come from the point-in-time refits (or frozen v3, live).
    """
    eligible: pd.DataFrame
    x: pd.DataFrame
    Z: pd.DataFrame
    q_next: pd.DataFrame
    P_panic: pd.DataFrame
    sigma: pd.DataFrame
    r: pd.DataFrame
    rB: pd.Series
    beta: pd.DataFrame
    u: pd.DataFrame          # (r - beta rB) / sigma
    v: pd.DataFrame          # u minus its segment median
    gamma: pd.DataFrame      # overnight gap / sigma
    g_seg: pd.DataFrame      # gamma minus its segment median
    g_all: pd.DataFrame      # gamma minus its all-stock median
    relvol: pd.DataFrame     # share volume / median of the previous 20 sessions
    price_ok: pd.DataFrame   # raw close >= $5
    dv_med20: pd.DataFrame   # median dollar volume (raw close x volume) over t-20..t-1 (>= 15 values)
    liquid20: pd.DataFrame   # dv_med20 >= $20M
    systemic: pd.Series
    share_hit2: pd.Series
    half_day: pd.Series
    M_pct: pd.Series
    P_calm: pd.DataFrame | None = None       # for sizing's regime forecast; not used by any rule
    P_stressed: pd.DataFrame | None = None


def fear_features(panel=None, live_path=None):
    """The fear model's deterministic features (r, sigma, x, eligibility) on the live panel.

    Backtests pass the frozen snapshot of the live bars (protocol.FROZEN_LIVE), so a
    routine `fearsignal live` refresh can never move a locked input."""
    from .. import data, features
    panel = panel if panel is not None else data.load_live_panel("yahoo", live_path)
    return features.compute(panel)


def rule_features(inp: Inputs, feat=None, beta_window: int = 126, beta_min: int = 100) -> RuleFeatures:
    from .. import features as fmod
    feat = feat if feat is not None else fear_features()
    idx, cols = inp.dates, inp.prices.adj_close.columns
    obs = feat.obs

    def frame(a):
        return pd.DataFrame(a, index=obs.dates, columns=list(obs.tickers)).reindex(index=idx, columns=cols)

    el = frame(obs.eligible.astype(float)).fillna(0.0).astype(bool)
    x = frame(obs.x).where(el)
    sigma = frame(feat.sigma).where(el)
    r = frame(feat.r)
    rB = r.where(el).mean(axis=1)
    RB = pd.DataFrame(np.repeat(rB.to_numpy()[:, None], len(cols), axis=1), index=idx, columns=cols).where(r.notna())
    cov = r.rolling(beta_window, min_periods=beta_min).cov(RB)
    var = RB.rolling(beta_window, min_periods=beta_min).var()
    beta = (2 / 3) * (cov / var).shift(1) + 1 / 3
    u = (r - beta.mul(rB, axis=0)) / sigma
    gamma = np.log(inp.prices.adj_open / inp.prices.adj_close.shift(1)) / sigma
    v = segment_demean(u, el & u.notna())
    g_seg = segment_demean(gamma, el & gamma.notna())
    g_all = gamma.sub(gamma.where(el).median(axis=1), axis=0)
    close = inp.prices.close if inp.prices.close is not None else inp.prices.adj_close
    dv = inp.prices.dollar_volume
    dv_med20 = dv.rolling(20, min_periods=15).median().shift(1)
    liquid20 = dv_med20 >= 20e6
    hit2 = el & (x >= fmod.SYSTEMIC_X)
    share = hit2.sum(axis=1) / el.sum(axis=1).replace(0, np.nan)
    hd = pd.Series(idx.isin(half_days()), index=idx)
    return RuleFeatures(
        eligible=el, x=x, Z=inp.wide("Z").where(el), q_next=inp.wide("q_next").where(el),
        P_panic=inp.wide("P_panic").where(el), sigma=sigma, r=r, rB=rB, beta=beta, u=u, v=v,
        gamma=gamma, g_seg=g_seg, g_all=g_all, relvol=relative_volume(inp.prices),
        price_ok=close >= 5.0, dv_med20=dv_med20, liquid20=liquid20.fillna(False),
        systemic=(share >= fmod.SYSTEMIC_SHARE).fillna(False), share_hit2=share,
        half_day=hd, M_pct=inp.market["M_pct"],
        P_calm=inp.wide("P_calm").where(el), P_stressed=inp.wide("P_stressed").where(el))
