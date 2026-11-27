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
    """Same-day dollar volume over its median of the previous ``window`` sessions."""
    v = prices.dollar_volume
    return v / v.rolling(window, min_periods=15).median().shift(1)
