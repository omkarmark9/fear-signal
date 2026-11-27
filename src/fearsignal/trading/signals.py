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
