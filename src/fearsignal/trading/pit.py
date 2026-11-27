"""Point-in-time fear signals for backtesting: quarterly walk-forward refits.

The frozen v3 model was estimated on data through 2026-11-23, so its readings
for 2024-2026 are in-sample: parameters, quantile maps and percentile scales
all saw the future. A backtest must instead use what a live system would have
had at each date. At the first session of every quarter from 2024-01-02, the
fear model (S1, the v3 structure) is refit on all data before that session.
Its quantile maps, linear market scale and zero rate are re-estimated on the
same window, and it is applied forward until the next refit. The filter runs
continuously from 2023 (it is causal), so each quarter's outputs use only
data up to each day and parameters estimated before the quarter began.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import data, features, model as mdl, scoring
from ..config import ERA_START
from ..data import RAW

CACHE = RAW / "pit_signals.parquet"
MARKET_CACHE = RAW / "pit_market.parquet"
START = pd.Timestamp("2024-01-02")


def refit_dates(dates: pd.DatetimeIndex, start=START) -> list[pd.Timestamp]:
    """First session of each calendar quarter, from ``start``."""
    d = dates[dates >= start]
    q = pd.Series(d, index=d).groupby(d.to_period("Q")).min()
    return list(q.values.astype("datetime64[ns]").astype(object))


@dataclass
class PIT:
    stock: pd.DataFrame     # long: date, ticker, x, Z, I, q_next, P_calm, P_stressed, P_panic, sigma, r, refit
    market: pd.DataFrame    # by date: M, M_pct, breadth, systemic, share_hit2, refit
    refits: list[dict]


def build(panel=None, n_random: int = 2, seed: int = 0) -> PIT:
    panel = panel if panel is not None else data.load_live_panel("yahoo")
    feat = features.compute(panel)
    obs = feat.obs
    dates = obs.dates
    T, N = obs.x.shape
    spec = next(c for c in mdl.CANDIDATES if c.name == "S1_v1_refit")
    fields = {k: np.full((T, N), np.nan) for k in ("Z", "I", "q_next", "P_calm", "P_stressed", "P_panic")}
    mk = {k: np.full(T, np.nan) for k in ("M", "M_pct", "breadth")}
    refit_of = np.full(T, np.datetime64("NaT"), dtype="datetime64[ns]")
    refits = []
    rd = refit_dates(dates)
    for i, d in enumerate(rd):
        d = pd.Timestamp(d)
        end = pd.Timestamp(rd[i + 1]) if i + 1 < len(rd) else dates[-1] + pd.Timedelta(days=1)
        train = obs.slice_time(dates[dates < d][-1])
        m = mdl.fit_stock_model(spec, train, n_random=n_random, seed=seed)
        z0 = mdl.zero_rate(train)
        base = mdl.compute_signal(m, train, None, None, z0)
        z_map = scoring.QuantileMap.fit(base.L[train.eligible])
        bl = base.breadth_L[np.isfinite(base.breadth_L)]
        m_map = scoring.QuantileMap.fit(bl, dz=0.02)
        scale = (float(bl.mean()), float(bl.std(ddof=1)))
        sig = mdl.compute_signal(m, obs, z_map, m_map, z0, scale)
        w = (dates >= d) & (dates < end)
        for k, arr in (("Z", sig.Z), ("I", sig.I), ("q_next", sig.q_next)):
            fields[k][w] = arr[w]
        for j, name in enumerate(("P_calm", "P_stressed", "P_panic")):
            fields[name][w] = sig.alpha[w][:, :, j]
        mk["M"][w], mk["M_pct"][w], mk["breadth"][w] = sig.M[w], sig.M_pct[w], sig.breadth[w]
        refit_of[w] = np.datetime64(d)
        refits.append({"refit": str(d.date()), "train_end": str(train.dates[-1].date()),
                       "n_obs": int(train.eligible.sum()), "converged": bool(m.converged),
                       "s": [float(v) for v in m.emission.s], "A_diag": [float(v) for v in np.diag(m.A)],
                       "market_scale": scale})
    el = obs.eligible
    hit2 = el & (np.nan_to_num(obs.x) >= features.SYSTEMIC_X)
    share = hit2.sum(axis=1) / np.maximum(el.sum(axis=1), 1)
    keep = ~np.isnat(refit_of)
    tt, nn = np.nonzero(el & keep[:, None])
    stock = pd.DataFrame({"date": dates[tt], "ticker": np.asarray(obs.tickers)[nn],
                          "x": obs.x[tt, nn], "sigma": feat.sigma[tt, nn], "r": feat.r[tt, nn],
                          **{k: v[tt, nn] for k, v in fields.items()},
                          "refit": refit_of[tt]})
    market = pd.DataFrame({**{k: v for k, v in mk.items()}, "share_hit2": share,
                           "systemic": share >= features.SYSTEMIC_SHARE, "n_stocks": el.sum(axis=1),
                           "refit": refit_of}, index=dates)[keep]
    market.index.name = "date"
    return PIT(stock, market, refits)


def save(p: PIT) -> None:
    p.stock.to_parquet(CACHE, index=False)
    p.market.to_parquet(MARKET_CACHE)
    (RAW / "pit_refits.json").write_text(json.dumps(p.refits, indent=1) + "\n")


def load() -> PIT:
    return PIT(pd.read_parquet(CACHE), pd.read_parquet(MARKET_CACHE),
               json.loads((RAW / "pit_refits.json").read_text()))
