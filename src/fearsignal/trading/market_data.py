"""Execution prices for the trading model: daily OHLCV for the AI universe.

The fear model reads only adjusted close, close and low. The trading model
also needs opens, because signals computed at the close of t are executed at
the open of t+1, and it uses volume to gauge liquidity. Bars come from the
same vendor as the fear model's history and are cached in the git-ignored
data/raw/trade_ohlcv.parquet. Adjusted open = open x adj_close / close, using
the same day's factor, so returns are on one basis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import universe
from ..data import RAW

PATH = RAW / "trade_ohlcv.parquet"            # the declared backtest input (hash-locked)
LIVE_PATH = RAW / "trade_ohlcv_live.parquet"  # live refreshes; never read by a backtest
COLS = ("open", "high", "low", "close", "adj_close", "volume")


def download(start: str = "2023-01-03", end_exclusive: str = "2026-11-28", live: bool = True) -> pd.DataFrame:
    """Fetch OHLCV. By default it writes the LIVE file; only an explicit live=False
    rewrites the hash-locked backtest input.

    Tickers the batch request misses are fetched again one by one, twice, as
    data.download does. A live pull is also archived per ticker (archive.py), so a
    name the vendor later stops serving keeps its bars."""
    import yfinance as yf

    tickers = universe.tickers()

    def fetch(batch: list[str]) -> dict[str, pd.DataFrame]:
        raw = yf.download(batch, start=start, end=end_exclusive, auto_adjust=False, actions=False,
                          progress=False, group_by="ticker", threads=True)
        got = {}
        for t in batch:
            if t not in raw.columns.get_level_values(0):
                continue
            sub = raw[t].rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close",
                                         "Adj Close": "adj_close", "Volume": "volume"})[list(COLS)]
            sub = sub.dropna(how="all")
            if not sub.empty:
                got[t] = sub
        return got

    bars = fetch(tickers)
    for _ in range(2):  # transient vendor failures: retry the stragglers one by one
        for t in [t for t in tickers if t not in bars]:
            bars.update(fetch([t]))
    frames = []
    for t in tickers:
        if t not in bars:
            continue
        sub = bars[t]
        sub.index = pd.DatetimeIndex(sub.index).tz_localize(None).normalize()
        sub = sub[~sub.index.duplicated(keep="last")]      # one bar per session (a duplicate would break the pivot)
        sub.index.name = "date"
        frames.append(sub.assign(ticker=t).reset_index())
    long = pd.concat(frames, ignore_index=True)
    long.to_parquet(LIVE_PATH if live else PATH, index=False)
    if live:
        from .. import archive
        archive.update(long, "trade")
    return long


def load(path=None, archive_kind: str | None = None) -> dict[str, pd.DataFrame]:
    """Wide frames (dates x tickers): adj_open, adj_close, close, volume, dollar_volume.

    Rows before a ticker's universe trade_start are dropped (OTC prints, SPAC shells,
    when-issued spin-offs), exactly as data.to_panel does for the fear panel. They
    are not the security the universe holds and must not enter the benchmark basket.

    ``archive_kind`` (live caches only) restores, from the per-ticker archive, any name
    the pull lacks or whose pulled series the archive does not accept (archive.merge).
    The result then carries the merge report under "report".
    """
    long = pd.read_parquet(path or PATH)
    report = None
    if archive_kind is not None:
        from .. import archive
        long, report = archive.merge(long, archive_kind)
    long["date"] = pd.to_datetime(long["date"])
    for t, d in universe.table()["trade_start"].dropna().items():
        long = long[~((long["ticker"] == t) & (long["date"] < pd.Timestamp(d)))]
    w = {c: long.pivot(index="date", columns="ticker", values=c).reindex(columns=universe.tickers())
         for c in COLS}
    factor = w["adj_close"] / w["close"]
    bad = ~(np.isfinite(w["open"]) & (w["open"] > 0) & np.isfinite(factor) & (factor > 0))
    adj_open = (w["open"] * factor).mask(bad)
    out = {"adj_open": adj_open, "adj_close": w["adj_close"], "close": w["close"],
           "volume": w["volume"], "dollar_volume": w["close"] * w["volume"]}
    if report is not None:
        out["report"] = report
    return out
