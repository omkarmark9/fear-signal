"""Price data: download, cache, verify, and load as (date x ticker) panels.

The signal consumes exactly three fields per stock and day: adjusted close,
close, and low. The vendor's low is split-adjusted but not dividend-adjusted
(it is on the same basis as ``close``), so the dividend-and-split adjusted low
is ``low * adj_close / close``.

Why adjusted prices create no look-ahead: the vendor rescales the whole
history before each new ex-date by one constant. Every quantity used here is a
ratio of two prices on or before date t, so later rescalings cancel exactly.

Storage layout (``data/raw`` is git-ignored)::

    data/raw/dev.parquet       sessions 2023-01-03 .. 2025-12-31
    data/raw/holdout.parquet   sessions 2026-01-01 .. (frozen run only)
    data/manifest_<name>.json  committed: rows, tickers, date span, sha256
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import universe
from .config import DEV_END, ERA_START, HOLDOUT_START, HoldoutLocked, assert_dev, assert_era

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
FIELDS = ("adj_close", "close", "low")


@dataclass
class Panel:
    """Aligned (T x N) arrays on a common session calendar."""

    dates: pd.DatetimeIndex
    tickers: list[str]
    adj_close: np.ndarray
    close: np.ndarray
    low: np.ndarray

    @property
    def adj_low(self) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return self.low * (self.adj_close / self.close)

    def slice(self, start=None, end=None) -> "Panel":
        m = np.ones(len(self.dates), bool)
        if start is not None:
            m &= self.dates >= pd.Timestamp(start)
        if end is not None:
            m &= self.dates <= pd.Timestamp(end)
        return Panel(self.dates[m], list(self.tickers), self.adj_close[m], self.close[m], self.low[m])


# --------------------------------------------------------------------------- download

def download(name: str, start: str, end_exclusive: str, tickers: list[str] | None = None) -> Path:
    """Download daily bars with yfinance and write ``data/raw/<name>.parquet``."""
    import yfinance as yf  # optional dependency

    tickers = tickers or universe.tickers()

    def fetch(batch: list[str]) -> dict[str, pd.DataFrame]:
        raw = yf.download(
            batch, start=start, end=end_exclusive, auto_adjust=False, actions=False,
            progress=False, group_by="ticker", threads=True,
        )
        got = {}
        for t in batch:
            if t not in raw.columns.get_level_values(0):
                continue
            sub = raw[t][["Adj Close", "Close", "Low"]].dropna(how="all")
            if not sub.empty:
                got[t] = sub
        return got

    bars = fetch(tickers)
    for _ in range(2):  # transient vendor failures: retry the stragglers one by one
        for t in [t for t in tickers if t not in bars]:
            bars.update(fetch([t]))
    frames = []
    for t, sub in bars.items():
        sub = sub.rename(columns={"Adj Close": "adj_close", "Close": "close", "Low": "low"})
        sub.index = pd.DatetimeIndex(sub.index).tz_localize(None).normalize()
        sub.index.name = "date"
        sub = sub.copy()
        sub["ticker"] = t
        frames.append(sub.reset_index())
    long = pd.concat(frames, ignore_index=True)
    long = long[["date", "ticker", *FIELDS]].sort_values(["ticker", "date"]).reset_index(drop=True)
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / f"{name}.parquet"
    long.to_parquet(path, index=False)
    write_manifest(name, long)
    return path


def _digest(long: pd.DataFrame) -> str:
    canon = long.sort_values(["ticker", "date"]).to_csv(index=False, float_format="%.6f")
    return hashlib.sha256(canon.encode()).hexdigest()


def write_manifest(name: str, long: pd.DataFrame) -> dict:
    man = {
        "name": name,
        "rows": int(len(long)),
        "tickers": sorted(long["ticker"].unique().tolist()),
        "first_date": str(long["date"].min().date()),
        "last_date": str(long["date"].max().date()),
        "sha256_csv_6dp": _digest(long),
    }
    (ROOT / "data" / f"manifest_{name}.json").write_text(json.dumps(man, indent=2) + "\n")
    return man


def verify(name: str) -> bool:
    """True when the cached parquet matches its committed manifest hash."""
    man = json.loads((ROOT / "data" / f"manifest_{name}.json").read_text())
    long = pd.read_parquet(RAW / f"{name}.parquet")
    return _digest(long) == man["sha256_csv_6dp"]


# --------------------------------------------------------------------------- panels

def to_panel(long: pd.DataFrame) -> Panel:
    """Pivot long bars to a panel, apply listing overrides and the era rule."""
    long = long.copy()
    long["date"] = pd.to_datetime(long["date"])
    long = long[long["date"] >= ERA_START]
    starts = universe.table()["trade_start"].dropna()
    for t, d in starts.items():
        long = long[~((long["ticker"] == t) & (long["date"] < pd.Timestamp(d)))]
    order = [t for t in universe.tickers() if t in set(long["ticker"])]
    wide = {f: long.pivot(index="date", columns="ticker", values=f).reindex(columns=order) for f in FIELDS}
    dates = pd.DatetimeIndex(wide["close"].index)
    assert_era(dates)
    arr = {f: np.array(wide[f].to_numpy(dtype=float), copy=True) for f in FIELDS}
    # A bar is usable only if all three fields are positive and finite.
    bad = ~np.all([np.isfinite(arr[f]) & (arr[f] > 0) for f in FIELDS], axis=0)
    for f in FIELDS:
        arr[f][bad] = np.nan
    return Panel(dates, order, arr["adj_close"], arr["close"], arr["low"])


def load_dev_panel() -> Panel:
    """Development panel: 2023-01-03 .. 2025-12-31. Cannot contain 2026 data."""
    long = pd.read_parquet(RAW / "dev.parquet")
    long = long[pd.to_datetime(long["date"]) <= DEV_END]
    panel = to_panel(long)
    assert_dev(panel.dates)
    return panel


def load_holdout_panel(frozen_spec_sha256: str) -> Panel:
    """Full panel 2023-01-03 .. latest, for the single frozen holdout run.

    The caller must present the SHA-256 of ``config/frozen.json``; the file must
    be committed to git and unchanged in the working tree.
    """
    from .frozen import committed_spec_hash  # local import: avoids a cycle

    actual = committed_spec_hash()
    if frozen_spec_sha256 != actual:
        raise HoldoutLocked("frozen spec hash mismatch; holdout stays locked")
    dev = pd.read_parquet(RAW / "dev.parquet")
    hold = pd.read_parquet(RAW / "holdout.parquet")
    dev = dev[pd.to_datetime(dev["date"]) <= DEV_END]
    dev = align_adjustment_basis(dev, hold)
    hold = hold[pd.to_datetime(hold["date"]) >= HOLDOUT_START]
    long = pd.concat([dev, hold], ignore_index=True)
    return to_panel(long)


def align_adjustment_basis(dev: pd.DataFrame, hold: pd.DataFrame, tol: float = 1e-4) -> pd.DataFrame:
    """Put the dev pull's adjusted closes on the holdout pull's basis.

    The two pulls overlap in December 2025. Raw closes must agree; adjusted
    closes may differ by one constant per ticker (a dividend paid between the
    pulls rescales the vendor's whole history). That constant is estimated on
    the overlap and applied, so the first 2026 return is not contaminated by a
    change of basis. Any non-constant disagreement is an error.
    """
    ov = dev.merge(hold, on=["date", "ticker"], suffixes=("_d", "_h"))
    if ov.empty:
        raise ValueError("dev and holdout pulls do not overlap; cannot align adjustment basis")
    raw_gap = (ov["close_h"] / ov["close_d"] - 1).abs().max()
    if raw_gap > tol:
        raise ValueError(f"raw closes disagree between pulls (max rel. gap {raw_gap:.2e})")
    ratio = ov["adj_close_h"] / ov["adj_close_d"]
    c = ratio.groupby(ov["ticker"]).median()
    spread = (ratio / ov["ticker"].map(c) - 1).abs().max()
    if spread > tol:
        raise ValueError(f"adjusted closes differ by more than a constant (max {spread:.2e})")
    dev = dev.copy()
    dev["adj_close"] = dev["adj_close"] * dev["ticker"].map(c).fillna(1.0)
    return dev
