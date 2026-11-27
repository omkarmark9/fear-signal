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
from .config import (DEV_END, ERA_START, HOLDOUT_START, PROSPECTIVE_SESSIONS, PROSPECTIVE_START,
                     V3_ESTIMATION_END, HoldoutLocked, assert_dev, assert_era)

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

def download(name: str, start: str, end_exclusive: str, tickers: list[str] | None = None,
             manifest: bool = True) -> Path:
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
    if manifest:
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
    """True when the cached parquet matches its committed manifest hash.

    Reading the holdout file, even to hash it, requires a committed frozen spec.
    """
    if name != "dev":
        from .frozen import committed_spec_hash

        committed_spec_hash()
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
    """Put an earlier pull (``dev``) on a later pull's (``hold``) price basis.

    The pulls must overlap. Between pulls the vendor rescales history by one
    constant per ticker:
    - a dividend rescales adjusted closes only;
    - a split rescales close, low and adjusted close alike.
    Both constants are estimated on the overlap, per ticker, and applied to the
    earlier pull, so the first return after the overlap carries no change of
    basis. Each must really be a constant on the overlap (spread <= tol), or
    it is an error. A ticker that has rows on both sides of the overlap but
    none inside it cannot be aligned, which is also an error, not a silent
    mix of bases. When nothing happened between pulls both constants are
    exactly 1 and the data are returned unchanged.
    """
    ov = dev.merge(hold, on=["date", "ticker"], suffixes=("_d", "_h"))
    if ov.empty:
        raise ValueError("pulls do not overlap; cannot align the price basis")
    tick = ov["ticker"]
    k_raw = (ov["close_h"] / ov["close_d"]).groupby(tick).median()      # split factor
    k_adj = (ov["adj_close_h"] / ov["adj_close_d"]).groupby(tick).median()
    for name, col, k in (("raw closes", "close", k_raw), ("raw lows", "low", k_raw),
                         ("adjusted closes", "adj_close", k_adj)):
        spread = (ov[f"{col}_h"] / ov[f"{col}_d"] / tick.map(k) - 1).abs().max()
        if spread > tol:
            raise ValueError(f"{name} differ between pulls by more than a per-ticker constant "
                             f"(max rel. spread {spread:.2e})")
    # tickers with rows on both sides of the overlap must have overlap rows
    lo, hi = pd.to_datetime(ov["date"]).min(), pd.to_datetime(ov["date"]).max()
    before = set(dev.loc[pd.to_datetime(dev["date"]) < lo, "ticker"])
    after = set(hold.loc[pd.to_datetime(hold["date"]) > hi, "ticker"])
    # a factor that is NaN (e.g. every overlap close missing) does not count as aligned
    aligned = set(k_raw.dropna().index) & set(k_adj.dropna().index)
    unaligned = sorted((before & after) - aligned)
    if unaligned:
        raise ValueError(f"no overlap rows to align {unaligned}; refusing to mix price bases")
    dev = dev.copy()
    kr = dev["ticker"].map(k_raw).fillna(1.0)
    dev["close"] = dev["close"] * kr
    dev["low"] = dev["low"] * kr
    dev["adj_close"] = dev["adj_close"] * dev["ticker"].map(k_adj).fillna(1.0)
    return dev


def load_v3_estimation_panel() -> Panel:
    """2023-01-03 .. 2026-11-23: everything v3 may be estimated on.

    The 2026 part was opened by v2's single holdout run (it requires v2's
    committed frozen spec to read) and is now estimation data, never
    selection data (docs/V3.md).
    """
    from .frozen import committed_spec_hash

    panel = load_holdout_panel(committed_spec_hash()).slice(end=V3_ESTIMATION_END)
    if panel.dates.max() != V3_ESTIMATION_END:
        raise ValueError(f"estimation panel ends {panel.dates.max().date()}, expected {V3_ESTIMATION_END.date()}")
    return panel


def load_prospective_panel(frozen_v3_sha256: str) -> Panel:
    """Full panel through the first PROSPECTIVE_SESSIONS sessions after 2026-11-23.

    Requires the committed, unmodified config/frozen_v3.json with the presented
    hash, and refuses to return anything until that many sessions exist: the
    pre-registered design is a single read at a fixed sample size.
    """
    from .frozen import FROZEN_V3_PATH, committed_spec_hash

    if frozen_v3_sha256 != committed_spec_hash(FROZEN_V3_PATH):
        raise HoldoutLocked("frozen v3 spec hash mismatch; prospective sample stays locked")
    base = load_v3_estimation_panel()
    dev = pd.read_parquet(RAW / "dev.parquet")
    hold = pd.read_parquet(RAW / "holdout.parquet")
    pro = pd.read_parquet(RAW / "prospective.parquet")
    # dev and holdout share one basis (aligned as in v2); the prospective pull,
    # made much later, overlaps November 2026 and sets the basis for all of it
    dev = align_adjustment_basis(dev[pd.to_datetime(dev["date"]) <= DEV_END], hold)
    hold = hold[(pd.to_datetime(hold["date"]) >= HOLDOUT_START)
                & (pd.to_datetime(hold["date"]) <= V3_ESTIMATION_END)]
    past = align_adjustment_basis(pd.concat([dev, hold], ignore_index=True), pro)
    new = pro[pd.to_datetime(pro["date"]) >= PROSPECTIVE_START]
    sessions = pd.DatetimeIndex(sorted(pd.to_datetime(new["date"]).unique()))
    if len(sessions) < PROSPECTIVE_SESSIONS:
        raise HoldoutLocked(f"only {len(sessions)} prospective sessions; the pre-registered read needs "
                            f"{PROSPECTIVE_SESSIONS}")
    new = new[pd.to_datetime(new["date"]) <= sessions[PROSPECTIVE_SESSIONS - 1]]
    panel = to_panel(pd.concat([past, new], ignore_index=True))
    if not panel.dates[panel.dates <= V3_ESTIMATION_END].equals(base.dates):
        raise ValueError("prospective panel's history does not match the v3 estimation calendar")
    return panel


def _frozen_history() -> pd.DataFrame:
    """Yahoo dev + holdout bars through 2026-11-23, on the holdout pull's basis."""
    from .frozen import committed_spec_hash

    committed_spec_hash()   # v2's spec gates the 2026 file, as everywhere else
    dev = pd.read_parquet(RAW / "dev.parquet")
    hold = pd.read_parquet(RAW / "holdout.parquet")
    dev = align_adjustment_basis(dev[pd.to_datetime(dev["date"]) <= DEV_END], hold)
    hold = hold[(pd.to_datetime(hold["date"]) >= HOLDOUT_START)
                & (pd.to_datetime(hold["date"]) <= V3_ESTIMATION_END)]
    return pd.concat([dev, hold], ignore_index=True)


def load_live_panel(source: str = "yahoo", live_path=None) -> Panel:
    """History through 2026-11-23 plus newer bars, for display (see live_panel)."""
    return live_panel(source, live_path)[0]


def live_panel(source: str = "yahoo", live_path=None) -> tuple[Panel, dict]:
    """History through 2026-11-23 plus newer bars, for display, with a data report.

    Live monitoring applies the frozen v3 spec to sessions after the freeze.
    It is not the prospective test: its bars live in their own git-ignored
    cache, it computes no pre-registered statistic, and it cannot alter the
    hash-locked spec.

    source="yahoo": newer bars from data/raw/live.parquet. The earlier pulls
    are put on its price basis through their November 2026 overlap.
    source="robinhood": completed sessions from the Robinhood connector's
    daily bars in data/raw/robinhood/*.json (fetched by the agent, since only
    it can call the connector). They are scaled onto the frozen history and
    also fill frozen-history bars the Yahoo pull is missing or broken on
    (stitch_robinhood). A session still in progress is not in Robinhood's
    daily bars; if data/raw/live.parquet has a newer, partial Yahoo bar it is
    appended as a provisional reading, scaled the same way.
    """
    past = _frozen_history()
    if source == "yahoo":
        live = pd.read_parquet(live_path or (RAW / "live.parquet"))
        past = align_adjustment_basis(past, live)
        new = live[pd.to_datetime(live["date"]) > V3_ESTIMATION_END]
        return to_panel(pd.concat([past, new], ignore_index=True)), {"source": "yahoo"}
    if source != "robinhood":
        raise ValueError(f"unknown source {source!r}")
    rh = robinhood_long(sorted((RAW / "robinhood").glob("*.json")))
    merged, report = stitch_robinhood(past, rh)
    report["source"] = "robinhood"
    last_rh = pd.to_datetime(rh["date"]).max()
    report["last_robinhood_session"] = str(last_rh.date())
    live_path = RAW / "live.parquet"
    if live_path.exists():
        live = pd.read_parquet(live_path)
        dates = pd.to_datetime(live["date"])
        partial = live[dates > last_rh]
        if len(partial):
            merged, prov = stitch_robinhood(merged, live[dates <= last_rh], fill_history=False,
                                            start_after=last_rh, extra=partial)
            report["provisional_session"] = str(pd.to_datetime(partial["date"]).max().date())
            report["provisional_max_close_spread_vs_robinhood"] = prov["max_close_spread"]
    return to_panel(merged), report


def robinhood_long(paths) -> pd.DataFrame:
    """Daily bars saved from the Robinhood connector (adjustment_type="all") as long rows.

    The bars are fully split- and dividend-adjusted, so close and low share one
    basis: adj_close = close = close_price and low = low_price. Gap-fill
    (interpolated) bars are dropped. If several files hold the same
    (ticker, date), the last file wins.
    """
    rows = []
    for path in paths:
        doc = json.loads(Path(path).read_text())
        for res in doc["data"]["results"]:
            for b in res.get("bars", []):
                if b.get("interpolated"):
                    continue
                c, lo = float(b["close_price"]), float(b["low_price"])
                rows.append((pd.Timestamp(b["begins_at"][:10]), res["symbol"], c, c, lo))
    df = pd.DataFrame(rows, columns=["date", "ticker", "adj_close", "close", "low"])
    return df.drop_duplicates(["ticker", "date"], keep="last").reset_index(drop=True)


def stitch_robinhood(past: pd.DataFrame, rh: pd.DataFrame, tol: float = 2e-3,
                     fill_history: bool = True, start_after=V3_ESTIMATION_END,
                     extra: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """Scale another vendor's bars onto ``past``'s basis and merge them.

    Per ticker: c = median(past adj_close / other close) over the overlap
    dates, which must agree with c within ``tol`` on every overlap date
    (vendors print the same exchange closes; an adjustment difference is a
    constant). The other vendor's rows are scaled by c on close, adjusted close
    and low, so close and low stay on one basis. They are used:
    - for every date after ``start_after``;
    - with fill_history, also for earlier dates where ``past`` has no usable
      bar (missing, or not all fields positive and finite).
    ``extra`` rows, which are not part of the overlap, are scaled with the same
    c and appended.
    Raises if a ticker that needs the other vendor's rows has no overlap.
    """
    past = past.copy()
    past["date"] = pd.to_datetime(past["date"])
    rh = rh.copy()
    rh["date"] = pd.to_datetime(rh["date"])
    good = np.all([np.isfinite(past[f]) & (past[f] > 0) for f in FIELDS], axis=0)
    ov = past[good].merge(rh, on=["date", "ticker"], suffixes=("_p", "_o"))
    ratio = ov["adj_close_p"] / ov["adj_close_o"]
    c = ratio.groupby(ov["ticker"]).median()
    spread = (ratio / ov["ticker"].map(c) - 1).abs()
    worst = spread.groupby(ov["ticker"]).max()
    if (worst > tol).any():
        bad = worst[worst > tol].round(4).to_dict()
        raise ValueError(f"closes differ between vendors by more than a constant: {bad}")
    adj_low_p = ov["low_p"] * ov["adj_close_p"] / ov["close_p"]
    low_gap = (adj_low_p / (ov["low_o"] * ov["ticker"].map(c)) - 1).abs()
    use = rh["date"] > pd.Timestamp(start_after)
    if fill_history:
        have = set(zip(past.loc[good, "ticker"], past.loc[good, "date"]))
        use |= ~pd.Series([(t, d) in have for t, d in zip(rh["ticker"], rh["date"])], index=rh.index)
    take = rh[use]
    if extra is not None:
        e = extra.copy()
        e["date"] = pd.to_datetime(e["date"])
        take = pd.concat([take, e], ignore_index=True)
    missing = sorted(set(take["ticker"]) - set(c.index))
    if missing:
        raise ValueError(f"no overlap to scale {missing}; refusing to mix price bases")
    k = take["ticker"].map(c)
    take = take.assign(adj_close=take["adj_close"] * k, close=take["adj_close"] * k,
                       low=take["low"] * take["adj_close"] / take["close"] * k)
    filled = take[take["date"] <= pd.Timestamp(start_after)]
    keep = ~pd.Series(list(zip(past["ticker"], past["date"]))).isin(
        set(zip(take["ticker"], take["date"]))).to_numpy()
    merged = pd.concat([past[keep], take[past.columns]], ignore_index=True)
    report = {"tickers_scaled": int(len(c)), "overlap_rows": int(len(ov)),
              "max_close_spread": float(worst.max()) if len(worst) else None,
              "median_low_gap": float(low_gap.median()) if len(low_gap) else None,
              "max_low_gap": float(low_gap.max()) if len(low_gap) else None,
              "history_bars_filled": sorted(f"{t} {d.date()}" for t, d in zip(filled["ticker"], filled["date"])),
              "new_sessions": sorted({str(d.date()) for d in take["date"] if d > pd.Timestamp(start_after)})}
    return merged, report
