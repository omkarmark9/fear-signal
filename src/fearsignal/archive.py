"""A per-ticker archive of the live pulls, so a name the vendor stops serving keeps its bars.

Every live refresh downloads again and overwrites its cache:
- `fearsignal live` writes data/raw/live.parquet (bars from 2026-11-02);
- `fearsignal trading-data` writes data/raw/trade_ohlcv_live.parquet (OHLCV from 2023-01-03).

After a delisting, an acquisition or a symbol change, the vendor stops serving a ticker's
history (universe.py lists six names lost this way), or serves another company's history
under the reused symbol. Without an archive such a name would leave the prospective read
together with every one of its sample events. Acquisition targets are the likely ones, and
a takeover premium is a large loss for a bearish signal, so the loss would not be neutral.

``update`` runs after each live pull. It stores each ticker's whole series in
data/raw/archive/<kind>/<ticker>.parquet, which is git-ignored like all vendor data. The
stored series is replaced unless the new one is not acceptable:
- it has fewer bars since ARCHIVE_FROM (the vendor truncated or purged it); or
- its daily returns disagree with the stored series' on the bars both hold (the history
  was replaced).
Every decision other than a plain replacement is appended to data/raw/archive/<kind>/log.jsonl.

``merge`` runs when a live cache is loaded. A ticker the pull lacks, or whose pulled series
``update`` would not accept, is taken whole from the archive. One series keeps one price
basis, so every return inside it is the vendor's own.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import universe
from .data import RAW

ARCHIVE = RAW / "archive"
ARCHIVE_FROM = pd.Timestamp("2026-11-02")   # both live pulls cover every session from here
RET_TOL = 1e-4                              # two pulls agree on a return within this
AGREE_SHARE = 0.95                          # ... on at least this share of the returns both hold
KINDS = {"live": "adj_close", "trade": "adj_close"}


def _dir(kind: str):
    if kind not in KINDS:
        raise ValueError(kind)
    return ARCHIVE / kind


def acceptable(new: pd.DataFrame, old: pd.DataFrame, col: str = "adj_close") -> tuple[bool, str]:
    """Whether a newly pulled series may replace the stored one (see the module docstring)."""
    def bars(df):
        s = df.assign(date=pd.to_datetime(df["date"])).set_index("date")[col].astype(float)
        return s[np.isfinite(s) & (s > 0)].sort_index()
    a, b = bars(new), bars(old)
    n_new, n_old = int((a.index >= ARCHIVE_FROM).sum()), int((b.index >= ARCHIVE_FROM).sum())
    if n_new < n_old:
        return False, f"fewer bars since {ARCHIVE_FROM.date()} ({n_new} < {n_old})"
    j = a.index.intersection(b.index)
    ra, rb = np.log(a.loc[j]).diff().iloc[1:], np.log(b.loc[j]).diff().iloc[1:]
    if len(ra) >= 5:
        agree = float(((ra - rb).abs() <= RET_TOL).mean())
        if agree < AGREE_SHARE:
            return False, f"daily returns agree on only {agree:.0%} of the {len(ra)} shared bars (history replaced?)"
    return True, "ok"


def _log(kind: str, rows: list[dict]) -> None:
    if rows:
        with open(_dir(kind) / "log.jsonl", "a") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")


def update(long: pd.DataFrame, kind: str) -> dict:
    """Archive each universe ticker's series from a fresh pull (``long``: one row per date and ticker)."""
    d = _dir(kind)
    d.mkdir(parents=True, exist_ok=True)
    when = datetime.now(timezone.utc).isoformat(timespec="seconds")
    have = set(long["ticker"])
    log, stored, kept, absent = [], 0, [], []
    for t in universe.tickers():
        f = d / f"{t}.parquet"
        if t not in have:
            if f.exists():
                absent.append(t)
                log.append({"pull_utc": when, "ticker": t, "action": "absent from the pull; archive kept"})
            continue
        new = long[long["ticker"] == t]
        if f.exists():
            ok, why = acceptable(new, pd.read_parquet(f), KINDS[kind])
            if not ok:
                kept.append(t)
                log.append({"pull_utc": when, "ticker": t, "action": "archive kept", "why": why})
                continue
        else:
            log.append({"pull_utc": when, "ticker": t, "action": "first stored"})
        new.reset_index(drop=True).to_parquet(f, index=False)
        stored += 1
    _log(kind, log)
    return {"pull_utc": when, "stored": stored, "kept": kept, "absent": absent}


def merge(long: pd.DataFrame, kind: str) -> tuple[pd.DataFrame, dict]:
    """The pull, with each ticker it lacks, or whose series is not acceptable, taken from the archive."""
    d = _dir(kind)
    if not d.exists():
        return long, {"from_archive": {}}
    have = set(long["ticker"])
    parts, used = [long], {}
    for t in universe.tickers():
        f = d / f"{t}.parquet"
        if not f.exists():
            continue
        old = pd.read_parquet(f)
        if t in have:
            ok, why = acceptable(long[long["ticker"] == t], old, KINDS[kind])
            if ok:
                continue
        else:
            why = "absent from the pull"
        parts[0] = parts[0][parts[0]["ticker"] != t]
        parts.append(old[[c for c in long.columns if c in old.columns]])
        used[t] = why
    if not used:
        return long, {"from_archive": {}}
    out = pd.concat(parts, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    return out.sort_values(["ticker", "date"]).reset_index(drop=True), {"from_archive": used}
