"""The trading model's protocol: declared families, dev selection, one holdout read.

1. research/trading/candidates.md and trading/families.py declare every
   family and variant. They are committed before any backtest.
2. ``dev_selection`` runs every variant on 2024-01-02 .. 2025-12-31 (cohorts
   must exit by 2025-12-31) and applies the gates. It writes
   outputs/trading/dev/ and config/trading_selected.json.
3. config/trading_selected.json is committed. Its SHA-256 unlocks
   ``holdout_read``, which runs the selected variants on 2026-01-02 ..
   2026-11-27 once. A second read needs a logged reason.
4. Live tickets use the selected variants only.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .. import frozen
from ..config import HoldoutLocked
from ..data import RAW
from ..frozen import ROOT

OUT = ROOT / "outputs" / "trading"
SELECTED = ROOT / "config" / "trading_selected.json"
INPUT_FILES = ("pit_signals.parquet", "pit_market.parquet", "pit_refits.json", "trade_ohlcv.parquet")


def input_manifest() -> dict:
    """SHA-256 of the git-ignored inputs, so a committed result names what it used."""
    out = {}
    for f in INPUT_FILES:
        p = RAW / f
        out[f] = hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
    return out


def committed_selection_hash() -> str:
    return frozen.committed_spec_hash(SELECTED)


def check_holdout_unlock(presented_sha256: str, rerun_reason: str | None) -> None:
    sha = committed_selection_hash()
    if presented_sha256 != sha:
        raise HoldoutLocked("presented hash does not match the committed config/trading_selected.json")
    if (OUT / "holdout" / "holdout_results.json").exists() and not rerun_reason:
        raise RuntimeError("the trading holdout has already been read; pass a rerun reason to read it again")


def log_holdout(presented_sha256: str, last_session, reason: str | None) -> None:
    log = OUT / "holdout" / "RUNLOG.md"
    new = not log.exists()
    with open(log, "a") as fh:
        if new:
            fh.write("# Trading holdout run log\n\n| when (UTC) | selection sha256 | last session | reason |\n"
                     "|---|---|---|---|\n")
        when = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
        fh.write(f"| {when} | {presented_sha256[:12]}... | {pd.Timestamp(last_session).date()} | "
                 f"{reason or 'first read'} |\n")


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, pd.Timestamp):
        return str(o.date())
    return o


def write_json(path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(obj), indent=1) + "\n")
