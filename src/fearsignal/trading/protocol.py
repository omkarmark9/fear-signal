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
FROZEN_LIVE = RAW / "live_frozen_20261127.parquet"   # byte copy of live.parquet as of the declaration
INPUT_FILES = ("pit_signals.parquet", "pit_market.parquet", "pit_refits.json", "trade_ohlcv.parquet",
               # the fear panel that r, sigma, x and eligibility are recomputed from; backtests read the
               # frozen snapshot of the live bars, so `fearsignal live` can refresh live.parquet freely
               "dev.parquet", "holdout.parquet", FROZEN_LIVE.name)
INPUTS_LOCK = ROOT / "config" / "trading_inputs.json"


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
    if ((OUT / "holdout" / "holdout_results.json").exists() or (OUT / "holdout" / "RUNLOG.md").exists()) \
            and not rerun_reason:
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


DECLARED = ("research/trading/candidates.md", "research/trading/declaration.json",
            "src/fearsignal/trading/families.py")


def declaration_commit() -> str:
    """The commit that last changed the declaration; refuses if any declared file is dirty or untracked."""
    import subprocess
    for rel in DECLARED:
        tracked = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=ROOT, capture_output=True)
        if tracked.returncode != 0:
            raise HoldoutLocked(f"{rel} is not committed")
        dirty = subprocess.run(["git", "status", "--porcelain", "--", rel], cwd=ROOT,
                               capture_output=True, text=True).stdout.strip()
        if dirty:
            raise HoldoutLocked(f"{rel} has uncommitted changes")
    out = subprocess.run(["git", "log", "-n1", "--format=%H", "--", *DECLARED], cwd=ROOT,
                         capture_output=True, text=True).stdout.strip()
    return out


def check_dev_once(rerun_reason: str | None) -> None:
    if SELECTED.exists() and not rerun_reason:
        raise RuntimeError("the dev selection has already been run; pass a rerun reason to run it again")


def log_dev(commit: str, reason: str | None) -> None:
    log = OUT / "dev" / "RUNLOG.md"
    log.parent.mkdir(parents=True, exist_ok=True)
    new = not log.exists()
    with open(log, "a") as fh:
        if new:
            fh.write("# Trading dev-selection run log\n\n| when (UTC) | declaration commit | reason |\n|---|---|---|\n")
        when = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
        fh.write(f"| {when} | {commit[:12]} | {reason or 'first run'} |\n")


def check_inputs() -> dict:
    """Refuse unless every input matches the committed lock (config/trading_inputs.json).

    The first four hashes are the ones declared in research/trading/candidates.md. The
    fear-panel files were added before the dev run, after the pre-backtest audit.
    """
    lock = json.loads(INPUTS_LOCK.read_text())["sha256"]
    now = input_manifest()
    bad = {k: (lock.get(k), now.get(k)) for k in INPUT_FILES if lock.get(k) != now.get(k)}
    if bad:
        raise HoldoutLocked(f"inputs differ from config/trading_inputs.json: {sorted(bad)}")
    frozen.committed_spec_hash(INPUTS_LOCK)          # the lock itself must be committed and clean
    return now


CODE_PATHS = ("src", "research/trading", "config", "tests")


def code_state() -> dict:
    """HEAD and the src/fearsignal tree hash; refuses if any protocol-relevant path is dirty.

    Every gate, engine, feature and rule module lives under src/fearsignal, so the tree
    hash pins exactly the code a dev run or a holdout read executes."""
    import subprocess
    dirty = subprocess.run(["git", "status", "--porcelain", "--", *CODE_PATHS], cwd=ROOT,
                           capture_output=True, text=True).stdout.strip()
    if dirty:
        raise HoldoutLocked("uncommitted changes under " + ", ".join(CODE_PATHS) + ":\n" + dirty)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    tree = subprocess.run(["git", "rev-parse", "HEAD:src/fearsignal"], cwd=ROOT,
                          capture_output=True, text=True).stdout.strip()
    return {"head": head, "src_tree": tree}


def check_same_code(selection: dict) -> dict:
    """The holdout read must run the exact code the dev selection ran."""
    now = code_state()
    want = selection.get("code", {}).get("src_tree")
    if not want or now["src_tree"] != want:
        raise HoldoutLocked(f"src/fearsignal tree {now['src_tree'][:12]} differs from the dev selection's "
                            f"{(want or 'missing')[:12]}; the holdout must read the selected code")
    return now


def log_holdout_start(presented_sha256: str, reason: str | None) -> None:
    """Written BEFORE any 2026 outcome is computed, so an interrupted read is still on record."""
    log = OUT / "holdout" / "RUNLOG.md"
    log.parent.mkdir(parents=True, exist_ok=True)
    new = not log.exists()
    with open(log, "a") as fh:
        if new:
            fh.write("# Trading holdout run log\n\n| when (UTC) | selection sha256 | last session | reason |\n"
                     "|---|---|---|---|\n")
        when = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
        fh.write(f"| {when} | {presented_sha256[:12]}... | (read started) | {reason or 'first read'} |\n")


def holdout_matches_selection() -> bool:
    """True only if a holdout read exists and was made under the committed selection."""
    h = OUT / "holdout" / "holdout_results.json"
    if not (h.exists() and SELECTED.exists()):
        return False
    try:
        return json.loads(h.read_text())["_meta"]["selection_sha256"] == committed_selection_hash()
    except Exception:
        return False
