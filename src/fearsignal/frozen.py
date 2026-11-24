"""The frozen model specification and its integrity check.

``config/frozen.json`` holds every fitted parameter and every pre-declared
choice the signal needs. The holdout loader refuses to run unless that file
is committed to git, unmodified in the working tree, and its SHA-256 matches
what the caller presents. The commit that adds it therefore precedes, in the
repository history, every commit that contains holdout results.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from .config import HoldoutLocked

ROOT = Path(__file__).resolve().parents[2]
FROZEN_PATH = ROOT / "config" / "frozen.json"


def spec_hash(path: Path = FROZEN_PATH) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def committed_spec_hash(path: Path = FROZEN_PATH) -> str:
    """SHA-256 of the frozen spec, after checking it is committed and clean."""
    if not path.exists():
        raise HoldoutLocked(f"{path} does not exist; freeze the model first")
    rel = path.relative_to(ROOT).as_posix()
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", rel], cwd=ROOT, capture_output=True, text=True
    )
    if tracked.returncode != 0:
        raise HoldoutLocked(f"{rel} is not committed")
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", rel], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()
    if dirty:
        raise HoldoutLocked(f"{rel} has uncommitted changes")
    return spec_hash(path)


def load(path: Path = FROZEN_PATH) -> dict:
    return json.loads(path.read_text())
