"""Project-wide constants and the research-protocol guards.

Two rules are hard and are enforced in code, not by convention:

* **Era rule.** No observation dated before ``ERA_START`` (2023-01-03, the
  first session of 2023) may enter any computation. Rolling windows warm up
  inside the era; nothing is back-filled from 2022.
* **Holdout lock.** Calendar 2026 (``HOLDOUT_START`` onward) is the locked
  out-of-sample period. Development code can only obtain data that ends on or
  before ``DEV_END``. Holdout data is reachable only through
  :func:`fearsignal.data.load_holdout_panel`, which requires the SHA-256 of a
  frozen model specification committed beforehand.
"""

from __future__ import annotations

import pandas as pd

ERA_START = pd.Timestamp("2023-01-03")
DEV_END = pd.Timestamp("2025-12-31")
HOLDOUT_START = pd.Timestamp("2026-01-01")

# Observation construction (identical to the v1 starting point).
SIGMA_WINDOW = 126          # sessions in the robust scale window, today excluded
MAD_TO_SIGMA = 1.4826       # 1 / Phi^{-1}(3/4): median|N(0, s^2)| * 1.4826 = s
ZERO_EPS = 0.001            # x <= ZERO_EPS is a "no dip" day
X_CLIP = 10.0               # x is clipped (v1) / right-censored (v2) here
SIGMA_FLOOR = 0.003         # a 0.3% daily scale floor: below this the series is stale

# Score bands on Z (identical to v1).
BANDS = (
    (-float("inf"), 1.0, "calm"),
    (1.0, 1.5, "uneasy"),
    (1.5, 2.0, "stressed"),
    (2.0, 2.5, "acute fear"),
    (2.5, float("inf"), "panic"),
)
Z_CLIP = 6.0

STATE_NAMES = ("calm", "stressed", "panic")


class EraViolation(RuntimeError):
    """Raised when data dated before ERA_START would enter a computation."""


class HoldoutLocked(RuntimeError):
    """Raised when development code tries to read 2026 data."""


def assert_era(index: pd.DatetimeIndex) -> None:
    """Fail loudly if any timestamp precedes the era start."""
    if len(index) and index.min() < ERA_START:
        raise EraViolation(
            f"data begins {index.min().date()}, before the hard era start {ERA_START.date()}"
        )


def assert_dev(index: pd.DatetimeIndex) -> None:
    """Fail loudly if any timestamp falls in the locked holdout."""
    if len(index) and index.max() > DEV_END:
        raise HoldoutLocked(
            f"data ends {index.max().date()}, inside the locked holdout (> {DEV_END.date()})"
        )
