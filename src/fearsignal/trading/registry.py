"""The declared families as code: one Variant per declared parameter set.

A Variant knows its rule (``select``: RuleFeatures -> boolean entry mask,
dates x tickers), its holding period, its side and benchmark, and which daily
excess series the gate reads. ``run`` executes it through the engine.
families.py fills ``FAMILIES``; nothing here depends on any result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from . import engine, signals


@dataclass(frozen=True)
class Variant:
    family: str
    name: str
    H: int
    select: Callable = field(compare=False, repr=False)
    side: str = "long"                  # "long" or "short_info"
    beta_benchmark: bool = True         # benchmark beta x basket (else the plain basket)
    series: str = "ct_excess_net"       # gate series: calendar-time or "excess_net" (1/H cohorts)
    cap: int | None = None              # max new names per signal day
    priority: str | None = None         # RuleFeatures field ranked ascending for the cap
    params: dict = field(default_factory=dict, compare=False)

    @property
    def id(self) -> str:
        return f"{self.family}.{self.name}"


@dataclass
class Family:
    id: str
    title: str
    variants: list[Variant]
    holm: bool = True                   # spends a Holm slot (False: reported, never adopted for trading)


def entries(v: Variant, rf: signals.RuleFeatures) -> pd.DataFrame:
    """Signal rows (date, ticker, weight) after no-re-entry and the per-day cap."""
    mask = v.select(rf, v)
    prio = getattr(rf, v.priority) if v.priority else None
    mask = signals.suppress_reentry(mask, v.H, priority=prio, cap=v.cap)
    return signals.to_signals(mask)


def run(v: Variant, rf: signals.RuleFeatures, prices: engine.Prices, start=None, end=None,
        cost_bps: float = 10.0, delay: int = 1, cache: dict | None = None,
        sig: pd.DataFrame | None = None) -> engine.Result:
    sig = sig if sig is not None else entries(v, rf)
    return engine.run_cohorts(sig, prices, H=v.H, cost_bps=cost_bps, delay=delay, side=v.side,
                              start=start, end=end, cache=cache,
                              beta=rf.beta if v.beta_benchmark else None)
