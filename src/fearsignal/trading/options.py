"""Option helpers for expressing trades with level-2 approval.

Level 2 allows long calls, long puts, cash-secured puts and covered calls,
and nothing else: no spreads and no naked short options.

Black-Scholes-Merton with continuous dividend yield ``q`` (European, which is
adequate for choosing strikes; early exercise matters little at these DTEs for
non-dividend AI names). Live contract choice uses the broker's quoted implied
volatility and delta. Black-Scholes is used to back implied volatility out of
historical option prices for the expired-contract check, and to price
scenarios.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

RATE = 0.04          # flat risk-free rate assumption (annual, continuous)
YEAR = 365.0


def _d1(S, K, T, vol, r=RATE, q=0.0):
    return (math.log(S / K) + (r - q + 0.5 * vol * vol) * T) / (vol * math.sqrt(T))


def price(S: float, K: float, T: float, vol: float, kind: str, r: float = RATE, q: float = 0.0) -> float:
    """BSM price; T in years."""
    if T <= 0 or vol <= 0:
        return max(S - K, 0.0) if kind == "call" else max(K - S, 0.0)
    d1 = _d1(S, K, T, vol, r, q)
    d2 = d1 - vol * math.sqrt(T)
    if kind == "call":
        return S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)


def delta(S: float, K: float, T: float, vol: float, kind: str, r: float = RATE, q: float = 0.0) -> float:
    if T <= 0 or vol <= 0:
        itm = S > K if kind == "call" else S < K
        return (1.0 if kind == "call" else -1.0) * float(itm)
    d1 = _d1(S, K, T, vol, r, q)
    return math.exp(-q * T) * (norm.cdf(d1) if kind == "call" else norm.cdf(d1) - 1.0)


def implied_vol(px: float, S: float, K: float, T: float, kind: str, r: float = RATE, q: float = 0.0) -> float:
    """Implied volatility by Brent's method; NaN if the price is outside no-arbitrage bounds."""
    lo_px = price(S, K, T, 1e-4, kind, r, q)
    hi_px = price(S, K, T, 5.0, kind, r, q)
    if not (lo_px < px < hi_px):
        return float("nan")
    return brentq(lambda v: price(S, K, T, v, kind, r, q) - px, 1e-4, 5.0, xtol=1e-7)


def strike_for_delta(S: float, T: float, vol: float, kind: str, target: float,
                     r: float = RATE, q: float = 0.0) -> float:
    """Continuous strike with the given delta (calls: target in (0, 1); puts: in (-1, 0))."""
    lo, hi = S * 0.2, S * 5.0
    return brentq(lambda K: delta(S, K, T, vol, kind, r, q) - target, lo, hi, xtol=1e-6)


@dataclass
class Contract:
    symbol: str
    kind: str
    expiry: str
    strike: float
    bid: float
    ask: float
    mark: float
    iv: float
    delta: float
    theta: float
    open_interest: int
    instrument_id: str = ""

    @property
    def spread_pct(self) -> float:
        return (self.ask - self.bid) / self.mark if self.mark > 0 else float("inf")


def pick_by_delta(contracts: list[Contract], target: float) -> Contract | None:
    """Contract whose delta is closest to target (sign included)."""
    ok = [c for c in contracts if np.isfinite(c.delta) and c.mark > 0]
    if not ok:
        return None
    return min(ok, key=lambda c: abs(c.delta - target))


def summarize(c: Contract, spot: float, contracts: int = 1) -> dict:
    """What the trade costs and needs: premium at the ask, breakeven, max loss.

    A cash-secured put is quoted at the bid (it is sold), with the cash it ties up.
    """
    mult = 100 * contracts
    out = {"symbol": c.symbol, "type": c.kind, "expiry": c.expiry, "strike": c.strike,
           "bid": c.bid, "ask": c.ask, "mark": c.mark, "spread_pct_of_mark": round(c.spread_pct, 4),
           "iv": c.iv, "delta": c.delta, "theta_per_day": c.theta, "open_interest": c.open_interest,
           "spot": spot}
    if c.kind == "call":
        out.update(premium=c.ask * mult, breakeven=c.strike + c.ask,
                   breakeven_move=(c.strike + c.ask) / spot - 1, max_loss=c.ask * mult)
    elif c.kind == "put":
        out.update(premium=c.ask * mult, breakeven=c.strike - c.ask,
                   breakeven_move=(c.strike - c.ask) / spot - 1, max_loss=c.ask * mult)
    elif c.kind == "csp":
        out.update(credit=c.bid * mult, cash_secured=c.strike * mult,
                   effective_buy_price=c.strike - c.bid,
                   discount_to_spot=(c.strike - c.bid) / spot - 1,
                   yield_on_cash=c.bid / c.strike)
    return out
