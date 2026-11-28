"""Model 2's prospective test (research/trading/PROSPECTIVE.md), and its live monitors.

The sample is exactly the first N_SESSIONS sessions after 2026-11-27, read once at the end.
- **P2-1, ND1:** the declared gate series. This is the calendar-time short-info excess:
  beta x the AI basket minus the name, net of 10 bps a side, over 20 sessions from the next
  open. The statistic is its stationary-bootstrap t, one-sided. It is claimed only if the
  per-event mean is also positive.
- **P2-2, the risk yardstick:** the daily mean QLIKE differential at h = 20 between frozen
  v3's sigma-tilde and EWMA(0.94), each with its dev level factor. The statistic is its
  stationary-bootstrap t (block 20), one-sided.
- Holm applies over {P2-1, P2-2} at family-wise 0.10, each threshold capped at 0.05, so each
  needs p < 0.05.

Exploratory, reported only: the v2 StepM survivors on the same sample.

The monitors run live only. They can lower ND's tier but never raise it, and they never
touch the read.
- **Sequential guard:** after every 25 closed ND1 events, ND is suspended if the running
  mean net excess is below the dev mean minus 2 s.e. (CR1 by episode).
- **Demotion:** at every 40th closed event, the last 40 are checked. A mean below -1 s.e.
  drops one tier. A second demotion means Tier 0.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import stats

from .. import frozen
from ..config import HoldoutLocked
from ..frozen import ROOT
from . import backtest, families as F, gates, protocol, registry, sizing

LAST_KNOWN = pd.Timestamp("2026-11-27")
N_SESSIONS = 400
H_RISK = 20
ALPHA_EACH = 0.05
GUARD_EVERY = 25
DEMOTE_EVERY = 40
OUT = protocol.OUT / "prospective"
RUNLOG = OUT / "RUNLOG.md"
DECL = ROOT / "config" / "trading_prospective.json"


def nd1_variant() -> registry.Variant:
    return {v.name: v for fam in F.HOLM_FAMILIES for v in fam.variants}["ND1"]


def window(dates: pd.DatetimeIndex) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Exactly the first N_SESSIONS sessions after LAST_KNOWN; refuses until all of them exist."""
    after = dates[dates > LAST_KNOWN]
    if len(after) < N_SESSIONS:
        raise HoldoutLocked(f"the prospective sample is the first {N_SESSIONS} sessions after "
                            f"{LAST_KNOWN.date()}; only {len(after)} exist")
    return after[0], after[N_SESSIONS - 1]


def episode_stats(trades: pd.DataFrame, dates: pd.DatetimeIndex) -> dict:
    """Per-event mean net excess, with CR1 s.e. by 5-session signal episode."""
    from . import explore as X
    e = trades["excess_net"].to_numpy(float)
    if len(e) < 3:
        return {"n_events": int(len(e)), "mean": float(e.mean()) if len(e) else None, "se": None, "t": None}
    cl = X._episodes(trades["signal_date"], dates)
    r = gates.ols_cluster(e, np.ones((len(e), 1)), cl)
    return {"n_events": int(len(e)), "mean": float(e.mean()), "se": float(r["se"][0]), "t": float(r["t"][0]),
            "episodes": int(r["clusters"])}


# ----------------------------------------------------------------------------- the read

def p21(rf, prices, start, end) -> dict:
    res, x = backtest.run_window(nd1_variant(), rf, prices, start, end)
    st = gates.variant_stats(x, start, end)
    ev = episode_stats(res.trades, prices.dates)
    p = float(stats.norm.sf(st["t"]))
    return {"gate_series": {k: st[k] for k in ("n_days", "mean_daily", "se_daily", "t", "days_invested")},
            "p_one_sided": p, "per_event": ev,
            "claimed": bool(p < ALPHA_EACH and (ev["mean"] or 0.0) > 0)}


def p22(rf, A, s, kappa: dict, kappa_ewma: dict, start, end) -> dict:
    q = sizing.qlike_oos(rf, [], kappa, kappa_ewma, start, end, hs=(H_RISK,), A_live=A, s_live=s)[H_RISK]
    p = float(stats.norm.cdf(q["t"]))
    return {**q, "p_one_sided": p, "claimed": bool(p < ALPHA_EACH),
            "switch_to_ewma": bool(q["diff_mean"] > q["diff_se"])}


def ir_post_all(rf_live, prices_live, start, end) -> dict:
    """IR_post for ND1 from dev, 2026 and prospective events (sizing declaration, section 2).

    Dev and 2026 events come from the hash-locked backtest inputs, and prospective
    events from the live state."""
    v = nd1_variant()
    rf0, p0 = backtest.load_inputs()
    parts = []
    for a, b in (gates.DEV, gates.HOLDOUT):
        res, _ = backtest.run_window(v, rf0, p0, a, b)
        parts.append(res.trades[["signal_date", "excess_net"]])
    res, _ = backtest.run_window(v, rf_live, prices_live, start, end)
    parts.append(res.trades[["signal_date", "excess_net"]])
    ev = pd.concat(parts, ignore_index=True)
    return {**sizing.ir_post(ev["excess_net"].to_numpy(float), ev["signal_date"]),
            "n_events": {"dev": int(len(parts[0])), "2026": int(len(parts[1])), "prospective": int(len(parts[2]))}}


def exploratory(rf, prices, start, end) -> dict:
    """The v2 StepM survivors on the prospective sample, reported only. Earnings-based rules need an
    earnings calendar that covers the sample; a rule that cannot be computed is reported as such."""
    from . import diagnostics as D, explore as X
    sel = json.loads((X.REG / "selection.json").read_text())
    recs, _ = X.load_registry()
    ctx = X.Ctx(rf=rf, prices=prices)
    out = {}
    for cid in sel["stepm_rejected"]:
        r = recs[recs.id == cid].iloc[0]
        try:
            out[cid] = D.event_diagnostics(X.Trial(r["family"], r["kind"], r["params"], int(r["round"])), ctx,
                                           start, end, n_placebo=200, seed=2)
        except Exception as exc:                      # reported, never fatal: these are not claims
            out[cid] = {"not_computed": f"{type(exc).__name__}: {exc}"}
    return out


def decide(r1: dict, r2: dict, nd_tier_before: int, ir_post: float | None) -> dict:
    """The declared consequences (research/trading/PROSPECTIVE.md)."""
    if r1["claimed"] and (ir_post or 0.0) > 0:
        nd = max(nd_tier_before, 2)
    elif (r1["per_event"]["mean"] or 0.0) <= 0 or r1["gate_series"]["mean_daily"] <= 0:
        nd = 0
    else:
        nd = nd_tier_before
    return {"ND_tier_after": int(nd), "risk_yardstick": "ewma" if r2["switch_to_ewma"] else "regime",
            "risk_skill_confirmed": r2["claimed"]}


def read(rerun_reason: str | None = None) -> dict:
    """The single prospective read. It must run from a checkout of the declared code tree."""
    from . import live
    frozen.committed_spec_hash(DECL)
    decl = json.loads(DECL.read_text())
    code = protocol.code_state()
    if code["src_tree"] != decl["src_tree"]:
        raise HoldoutLocked(f"src/fearsignal tree {code['src_tree'][:12]} is not the declared "
                            f"{decl['src_tree'][:12]}; run the read from a checkout of the declared commit")
    out_path = OUT / "read.json"
    if out_path.exists() and not rerun_reason:
        raise HoldoutLocked("the prospective read has already run; a second read needs a logged reason")
    mk, st, rf, prices, report = live.state()
    start, end = window(prices.dates)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(RUNLOG, "a") as fh:
        fh.write(f"| {datetime.now(timezone.utc):%Y-%m-%d %H:%M} | read started | {start.date()}..{end.date()} "
                 f"| {rerun_reason or 'first read'} |\n")
    spec3 = json.loads(frozen.FROZEN_V3_PATH.read_text())["stock_model"]
    cal = json.loads((protocol.OUT / "sizing_calibration.json").read_text())
    r1 = p21(rf, prices, start, end)
    r2 = p22(rf, spec3["A"], spec3["s"], cal["kappa"], cal["kappa_ewma"], start, end)
    irp = ir_post_all(rf, prices, start, end)
    status = live.family_status()
    res = {"window": [str(start.date()), str(end.date())], "n_sessions": N_SESSIONS, "P2_1_ND1": r1,
           "P2_2_risk": r2, "ND_IR_post": irp,
           "decision": decide(r1, r2, int(status["ND"]["tier"]), irp["IR_post"]),
           "exploratory": exploratory(rf, prices, start, end),
           "code": code, "data": {k: v for k, v in report.items() if k != "history_bars_filled"}}
    protocol.write_json(out_path, res)
    with open(RUNLOG, "a") as fh:
        fh.write(f"| {datetime.now(timezone.utc):%Y-%m-%d %H:%M} | read completed | {end.date()} | |\n")
    return res


# ----------------------------------------------------------------------------- live monitors

def closed_nd1_events(rf, prices, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    """ND1 events signalled after LAST_KNOWN whose 20-session window has closed by ``as_of``."""
    v = nd1_variant()
    dates = prices.dates
    as_of = pd.Timestamp(as_of) if as_of is not None else dates[-1]
    after = dates[dates > LAST_KNOWN]
    if len(after) <= v.H + 1:
        return pd.DataFrame(columns=["signal_date", "ticker", "excess_net"])
    last = gates.last_signal_date(dates, v.H, 1, as_of)
    if last < after[0]:
        return pd.DataFrame(columns=["signal_date", "ticker", "excess_net"])
    res = registry.run(v, rf, prices, start=after[0], end=last, cost_bps=10.0, delay=1)
    return res.trades.sort_values(["signal_date", "ticker"]).reset_index(drop=True)


def monitor(trades: pd.DataFrame, dates: pd.DatetimeIndex, dev_mean: float, tier: int) -> dict:
    """The declared sequential guard and demotion rule on closed prospective ND1 events."""
    n = len(trades)
    out = {"closed_events": int(n), "tier_declared": int(tier), "suspended_at": None, "demotions_at": []}
    for k in range(GUARD_EVERY, n + 1, GUARD_EVERY):
        s = episode_stats(trades.iloc[:k], dates)
        if s["se"] is not None and s["mean"] < dev_mean - 2.0 * s["se"]:
            out["suspended_at"] = int(k)
            break
    for k in range(DEMOTE_EVERY, n + 1, DEMOTE_EVERY):
        s = episode_stats(trades.iloc[k - DEMOTE_EVERY:k], dates)
        if s["se"] is not None and s["mean"] < -s["se"]:
            out["demotions_at"].append(int(k))
    eff = 0 if out["suspended_at"] is not None or len(out["demotions_at"]) >= 2 else max(0, tier - len(out["demotions_at"]))
    out["tier_effective"] = int(eff)
    out["running"] = episode_stats(trades, dates) if n else {"n_events": 0}
    return out


def nd_monitor(rf, prices, tier: int) -> dict:
    sel = json.loads(protocol.SELECTED.read_text())
    dev_mean = float(sel["families"]["ND"]["chosen_stats"]["mean_event_excess"])
    tr = closed_nd1_events(rf, prices)
    return {**monitor(tr, prices.dates, dev_mean, tier), "dev_mean_event_excess": dev_mean}
