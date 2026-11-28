"""Model 2's prospective test (research/trading/PROSPECTIVE.md), and its live monitors.

**Sample.** Exactly the first N_SESSIONS NYSE sessions after 2026-11-27, read once at the end.
The sessions come from the exchange calendar, not from the vendor's dates. The price data
must hold exactly those sessions, each with bars for most names.

**Claims.** Holm over {P2-1, P2-2} at family-wise 0.10, each threshold capped at 0.05, so each
claim needs p < 0.05.
- **P2-1, ND1:** the declared gate series. This is the calendar-time short-info excess:
  beta x the AI basket minus the name, net of 10 bps a side, over 20 sessions from the next
  open. The statistic is its stationary-bootstrap t, one-sided. It is claimed only if the
  per-event mean is also positive.
- **P2-2, the risk yardstick:** the daily mean QLIKE differential at h = 20 between frozen
  v3's sigma-tilde and EWMA(0.94). Each is scaled by the PIT dev level factor in
  outputs/trading/sizing_calibration.json. The statistic is its stationary-bootstrap t
  (block 20), one-sided.

**Reported only.**
- The five v2 StepM survivors, each with a strict replication rule: one-sided episode p
  below 0.01 and a placebo percentile of at least 0.95. They read a separate earnings
  calendar, pinned at read time. Without coverage they are reported as not computable.
- SR1 and RF on the same sample.

**Locks.** `read()` refuses unless all of these match `config/trading_prospective.json`
(committed after the code):
- the src/fearsignal tree;
- every committed input it reads;
- the hash-locked raw inputs (config/trading_inputs.json);
- the frozen v3 spec.

An environment mismatch is allowed only with a logged reason. A second read needs a logged
reason when any trace of an earlier read exists:
- the run log;
- a git-ignored marker that survives checkouts;
- a commit on any ref.

**Monitors (live only).** They can lower ND's tier but never raise it. Each firing is
appended to a log and stands from then on.
- **Sequential guard:** after every 25 closed events, ND is suspended if the running mean
  is below the dev mean minus 2 s.e. (CR1 by episode).
- **Demotion (rolling 40-trade):** after each closed event from the 40th on, ND drops one
  tier if the mean of the last 40 is below -1 s.e. A further demotion needs 40 more events.
  A second demotion means Tier 0.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import stats

from .. import frozen
from ..config import HoldoutLocked
from ..frozen import ROOT
from . import backtest, calendar, families as F, gates, protocol, registry, sizing

LAST_KNOWN = pd.Timestamp("2026-11-27")
N_SESSIONS = 400
H_RISK = 20
ALPHA_EACH = 0.05
REPLICATION_P = 0.01             # Bonferroni over the five v2 survivors
REPLICATION_PLACEBO = 0.95
GUARD_EVERY = 25
DEMOTE_WINDOW = 40
MIN_NAME_SHARE = 0.90            # a sample session needs bars for >= 90% of the median name count
EARNINGS_LOOKBACK_MONTHS = 3     # calendar coverage needed before the first sample session
OUT = protocol.OUT / "prospective"
RUNLOG = OUT / "RUNLOG.md"
MARKER = protocol.RAW / "prospective_read_marker.jsonl"          # git-ignored: survives checkouts
EARNINGS_SAMPLE = protocol.RAW / "earnings_calendar_prospective.json"
MONITOR_LOG = protocol.OUT / "live" / "nd_monitor_firings.jsonl"  # append-only; firings stand
DECL = ROOT / "config" / "trading_prospective.json"
LOCKED_FILES = ("outputs/trading/sizing_calibration.json", "config/trading_selected.json",
                "outputs/trading/holdout/holdout_results.json", "outputs/trading/v2/selection.json",
                "outputs/trading/v2/trials.jsonl", "outputs/trading/v2/trial_series.parquet",
                "config/trading_inputs.json", "config/frozen_v3.json")
ENV_PACKAGES = ("numpy", "pandas", "scipy", "pyarrow", "yfinance")


def _sha(path) -> str | None:
    from pathlib import Path
    p = Path(path)
    p = p if p.is_absolute() else ROOT / p
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def variant(name: str) -> registry.Variant:
    return {v.name: v for fam in F.HOLM_FAMILIES for v in fam.variants}[name]


def nd1_variant() -> registry.Variant:
    return variant("ND1")


# ----------------------------------------------------------------------------- the sample

def exchange_sessions(after: pd.Timestamp, n: int) -> pd.DatetimeIndex:
    """The first n NYSE sessions after ``after``, from the exchange calendar."""
    from ..forecast import HALF_DAYS_COVER_THROUGH
    days = pd.bdate_range(after + pd.Timedelta(days=1), periods=2 * n + 30)
    s = days[~days.isin(calendar.NYSE_HOLIDAYS)][:n]
    cover = min(HALF_DAYS_COVER_THROUGH, calendar.NYSE_HOLIDAYS.max() + pd.offsets.YearEnd(0))
    if s[-1] > cover:
        raise ValueError(f"the NYSE tables cover sessions through {cover.date()}; extend them")
    return s


def window(dates: pd.DatetimeIndex, counts: pd.Series | None = None):
    """(first, last, sessions) of the sample. Refuses until every sample session is in the data.

    The data's dates inside the sample must be exactly the exchange's sessions, and each
    session must have bars for at least MIN_NAME_SHARE of the median name count (``counts``:
    names with a close, by date)."""
    sess = exchange_sessions(LAST_KNOWN, N_SESSIONS)
    if dates.max() < sess[-1]:
        n_have = int(((dates > LAST_KNOWN) & (dates <= dates.max())).sum())
        raise HoldoutLocked(f"the prospective sample is the first {N_SESSIONS} sessions after "
                            f"{LAST_KNOWN.date()} (through {sess[-1].date()}); only {n_have} exist")
    have = dates[(dates > LAST_KNOWN) & (dates <= sess[-1])]
    extra, missing = have.difference(sess), sess.difference(have)
    if len(extra) or len(missing):
        raise HoldoutLocked(f"price dates disagree with the NYSE calendar: extra {[str(d.date()) for d in extra]}, "
                            f"missing {[str(d.date()) for d in missing]}")
    if counts is not None:
        c = counts.reindex(sess)
        thin = c[c < MIN_NAME_SHARE * float(c.median())]
        if len(thin):
            raise HoldoutLocked(f"sessions with too few names: {[str(d.date()) for d in thin.index]}")
    return sess[0], sess[-1], sess


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


# ----------------------------------------------------------------------------- statistics

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

    Dev and 2026 events come from the hash-locked backtest inputs, prospective events from the
    live state. N_eff counts ISO weeks with at least one entry, as declared."""
    v = nd1_variant()
    rf0, p0 = backtest.load_inputs()
    parts = []
    for a, b in (gates.DEV, gates.HOLDOUT):
        res, _ = backtest.run_window(v, rf0, p0, a, b)
        parts.append(res.trades[["signal_date", "entry", "ticker", "excess_net"]])
    res, _ = backtest.run_window(v, rf_live, prices_live, start, end)
    parts.append(res.trades[["signal_date", "entry", "ticker", "excess_net"]])
    ev = pd.concat(parts, ignore_index=True)
    if ev.duplicated(["signal_date", "ticker"]).any():
        raise RuntimeError("an ND1 event appears in two windows")
    return {**sizing.ir_post(ev["excess_net"].to_numpy(float), ev["entry"]),
            "n_events": {"dev": int(len(parts[0])), "2026": int(len(parts[1])), "prospective": int(len(parts[2]))}}


def decide(r1: dict, r2: dict, tier_start: int, ir_post: float | None, suspended: bool) -> dict:
    """The declared consequences (research/trading/PROSPECTIVE.md, "Consequences").

    ``tier_start`` is ND's tier at the end of the sample: its declared tier after the live
    monitors, recomputed on the sample's closed events."""
    means_pos = (r1["per_event"]["mean"] or 0.0) > 0 and r1["gate_series"]["mean_daily"] > 0
    if ir_post is not None and ir_post <= 0:
        nd, why = 0, "IR_post <= 0"
    elif r1["claimed"]:
        nd = 2
        why = "P2-1 claimed" + ("; pending the declared dev-data review of the live suspension" if suspended else "")
    elif not means_pos:
        nd, why = 0, "a P2-1 mean is <= 0"
    else:
        nd, why = int(tier_start), "P2-1 not claimed; tier after the live monitors"
    return {"ND_tier_after": int(nd), "why": why, "risk_yardstick": "ewma" if r2["switch_to_ewma"] else "regime",
            "risk_skill_confirmed": bool(r2["claimed"])}


# ----------------------------------------------------------------------------- reported only

def calendar_covers(path, start: pd.Timestamp, end: pd.Timestamp) -> tuple[bool, str]:
    """Whether a monthly-window earnings calendar covers every month from LOOKBACK before start to end."""
    from pathlib import Path
    path = Path(path)
    if not path.exists():
        return False, f"{path.name} does not exist"
    wins = set(pd.to_datetime(json.loads(path.read_text()).get("windows", [])).to_period("M"))
    need = pd.period_range((start - pd.DateOffset(months=EARNINGS_LOOKBACK_MONTHS)).to_period("M"),
                           end.to_period("M"), freq="M")
    gap = [str(m) for m in need if m not in wins]
    return (not gap), ("covered" if not gap else f"months missing: {gap[:6]}{'...' if len(gap) > 6 else ''}")


def survivors(rf, prices, start, end, A, s) -> dict:
    """The five v2 StepM survivors, each with its pre-registered replication rule."""
    from . import diagnostics as D, explore as X
    sel = json.loads((X.REG / "selection.json").read_text())
    ok, why = calendar_covers(EARNINGS_SAMPLE, start, end)
    if not ok:
        return {"calendar": why, **{cid: {"not_computable": f"the earnings calendar does not cover the sample ({why})"}
                                    for cid in sel["stepm_rejected"]}}
    recs, _ = X.load_registry()
    ctx = X.Ctx(rf=rf, prices=prices, earnings_path=EARNINGS_SAMPLE, A_live=A, s_live=s)
    out = {"calendar": why, "calendar_sha256": _sha(EARNINGS_SAMPLE)}
    for cid in sel["stepm_rejected"]:
        r = recs[recs.id == cid].iloc[0]
        tr = X.Trial(r["family"], r["kind"], r["params"], int(r["round"]))
        try:
            d = D.event_diagnostics(tr, ctx, start, end, n_placebo=200, seed=2)
            sig, H, side, delay = D._signals(tr, ctx)
            res, _ = D._run(sig, ctx, H, side, delay, start, end)
            ev = episode_stats(res.trades, prices.dates) if len(res.trades) else {"n_events": 0, "t": None}
            p = float(stats.norm.sf(ev["t"])) if ev.get("t") is not None else None
            pl = (d.get("placebo") or {}).get("percentile")
            d["per_event_episode"] = ev
            d["replicates"] = bool(p is not None and p < REPLICATION_P and pl is not None and pl >= REPLICATION_PLACEBO)
            out[cid] = d
        except Exception as exc:                      # reported only; never fatal
            out[cid] = {"not_computable": f"{type(exc).__name__}: {exc}"}
    return out


def v1_off_families(rf, prices, start, end) -> dict:
    out = {}
    try:
        res, x = backtest.run_window(variant("SR1"), rf, prices, start, end)
        out["SR1"] = {"gate_series": {k: v for k, v in gates.variant_stats(x, start, end).items()
                                      if k in ("n_days", "mean_daily", "se_daily", "t")},
                      "per_event": episode_stats(res.trades, prices.dates) if len(res.trades) else {"n_events": 0}}
    except Exception as exc:
        out["SR1"] = {"not_computable": f"{type(exc).__name__}: {exc}"}
    try:
        out["RF"] = backtest.strip(F.rf_evaluate(rf, prices, start, end))
    except Exception as exc:
        out["RF"] = {"not_computable": f"{type(exc).__name__}: {exc}"}
    return out


# ----------------------------------------------------------------------------- locks and the read

def environment() -> dict:
    import importlib.metadata as md
    import sys
    pk = {}
    for n in ENV_PACKAGES:
        try:
            pk[n] = md.version(n)
        except md.PackageNotFoundError:
            pk[n] = None
    frz = sorted(f"{(d.metadata.get('Name') or '?').lower()}=={d.version}" for d in md.distributions())
    return {"python": sys.version.split()[0], "packages": pk,
            "freeze_sha256": hashlib.sha256("\n".join(frz).encode()).hexdigest()}


def declaration_record(code_commit: str) -> dict:
    """The content of config/trading_prospective.json (written once, after the code commit)."""
    tree = subprocess.run(["git", "rev-parse", f"{code_commit}:src/fearsignal"], cwd=ROOT,
                          capture_output=True, text=True, check=True).stdout.strip()
    sess = exchange_sessions(LAST_KNOWN, N_SESSIONS)
    return {"code_commit": code_commit, "src_tree": tree,
            "committed_inputs_sha256": {f: _sha(f) for f in LOCKED_FILES},
            "raw_inputs_sha256": json.loads((ROOT / "config" / "trading_inputs.json").read_text())["sha256"],
            "environment": environment(),
            "sample": {"after": str(LAST_KNOWN.date()), "n_sessions": N_SESSIONS,
                       "first": str(sess[0].date()), "last": str(sess[-1].date())}}


def prior_reads() -> list[str]:
    """Every trace of an earlier read: the run log, the git-ignored marker, and commits on any ref."""
    def rel(p):
        try:
            return str(p.relative_to(ROOT))
        except ValueError:
            return str(p)
    found = [rel(f) for f in (RUNLOG, MARKER) if f.exists()]
    log = subprocess.run(["git", "log", "--all", "--format=%H", "--", "outputs/trading/prospective/"], cwd=ROOT,
                         capture_output=True, text=True).stdout.split()
    if log:
        found.append(f"commits touching outputs/trading/prospective/: {log[:3]}")
    return found


def preflight(decl: dict, code: dict, rerun_reason: str | None, env_reason: str | None) -> dict:
    """Every refusal of the read, before any data is touched."""
    if code["src_tree"] != decl["src_tree"]:
        raise HoldoutLocked(f"src/fearsignal tree {code['src_tree'][:12]} is not the declared {decl['src_tree'][:12]}; "
                            f"check out the declaration commit in place (not a worktree: data/raw is git-ignored)")
    bad = {f: (h, _sha(f)) for f, h in decl["committed_inputs_sha256"].items() if _sha(f) != h}
    if bad:
        raise HoldoutLocked(f"committed inputs differ from the declaration: {sorted(bad)}")
    prior = prior_reads()
    if prior and not rerun_reason:
        raise HoldoutLocked(f"an earlier prospective read exists ({prior}); a second read needs a logged reason")
    env = environment()
    if env["packages"] != decl["environment"]["packages"] and not env_reason:
        raise HoldoutLocked(f"environment {env['packages']} differs from the declared "
                            f"{decl['environment']['packages']}; install it, or pass a logged env_reason")
    return {"prior_reads": prior, "environment": env}


def read(rerun_reason: str | None = None, env_reason: str | None = None) -> dict:
    """The single prospective read (research/trading/PROSPECTIVE.md)."""
    from . import live
    decl_sha = frozen.committed_spec_hash(DECL)
    decl = json.loads(DECL.read_text())
    code = protocol.code_state()
    pre = preflight(decl, code, rerun_reason, env_reason)
    raw = protocol.check_inputs()                       # the hash-locked backtest inputs (IR_post)
    mk, st, rf, prices, report = live.state()
    counts = prices.adj_close.notna().sum(axis=1)
    start, end, sess = window(prices.dates, counts)
    stamp = {"when_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "head": code["head"],
             "sample": [str(start.date()), str(end.date())], "rerun_reason": rerun_reason, "env_reason": env_reason}
    OUT.mkdir(parents=True, exist_ok=True)
    with open(MARKER, "a") as fh:                      # logged before any outcome is computed
        fh.write(json.dumps(stamp) + "\n")
    with open(RUNLOG, "a") as fh:
        fh.write(f"| {stamp['when_utc']} | read started | {start.date()}..{end.date()} | head {code['head'][:12]} "
                 f"| {rerun_reason or 'first read'} | {env_reason or ''} |\n")
    spec3 = json.loads(frozen.FROZEN_V3_PATH.read_text())["stock_model"]
    cal = json.loads((protocol.OUT / "sizing_calibration.json").read_text())
    r1 = p21(rf, prices, start, end)
    r2 = p22(rf, spec3["A"], spec3["s"], cal["kappa"], cal["kappa_ewma"], start, end)
    irp = ir_post_all(rf, prices, start, end)
    status = live.family_status(apply_prospective=False)
    mon = nd_monitor(rf, prices, int(status["ND"]["tier"]), as_of=end, persist=False)
    res = {"declaration_sha256": decl_sha, "window": [str(start.date()), str(end.date())],
           "sessions": [str(d.date()) for d in sess], "P2_1_ND1": r1, "P2_2_risk": r2, "ND_IR_post": irp,
           "monitor_at_end": mon,
           "decision": decide(r1, r2, mon["tier_effective"], irp["IR_post"], mon["suspended_at"] is not None),
           "reported": {"v2_survivors": survivors(rf, prices, start, end, spec3["A"], spec3["s"]),
                        "v1_off": v1_off_families(rf, prices, start, end)},
           "code": code, "environment": pre["environment"], "raw_inputs_sha256": raw,
           "live_data_sha256": {"live.parquet": _sha(protocol.RAW / "live.parquet"),
                                "trade_ohlcv_live": _sha(_live_price_path()),
                                "earnings_calendar_prospective.json": _sha(EARNINGS_SAMPLE)},
           "data": {k: v for k, v in report.items() if k != "history_bars_filled"}, "stamp": stamp}
    protocol.write_json(OUT / "read.json", res)
    with open(RUNLOG, "a") as fh:
        fh.write(f"| {datetime.now(timezone.utc).isoformat(timespec='seconds')} | read completed | {end.date()} | | | |\n")
    return res


def _live_price_path():
    from .market_data import LIVE_PATH
    return LIVE_PATH


# ----------------------------------------------------------------------------- live monitors

def closed_nd1_events(rf, prices, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    """ND1 events signalled after LAST_KNOWN whose 20-session window has closed by ``as_of``."""
    v = nd1_variant()
    dates = prices.dates
    as_of = pd.Timestamp(as_of) if as_of is not None else dates[-1]
    empty = pd.DataFrame(columns=["signal_date", "ticker", "excess_net"])
    after = dates[dates > LAST_KNOWN]
    if len(after) <= v.H + 1:
        return empty
    last = gates.last_signal_date(dates, v.H, 1, as_of)
    if last < after[0]:
        return empty
    res = registry.run(v, rf, prices, start=after[0], end=last, cost_bps=10.0, delay=1)
    return res.trades.sort_values(["signal_date", "ticker"]).reset_index(drop=True)


def monitor(trades: pd.DataFrame, dates: pd.DatetimeIndex, dev_mean: float, tier: int) -> dict:
    """The declared sequential guard and the rolling 40-trade demotion on closed prospective events."""
    n = len(trades)
    out = {"closed_events": int(n), "tier_declared": int(tier), "suspended_at": None, "demotions_at": []}
    for k in range(GUARD_EVERY, n + 1, GUARD_EVERY):
        s = episode_stats(trades.iloc[:k], dates)
        if s["se"] is not None and s["mean"] < dev_mean - 2.0 * s["se"]:
            out["suspended_at"] = int(k)
            break
    next_ok = DEMOTE_WINDOW
    for k in range(DEMOTE_WINDOW, n + 1):
        if k < next_ok:
            continue
        s = episode_stats(trades.iloc[k - DEMOTE_WINDOW:k], dates)
        if s["se"] is not None and s["mean"] < -s["se"]:
            out["demotions_at"].append(int(k))
            next_ok = k + DEMOTE_WINDOW                # a further demotion needs 40 more events
    out["tier_effective"] = effective_tier(tier, out["suspended_at"] is not None, len(out["demotions_at"]))
    out["running"] = episode_stats(trades, dates) if n else {"n_events": 0}
    return out


def effective_tier(tier: int, suspended: bool, demotions: int) -> int:
    if suspended or demotions >= 2:
        return 0
    return max(0, int(tier) - int(demotions))


def _persisted() -> list[dict]:
    if not MONITOR_LOG.exists():
        return []
    return [json.loads(x) for x in MONITOR_LOG.read_text().splitlines() if x.strip()]


def nd_monitor(rf, prices, tier: int, as_of=None, persist: bool = True) -> dict:
    """The monitors on today's closed events, merged with every firing logged before.

    A logged firing stands even if a later vendor revision would no longer produce it."""
    sel = json.loads(protocol.SELECTED.read_text())
    dev_mean = float(sel["families"]["ND"]["chosen_stats"]["mean_event_excess"])
    tr = closed_nd1_events(rf, prices, as_of)
    m = monitor(tr, prices.dates, dev_mean, tier)
    old = _persisted()
    seen = {(f["rule"], f["at_event"]) for f in old}
    new = []
    if m["suspended_at"] is not None and ("guard", m["suspended_at"]) not in seen:
        new.append({"rule": "guard", "at_event": m["suspended_at"]})
    for k in m["demotions_at"]:
        if ("demotion", k) not in seen:
            new.append({"rule": "demotion", "at_event": k})
    if persist and new:
        MONITOR_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(MONITOR_LOG, "a") as fh:
            for f in new:
                fh.write(json.dumps({**f, "as_of": str(prices.dates[-1].date()),
                                     "logged_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}) + "\n")
    allf = old + (new if persist else [])
    suspended = m["suspended_at"] is not None or any(f["rule"] == "guard" for f in allf)
    demotions = max(len(m["demotions_at"]), len({f["at_event"] for f in allf if f["rule"] == "demotion"}))
    return {**m, "dev_mean_event_excess": dev_mean, "logged_firings": allf,
            "tier_effective": effective_tier(tier, suspended, demotions)}


def apply_monitor(status: dict, mon: dict) -> dict:
    """Lower ND's live tier to the monitors' effective tier. It is never raised."""
    nd = dict(status["ND"])
    nd["monitor"] = mon
    if mon["tier_effective"] < nd.get("tier", 0):
        nd["why"] = nd.get("why", "") + f"; lowered to Tier {mon['tier_effective']} by the live monitor"
        nd["tier"] = mon["tier_effective"]
    return {**status, "ND": nd}
