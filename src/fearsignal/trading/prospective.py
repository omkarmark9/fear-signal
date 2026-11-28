"""Model 2's prospective test (research/trading/PROSPECTIVE.md), and its live monitors.

**Sample.** Exactly the first N_SESSIONS sessions the NYSE held after 2026-11-27, read
once at the end.
- The sessions come from the exchange calendar, not from the vendor's dates.
- An unscheduled closure (a day of mourning, a storm) is declared at the read with a
  logged reason. It is accepted only if neither dataset has a bar that day, and the
  sample then runs one session longer.
- Both datasets, the trading price cache and model 1's panel, must hold exactly those
  sessions. Each session needs bars for at least 90% of the universe.

**Claims.** Holm over {P2-1, P2-2} at family-wise 0.10, each threshold capped at 0.05, so
each claim needs p < 0.05.
- **P2-1, ND1:** the declared gate series.
  - This is the calendar-time short-info excess: beta x the AI basket minus the name,
    net of 10 bps a side, over 20 sessions from the next open.
  - The statistic is its stationary-bootstrap t, one-sided.
  - It is claimed only if the per-event mean is also positive.
- **P2-2, the risk yardstick:** the daily mean QLIKE differential at h = 20 between frozen
  v3's sigma-tilde and EWMA(0.94).
  - Each is scaled by the PIT dev level factor in outputs/trading/sizing_calibration.json.
  - The statistic is its stationary-bootstrap t (block 20), one-sided.

**Reported only.**
- The five v2 StepM survivors, each with a strict replication rule: a one-sided episode p
  below 0.01, from the CR1 t against t(G-1), and a placebo percentile of at least 0.95.
  - They read a separate earnings calendar, pinned at read time.
  - Without coverage they are reported as not computable.
- SR1 and the declared RF1, by the criteria the 2026 read applied.

**Locks.** `read()` refuses unless all of these match `config/trading_prospective.json`
(committed after the code):
- the src/fearsignal tree;
- every committed input it reads;
- the hash-locked raw inputs (config/trading_inputs.json);
- the frozen v3 spec;
- Python (major.minor) and numpy, pandas, scipy and pyarrow, unless a reason is logged.
  yfinance and a digest of every installed package are recorded only.

A second read needs a logged reason when any trace of an earlier read exists:
- the run log;
- a git-ignored marker that survives checkouts;
- a commit on any ref.

Every live input the read loads is hashed before and after loading and copied byte for
byte into data/raw/prospective_reads/ before any outcome is computed.

**Monitors (live only).** They can lower ND's tier but never raise it. Each firing is
appended to a git-ignored log (which survives checkouts) and stands from then on.
- **Sequential guard:** after every 25 closed events, ND is suspended if the running mean
  is below the dev mean minus 2 s.e.
- **Demotion (rolling 40-trade):** after each closed event from the 40th on, ND drops one
  tier if the mean of the last 40 is below -1 s.e. A further demotion needs 40 more events.
  A second demotion means Tier 0.
- Both use the per-trade s.e. of the mean (sd / sqrt(n)).
- A logged firing is matched to today's events by the event it fired on, so a vendor
  revision that shifts the event count cannot turn one demotion into two.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import stats

from .. import frozen, universe
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
REF_NAMES = len(universe.tickers())   # 134: a sample session needs bars for >= 90% of the universe
MIN_NAME_SHARE = 0.90
EARNINGS_LOOKBACK_MONTHS = 3     # calendar coverage needed before the first sample session
CAL_MIN_ROWS = 5                 # report rows a required calendar month must hold
OUT = protocol.OUT / "prospective"
RUNLOG = OUT / "RUNLOG.md"
MARKER = protocol.RAW / "prospective_read_marker.jsonl"          # git-ignored: survives checkouts
READ_LOCK = protocol.RAW / "prospective_read.lock"
READ_INPUTS = protocol.RAW / "prospective_reads"                 # byte copies of each read's live inputs
EARNINGS_SAMPLE = protocol.RAW / "earnings_calendar_prospective.json"
MONITOR_LOG = protocol.RAW / "nd_monitor_firings.jsonl"          # git-ignored: survives checkouts; firings stand
DECL = ROOT / "config" / "trading_prospective.json"
LOCKED_FILES = ("outputs/trading/sizing_calibration.json", "config/trading_selected.json",
                "outputs/trading/holdout/holdout_results.json", "outputs/trading/v2/selection.json",
                "outputs/trading/v2/trials.jsonl", "outputs/trading/v2/trial_series.parquet",
                "config/trading_inputs.json", "config/frozen_v3.json")
ENV_PACKAGES = ("numpy", "pandas", "scipy", "pyarrow")   # must match the declaration, or a logged reason
ENV_RECORDED = ("yfinance",)                              # recorded only: the refreshes need it, the read does not
TRADE_COLS = ["signal_date", "ticker", "weight", "entry", "exit", "ret", "basket_ret", "beta", "excess_net"]


def _sha(path) -> str | None:
    from pathlib import Path
    p = Path(path)
    p = p if p.is_absolute() else ROOT / p
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def variant(name: str) -> registry.Variant:
    return {v.name: v for fam in F.HOLM_FAMILIES for v in fam.variants}[name]


def nd1_variant() -> registry.Variant:
    return variant("ND1")


def _trades(res) -> pd.DataFrame:
    """A run's trades, typed even when there are none (the engine returns a frame with no columns)."""
    t = res.trades
    return t if len(t) else pd.DataFrame(columns=TRADE_COLS)


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


def _closures(closed) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(sorted(set(pd.to_datetime(list(closed))))) if closed is not None and len(closed) \
        else pd.DatetimeIndex([])


def sample_sessions(closed=()) -> pd.DatetimeIndex:
    """The first N_SESSIONS sessions the NYSE held after LAST_KNOWN.

    ``closed`` lists scheduled sessions on which the exchange did not open (unscheduled
    closures). Each must be a scheduled session of the sample, and the sample then runs one
    session further."""
    closed = _closures(closed)
    table = exchange_sessions(LAST_KNOWN, N_SESSIONS + len(closed))
    bad = closed.difference(table)
    if len(bad):
        raise HoldoutLocked(f"declared closures {[str(d.date()) for d in bad]} are not scheduled sessions "
                            f"of the sample")
    return table.difference(closed)


def window(dates: pd.DatetimeIndex, counts=None, closed=()):
    """(first, last, sessions) of the sample. Refuses until every sample session is in the data.

    ``counts``: names with a usable bar, by date, as one Series or a dict of labelled Series.
    The first decides whether a date has any bar at all; each must reach MIN_NAME_SHARE of
    the universe (REF_NAMES) on every sample session. The data's dates inside the sample must
    be exactly the exchange's sessions. A declared closure must have no bar."""
    closed = _closures(closed)
    sess = sample_sessions(closed)
    cnt = {} if counts is None else (counts if isinstance(counts, dict) else {"names with a close": counts})
    present = dates
    if cnt:
        first = next(iter(cnt.values())).reindex(dates).fillna(0).to_numpy(float)
        present = dates[first > 0]
    if not len(present) or present.max() < sess[-1]:
        n_have = int((present > LAST_KNOWN).sum())
        raise HoldoutLocked(f"the prospective sample is the first {N_SESSIONS} sessions after "
                            f"{LAST_KNOWN.date()} (through {sess[-1].date()}); only {n_have} exist")
    have = present[(present > LAST_KNOWN) & (present <= sess[-1])]
    opened = closed.intersection(present)
    if len(opened):
        raise HoldoutLocked(f"declared closures have bars, so the exchange was open: {[str(d.date()) for d in opened]}")
    extra, missing = have.difference(sess), sess.difference(have)
    if len(extra) or len(missing):
        hint = ("; a scheduled session the exchange did not hold is declared at the read "
                "(--closed DATE --calendar-reason TEXT)") if len(missing) else ""
        raise HoldoutLocked(f"price dates disagree with the NYSE calendar: extra {[str(d.date()) for d in extra]}, "
                            f"missing {[str(d.date()) for d in missing]}{hint}")
    floor = MIN_NAME_SHARE * REF_NAMES
    for label, c in cnt.items():
        cc = c.reindex(sess).fillna(0)
        thin = cc[cc < floor]
        if len(thin):
            raise HoldoutLocked(f"sessions with too few names ({label}; at least {floor:.1f} of {REF_NAMES} "
                                f"needed): {[str(d.date()) for d in thin.index]}")
    return sess[0], sess[-1], sess


def coverage_report(sess: pd.DatetimeIndex, frames: dict) -> dict:
    """Per dataset: names per sample session, and the names with a usable bar on fewer than 90%
    of them (``frames``: boolean dates x tickers, True where the bar is usable)."""
    out = {}
    for label, ok in frames.items():
        x = ok.reindex(index=sess, fill_value=False).astype(bool)
        per_day, share = x.sum(axis=1), x.mean(axis=0)
        low = share[share < MIN_NAME_SHARE]
        out[label] = {"names_per_session": {"min": int(per_day.min()), "median": float(per_day.median())},
                      "names_below_90pct_of_sessions": {str(t): round(float(v), 3) for t, v in low.items()}}
    return out


def episode_stats(trades: pd.DataFrame, dates: pd.DatetimeIndex) -> dict:
    """Per-event mean net excess, with CR1 s.e. by 5-session signal episode."""
    from . import explore as X
    e = trades["excess_net"].to_numpy(float) if len(trades) else np.zeros(0)
    if len(e) < 3:
        return {"n_events": int(len(e)), "mean": float(e.mean()) if len(e) else None, "se": None, "t": None}
    cl = X._episodes(trades["signal_date"], dates)
    if len(np.unique(cl)) < 2:                        # CR1 needs two clusters
        return {"n_events": int(len(e)), "mean": float(e.mean()), "se": None, "t": None, "episodes": 1}
    r = gates.ols_cluster(e, np.ones((len(e), 1)), cl)
    return {"n_events": int(len(e)), "mean": float(e.mean()), "se": float(r["se"][0]), "t": float(r["t"][0]),
            "episodes": int(r["clusters"])}


def trade_stats(e) -> dict:
    """Mean and the per-trade s.e. of the mean (sd / sqrt(n)), as the monitors use them."""
    e = np.asarray(e, float)
    n = len(e)
    if n < 3:
        return {"n_events": int(n), "mean": float(e.mean()) if n else None, "se": None}
    return {"n_events": int(n), "mean": float(e.mean()), "se": float(e.std(ddof=1) / np.sqrt(n))}


# ----------------------------------------------------------------------------- statistics

def p21(rf, prices, start, end) -> dict:
    res, x = backtest.run_window(nd1_variant(), rf, prices, start, end)
    st = gates.variant_stats(x, start, end)
    ev = episode_stats(_trades(res), prices.dates)
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
    cols = ["signal_date", "entry", "ticker", "excess_net"]
    rf0, p0 = backtest.load_inputs()
    parts = []
    for a, b in (gates.DEV, gates.HOLDOUT):
        res, _ = backtest.run_window(v, rf0, p0, a, b)
        parts.append(_trades(res)[cols])
    res, _ = backtest.run_window(v, rf_live, prices_live, start, end)
    parts.append(_trades(res)[cols])
    ev = pd.concat([p for p in parts if len(p)], ignore_index=True)
    if ev.duplicated(["signal_date", "ticker"]).any():
        raise RuntimeError("an ND1 event appears in two windows")
    return {**sizing.ir_post(ev["excess_net"].to_numpy(float), ev["entry"]),
            "n_events": {"dev": int(len(parts[0])), "2026": int(len(parts[1])), "prospective": int(len(parts[2]))}}


def decide(r1: dict, r2: dict, tier_start: int, ir_post: float | None, pending_review: bool) -> dict:
    """The declared consequences (research/trading/PROSPECTIVE.md, "Consequences").

    ``tier_start`` is ND's tier at the end of the sample: its declared tier after the live
    monitors, recomputed on the sample's closed events and on every logged firing.
    ``pending_review``: the monitors suspended ND or demoted it twice during the sample,
    which puts it at Tier 0 pending a review on dev data only (declaration, DEMOTION and
    section 7). A claim then waits for that review."""
    means_pos = (r1["per_event"]["mean"] or 0.0) > 0 and r1["gate_series"]["mean_daily"] > 0
    after_review = None
    if ir_post is not None and ir_post <= 0:
        nd, why = 0, "IR_post <= 0"
    elif r1["claimed"] and pending_review:
        nd, after_review = 0, 2
        why = ("P2-1 claimed; Tier 2 waits for the declared review on dev data only "
               "(the live monitors suspended ND or demoted it twice)")
    elif r1["claimed"]:
        nd, why = 2, "P2-1 claimed"
    elif not means_pos:
        nd, why = 0, "a P2-1 mean is <= 0"
    else:
        nd, why = int(tier_start), "P2-1 not claimed; tier after the live monitors"
    return {"ND_tier_after": int(nd), "ND_tier_after_review": after_review, "why": why,
            "risk_yardstick": "ewma" if r2["switch_to_ewma"] else "regime",
            "risk_skill_confirmed": bool(r2["claimed"])}


# ----------------------------------------------------------------------------- reported only

def calendar_covers(path, start: pd.Timestamp, end: pd.Timestamp) -> tuple[bool, str]:
    """Whether the read-time earnings calendar covers the sample, by its metadata and its data.

    - Its monthly windows cover every month from EARNINGS_LOOKBACK_MONTHS before ``start``
      through ``end``'s month.
    - Each of those months holds at least CAL_MIN_ROWS report rows (2023-11..2026-11: 10 or more).
    - In each calendar quarter whose months are all required, at least MIN_NAME_SHARE of the
      universe reports (every name has reported in every quarter since 2025Q2).
    - It was fetched after ``end``, so the sample's reports are in it."""
    from pathlib import Path
    path = Path(path)
    if not path.exists():
        return False, f"{path.name} does not exist"
    try:
        doc = json.loads(path.read_text())
        wins = set(pd.to_datetime(list(doc.get("windows", []))).to_period("M"))
        rows = pd.DataFrame([{"symbol": str(r["symbol"]), "date": pd.Timestamp(r["report"]["date"])}
                             for r in doc.get("rows", [])], columns=["symbol", "date"])
        fetched = pd.Timestamp(doc["fetched_utc"])
    except Exception as exc:                       # a malformed calendar is not computable, never fatal
        return False, f"{path.name} is unreadable ({type(exc).__name__}: {exc})"
    need = pd.period_range((start - pd.DateOffset(months=EARNINGS_LOOKBACK_MONTHS)).to_period("M"),
                           end.to_period("M"), freq="M")
    gap = [str(m) for m in need if m not in wins]
    if gap:
        return False, f"months missing: {gap[:6]}{'...' if len(gap) > 6 else ''}"
    by_month = rows.groupby(rows["date"].dt.to_period("M")).size() if len(rows) else pd.Series(dtype=int)
    thin = [str(m) for m in need if int(by_month.get(m, 0)) < CAL_MIN_ROWS]
    if thin:
        return False, f"months with fewer than {CAL_MIN_ROWS} report rows: {thin[:6]}{'...' if len(thin) > 6 else ''}"
    quarters = sorted({m.asfreq("Q") for m in need if all(q in set(need) for q in
                                                         pd.period_range(m.asfreq("Q").asfreq("M", "s"),
                                                                         m.asfreq("Q").asfreq("M", "e"), freq="M"))})
    names = rows.groupby(rows["date"].dt.to_period("Q"))["symbol"].nunique() if len(rows) else pd.Series(dtype=int)
    short = [str(q) for q in quarters if int(names.get(q, 0)) < MIN_NAME_SHARE * REF_NAMES]
    if short:
        return False, f"quarters where fewer than {MIN_NAME_SHARE:.0%} of the universe reports: {short}"
    when = fetched.tz_convert(None) if fetched.tzinfo is not None else fetched
    if when.normalize() <= end:
        return False, f"fetched {when.date()}, not after the sample's end {end.date()}"
    return True, "covered"


def survivors(rf, prices, start, end, A, s, cal_path=None) -> dict:
    """The five v2 StepM survivors, each with its pre-registered replication rule. Reported only."""
    from . import diagnostics as D, explore as X
    cal_path = cal_path or EARNINGS_SAMPLE
    sel = json.loads((X.REG / "selection.json").read_text())
    ok, why = calendar_covers(cal_path, start, end)
    if not ok:
        return {"calendar": why, **{cid: {"not_computable": f"the earnings calendar does not cover the sample ({why})"}
                                    for cid in sel["stepm_rejected"]}}
    out = {"calendar": why, "calendar_sha256": _sha(cal_path)}
    try:
        recs, _ = X.load_registry()
        ctx = X.Ctx(rf=rf, prices=prices, earnings_path=cal_path, A_live=A, s_live=s)
    except Exception as exc:
        return {**out, **{cid: {"not_computable": f"{type(exc).__name__}: {exc}"} for cid in sel["stepm_rejected"]}}
    for cid in sel["stepm_rejected"]:
        try:
            r = recs[recs.id == cid].iloc[0]
            tr = X.Trial(r["family"], r["kind"], r["params"], int(r["round"]))
            d = D.event_diagnostics(tr, ctx, start, end, n_placebo=200, seed=2)
            sig, H, side, delay = D._signals(tr, ctx)
            res, _ = D._run(sig, ctx, H, side, delay, start, end)
            ev = episode_stats(_trades(res), prices.dates)
            G = ev.get("episodes")
            p = float(stats.t.sf(ev["t"], G - 1)) if ev.get("t") is not None and G and G > 1 else None
            pl = (d.get("placebo") or {}).get("percentile")
            d["per_event_episode"] = {**ev, "p_one_sided": p, "reference": "t(G-1), G = episodes"}
            d["replicates"] = bool(p is not None and p < REPLICATION_P and pl is not None and pl >= REPLICATION_PLACEBO)
            out[cid] = d
        except Exception as exc:                      # reported only; never fatal
            out[cid] = {"not_computable": f"{type(exc).__name__}: {exc}"}
    return out


def v1_off_families(rf, prices, start, end) -> dict:
    """SR1 by its gate statistic, and the declared RF variant by the criteria the 2026 read applied."""
    out = {}
    try:
        res, x = backtest.run_window(variant("SR1"), rf, prices, start, end)
        out["SR1"] = {"gate_series": {k: v for k, v in gates.variant_stats(x, start, end).items()
                                      if k in ("n_days", "mean_daily", "se_daily", "t")},
                      "per_event": episode_stats(_trades(res), prices.dates)}
    except Exception as exc:
        out["SR1"] = {"not_computable": f"{type(exc).__name__}: {exc}"}
    try:
        rs = json.loads(protocol.SELECTED.read_text())["families"]["RF"]
        c = backtest.rf_criteria(rs["chosen"], rf, prices, start, end)
        out["RF"] = {"declared": rs["chosen"], "adopted_on_dev": bool(rs["adopted"]), **c,
                     "criteria_met": bool(c["A1"] and c["A2"])}
    except Exception as exc:
        out["RF"] = {"not_computable": f"{type(exc).__name__}: {exc}"}
    return out


# ----------------------------------------------------------------------------- locks and the read

def _version(name: str) -> str | None:
    import importlib.metadata as md
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def environment() -> dict:
    """The interpreter and packages. ``python`` (major.minor) and ``packages`` are checked at the
    read; the rest is recorded. The freeze digest counts each (name, version) once."""
    import importlib.metadata as md
    import sys
    frz = sorted({f"{(d.metadata.get('Name') or '?').lower()}=={d.version}" for d in md.distributions()})
    return {"python": f"{sys.version_info[0]}.{sys.version_info[1]}", "python_full": sys.version.split()[0],
            "packages": {n: _version(n) for n in ENV_PACKAGES},
            "recorded": {n: _version(n) for n in ENV_RECORDED},
            "freeze_sha256": hashlib.sha256("\n".join(frz).encode()).hexdigest()}


def _git(*args: str, binary: bool = False):
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=not binary)
    if r.returncode != 0:
        err = r.stderr.decode(errors="replace") if binary else r.stderr
        raise HoldoutLocked(f"git {' '.join(args)} failed: {err.strip()[:300]}")
    return r.stdout


def declaration_record(code_commit: str) -> dict:
    """The content of config/trading_prospective.json (written once, right after the code commit).

    Refuses unless the working tree is clean, HEAD has ``code_commit``'s src tree, and every
    locked input on disk is the committed one, so a checkout of the declaration commit
    reproduces every hash it records."""
    code = protocol.code_state()                     # refuses dirty src, research/trading, config, tests
    tree = _git("rev-parse", f"{code_commit}:src/fearsignal").strip()
    if code["src_tree"] != tree:
        raise HoldoutLocked(f"HEAD's src/fearsignal tree {code['src_tree'][:12]} is not {code_commit[:12]}'s {tree[:12]}")
    sha = {}
    for f in LOCKED_FILES:
        if _git("status", "--porcelain", "--", f).strip():
            raise HoldoutLocked(f"{f} has uncommitted changes")
        at = hashlib.sha256(_git("show", f"{code_commit}:{f}", binary=True)).hexdigest()
        if _sha(f) != at:
            raise HoldoutLocked(f"{f} on disk differs from its content at {code_commit[:12]}")
        sha[f] = at
    sess = sample_sessions()
    return {"code_commit": code_commit, "src_tree": tree,
            "committed_inputs_sha256": sha,
            "raw_inputs_sha256": json.loads((ROOT / "config" / "trading_inputs.json").read_text())["sha256"],
            "environment": environment(),
            "sample": {"after": str(LAST_KNOWN.date()), "n_sessions": N_SESSIONS,
                       "first": str(sess[0].date()), "last": str(sess[-1].date()),
                       "note": "the last session moves one session later for each declared closure"}}


def prior_reads() -> list[str]:
    """Every trace of an earlier read: the run log, the git-ignored marker, and commits on any ref.

    Refuses if git cannot answer, so a failed query never reads as 'no earlier read'."""
    found = [_rel(f) for f in (RUNLOG, MARKER) if f.exists()]
    log = _git("log", "--all", "--format=%H", "--", "outputs/trading/prospective/").split()
    if log:
        found.append(f"commits touching outputs/trading/prospective/: {log[:3]}")
    return found


def preflight(decl: dict, code: dict, rerun_reason: str | None, env_reason: str | None) -> dict:
    """Every refusal of the read that needs no data."""
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
    want = decl["environment"]
    checked = {"python": (env["python"], want.get("python")),
               **{n: (env["packages"].get(n), want["packages"].get(n)) for n in ENV_PACKAGES}}
    diff = {k: {"now": a, "declared": b} for k, (a, b) in checked.items() if a != b}
    if diff and not env_reason:
        raise HoldoutLocked(f"environment differs from the declaration {diff}; install the declared versions, "
                            f"or pass a logged reason (--env-reason)")
    env["differs_from_declaration"] = {
        "checked": diff,
        "recorded": {n: {"now": env["recorded"].get(n), "declared": want.get("recorded", {}).get(n)}
                     for n in ENV_RECORDED if env["recorded"].get(n) != want.get("recorded", {}).get(n)},
        "freeze": env["freeze_sha256"] != want.get("freeze_sha256")}
    return {"prior_reads": prior, "environment": env}


@contextlib.contextmanager
def _exclusive(path):
    """One read at a time: an exclusive lock held from the once-only check to the end."""
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise HoldoutLocked("another prospective read is running") from None
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def live_inputs() -> dict:
    """The git-ignored files a read loads, by name, the archive included."""
    from .. import archive
    from .market_data import LIVE_PATH
    files = {"live.parquet": protocol.RAW / "live.parquet", "trade_ohlcv_live.parquet": LIVE_PATH,
             "earnings_calendar_prospective.json": EARNINGS_SAMPLE, "nd_monitor_firings.jsonl": MONITOR_LOG}
    for kind in archive.KINDS:
        d = archive.ARCHIVE / kind
        if d.exists():
            for f in sorted(d.iterdir()):
                if f.is_file():
                    files[f"archive/{kind}/{f.name}"] = f
    return files


def live_inputs_sha256() -> dict:
    return {k: _sha(f) for k, f in live_inputs().items()}


def _copy_inputs(dest, want: dict) -> dict:
    """Byte copies of the read's live inputs, each checked against the hash taken at load."""
    dest.mkdir(parents=True, exist_ok=False)
    files = live_inputs()
    if set(files) != set(want):
        raise HoldoutLocked("the set of live inputs changed while the read loaded them")
    for k, f in files.items():
        if want[k] is None:
            continue
        out = dest / k
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(f, out)
        if _sha(out) != want[k]:
            raise HoldoutLocked(f"{k} changed while the read loaded it; run the read when no refresh is running")
    return {k: dest / k for k, h in want.items() if h is not None}


def _rel(p) -> str:
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def read(rerun_reason: str | None = None, env_reason: str | None = None, closed=(),
         calendar_reason: str | None = None) -> dict:
    """The single prospective read (research/trading/PROSPECTIVE.md)."""
    from . import live
    closed = _closures(closed)
    if len(closed) and not calendar_reason:
        raise HoldoutLocked("a declared closure needs a logged reason (--calendar-reason)")
    decl_sha = frozen.committed_spec_hash(DECL)
    decl = json.loads(DECL.read_text())
    code = protocol.code_state()
    with _exclusive(READ_LOCK):
        pre = preflight(decl, code, rerun_reason, env_reason)
        raw = protocol.check_inputs()                     # the hash-locked backtest inputs (IR_post)
        logged = _persisted()                             # refuses on a corrupt firings log
        h0 = live_inputs_sha256()
        mk, st, rf, prices, report, feat = live.state(full=True)
        if live_inputs_sha256() != h0:
            raise HoldoutLocked("the live inputs changed while the read loaded them; run it when no refresh is running")
        start, end, sess = window(prices.dates, {"price cache, closes": prices.adj_close.notna().sum(axis=1),
                                                 "price cache, opens": prices.adj_open.notna().sum(axis=1)}, closed)
        elig = pd.DataFrame(feat.obs.eligible, index=feat.obs.dates, columns=feat.obs.tickers)
        window(feat.obs.dates, {"model-1 panel, eligible names": elig.sum(axis=1)}, closed)
        when = datetime.now(timezone.utc)
        copies = _copy_inputs(READ_INPUTS / f"{when:%Y%m%dT%H%M%SZ}", h0)
        stamp = {"when_utc": when.isoformat(timespec="seconds"), "head": code["head"],
                 "sample": [str(start.date()), str(end.date())], "rerun_reason": rerun_reason,
                 "env_reason": env_reason, "closed": [str(d.date()) for d in closed],
                 "calendar_reason": calendar_reason, "live_inputs_sha256": h0}
        OUT.mkdir(parents=True, exist_ok=True)
        with open(MARKER, "a") as fh:                     # logged before any outcome is computed
            fh.write(json.dumps(stamp) + "\n")
        with open(RUNLOG, "a") as fh:
            if fh.tell() == 0:
                fh.write("# Prospective read log (model 2)\n\n| when (UTC) | event | sample | head / result | "
                         "rerun reason | env reason | closures |\n|---|---|---|---|---|---|---|\n")
            fh.write(f"| {stamp['when_utc']} | read started | {start.date()}..{end.date()} | head {code['head'][:12]} "
                     f"| {rerun_reason or 'first read'} | {env_reason or ''} | "
                     f"{('closed ' + ', '.join(stamp['closed']) + ': ' + calendar_reason) if len(closed) else ''} |\n")
        spec3 = json.loads(frozen.FROZEN_V3_PATH.read_text())["stock_model"]
        cal = json.loads((protocol.OUT / "sizing_calibration.json").read_text())
        r1 = p21(rf, prices, start, end)
        r2 = p22(rf, spec3["A"], spec3["s"], cal["kappa"], cal["kappa_ewma"], start, end)
        irp = ir_post_all(rf, prices, start, end)
        status = live.family_status(apply_prospective=False)
        mon = nd_monitor(rf, prices, int(status["ND"]["tier"]), as_of=end, persist=False, logged=logged)
        last_sig = gates.last_signal_date(prices.dates, nd1_variant().H, 1, end)
        cal_copy = copies.get("earnings_calendar_prospective.json", EARNINGS_SAMPLE)
        res = {"declaration_sha256": decl_sha, "window": [str(start.date()), str(end.date())],
               "last_countable_signal": str(last_sig.date()),
               "sessions": [str(d.date()) for d in sess],
               "closed": stamp["closed"], "calendar_reason": calendar_reason,
               "P2_1_ND1": r1, "P2_2_risk": r2, "ND_IR_post": irp,
               "monitor_at_end": mon,
               "decision": decide(r1, r2, mon["tier_effective"], irp["IR_post"], mon["pending_review"]),
               "reported": {"v2_survivors": survivors(rf, prices, start, end, spec3["A"], spec3["s"], cal_copy),
                            "v1_off": v1_off_families(rf, prices, start, end)},
               "coverage": coverage_report(sess, {"price cache": prices.adj_close.notna(),
                                                  "model-1 panel": elig}),
               "code": code, "environment": pre["environment"], "raw_inputs_sha256": raw,
               "live_inputs_sha256": h0, "live_inputs_copy": {k: _rel(v) for k, v in copies.items()},
               "data": {k: v for k, v in report.items() if k != "history_bars_filled"}, "stamp": stamp}
        name = f"read_{when:%Y%m%dT%H%M%SZ}.json"
        protocol.write_json(OUT / name, res)              # every read is kept; read.json is the latest
        protocol.write_json(OUT / "read.json", res)
        with open(RUNLOG, "a") as fh:
            fh.write(f"| {datetime.now(timezone.utc).isoformat(timespec='seconds')} | read completed | "
                     f"{end.date()} | {name} | sha256 {_sha(OUT / name)[:16]} | |\n")
    return res


# ----------------------------------------------------------------------------- live monitors

def closed_nd1_events(rf, prices, as_of: pd.Timestamp | None = None, after: pd.Timestamp = LAST_KNOWN) -> pd.DataFrame:
    """ND1 events signalled after ``after`` whose 20-session window has closed by ``as_of``."""
    v = nd1_variant()
    dates = prices.dates
    as_of = pd.Timestamp(as_of) if as_of is not None else dates[-1]
    empty = pd.DataFrame(columns=["signal_date", "ticker", "excess_net"])
    later = dates[dates > pd.Timestamp(after)]
    if len(later) <= v.H + 1:
        return empty
    last = gates.last_signal_date(dates, v.H, 1, as_of)
    if last < later[0]:
        return empty
    res = registry.run(v, rf, prices, start=later[0], end=last, cost_bps=10.0, delay=1)
    if not len(res.trades):
        return empty
    return res.trades.sort_values(["signal_date", "ticker"]).reset_index(drop=True)


def monitor(trades: pd.DataFrame, dates: pd.DatetimeIndex, dev_mean: float, tier: int) -> dict:
    """The declared sequential guard and the rolling 40-trade demotion on closed events,
    each with the per-trade s.e. of the mean."""
    e = trades["excess_net"].to_numpy(float) if len(trades) else np.zeros(0)
    n = len(e)
    out = {"closed_events": int(n), "tier_declared": int(tier), "suspended_at": None, "demotions_at": [],
           "se": "per-trade s.e. of the mean"}
    for k in range(GUARD_EVERY, n + 1, GUARD_EVERY):
        s = trade_stats(e[:k])
        if s["se"] is not None and s["mean"] < dev_mean - 2.0 * s["se"]:
            out["suspended_at"] = int(k)
            break
    next_ok = DEMOTE_WINDOW
    for k in range(DEMOTE_WINDOW, n + 1):
        if k < next_ok:
            continue
        s = trade_stats(e[k - DEMOTE_WINDOW:k])
        if s["se"] is not None and s["mean"] < -s["se"]:
            out["demotions_at"].append(int(k))
            next_ok = k + DEMOTE_WINDOW                # a further demotion needs 40 more events
    out["tier_effective"] = effective_tier(tier, out["suspended_at"] is not None, len(out["demotions_at"]))
    out["running"] = {**trade_stats(e), "episode": episode_stats(trades, dates)} if n else {"n_events": 0}
    return out


def effective_tier(tier: int, suspended: bool, demotions: int) -> int:
    if suspended or demotions >= 2:
        return 0
    return max(0, int(tier) - int(demotions))


def _persisted(path=None) -> list[dict]:
    """Every logged firing. Refuses on a line it cannot read: a firing that stands must not be
    silently dropped."""
    p = path or MONITOR_LOG
    if not p.exists():
        return []
    out = []
    for i, x in enumerate(p.read_text().splitlines(), 1):
        if not x.strip():
            continue
        try:
            f = json.loads(x)
            if f["rule"] not in ("guard", "demotion") or not isinstance(f["at_event"], int):
                raise ValueError("unknown rule or event")
            pd.Timestamp(f["as_of"])
        except Exception as exc:
            raise HoldoutLocked(f"{p.name} line {i} is not a valid firing ({type(exc).__name__}: {exc}); "
                                f"restore the log from its backup") from None
        out.append(f)
    return out


def _event_index(trades: pd.DataFrame) -> dict:
    if not len(trades):
        return {}
    return {(str(pd.Timestamp(d).date()), str(t)): i + 1 for i, (d, t) in
            enumerate(zip(trades["signal_date"], trades["ticker"]))}


def _event_id(trades: pd.DataFrame, k: int) -> dict | None:
    """The k-th closed event (1-based), the one a firing at event k fired on."""
    if not 1 <= k <= len(trades):
        return None
    r = trades.iloc[k - 1]
    return {"signal_date": str(pd.Timestamp(r["signal_date"]).date()), "ticker": str(r["ticker"])}


def _position(f: dict, index: dict) -> int:
    """A logged firing's position among today's events: found by its event, else its logged index."""
    ev = f.get("event") or {}
    return int(index.get((ev.get("signal_date"), ev.get("ticker")), f["at_event"]))


def merge_firings(m: dict, trades: pd.DataFrame, logged: list[dict]) -> dict:
    """The standing state: today's recomputed firings plus every logged one. Demotions closer
    than DEMOTE_WINDOW events apart are one demotion."""
    index = _event_index(trades)
    suspended = m["suspended_at"] is not None or any(f["rule"] == "guard" for f in logged)
    pos = sorted(set(m["demotions_at"]) | {_position(f, index) for f in logged if f["rule"] == "demotion"})
    counted = []
    for p in pos:
        if not counted or p >= counted[-1] + DEMOTE_WINDOW:
            counted.append(int(p))
    return {"suspended": bool(suspended), "demotions": counted}


def _new_firings(m: dict, trades: pd.DataFrame, logged: list[dict]) -> list[dict]:
    index = _event_index(trades)
    new = []
    if m["suspended_at"] is not None and not any(f["rule"] == "guard" for f in logged):
        new.append({"rule": "guard", "at_event": int(m["suspended_at"]), "event": _event_id(trades, m["suspended_at"])})
    seen = [_position(f, index) for f in logged if f["rule"] == "demotion"]
    for k in m["demotions_at"]:
        if any(abs(k - p) < DEMOTE_WINDOW for p in seen):
            continue                      # already logged, perhaps at a shifted index
        new.append({"rule": "demotion", "at_event": int(k), "event": _event_id(trades, k)})
        seen.append(k)
    return new


def nd_monitor(rf, prices, tier: int, as_of=None, persist: bool = True, after: pd.Timestamp = LAST_KNOWN,
               logged: list[dict] | None = None) -> dict:
    """The monitors on the closed events signalled after ``after``, merged with every firing
    logged before for the same count (``after``) up to ``as_of``.

    A logged firing stands even if a later vendor revision would no longer produce it. The
    prospective read passes ``as_of`` = the sample's end, so a firing logged after the sample
    does not enter its starting tier. After the read, the live monitors count afresh from the
    sample's last countable signal."""
    sel = json.loads(protocol.SELECTED.read_text())
    dev_mean = float(sel["families"]["ND"]["chosen_stats"]["mean_event_excess"])
    after = pd.Timestamp(after)
    stamp_as_of = pd.Timestamp(as_of) if as_of is not None else prices.dates[-1]
    tr = closed_nd1_events(rf, prices, as_of, after)
    m = monitor(tr, prices.dates, dev_mean, tier)
    allf = _persisted() if logged is None else logged
    old = [f for f in allf if f.get("after", str(LAST_KNOWN.date())) == str(after.date())
           and pd.Timestamp(f["as_of"]) <= stamp_as_of]
    new = _new_firings(m, tr, old)
    if persist and new:
        MONITOR_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(MONITOR_LOG, "a") as fh:
            for f in new:
                fh.write(json.dumps({**f, "after": str(after.date()), "as_of": str(stamp_as_of.date()),
                                     "logged_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}) + "\n")
    st = merge_firings(m, tr, old)
    return {**m, "dev_mean_event_excess": dev_mean, "counts_signals_after": str(after.date()),
            "as_of": str(stamp_as_of.date()), "logged_firings": old, "new_firings": new,
            "suspended": st["suspended"], "demotions_counted": st["demotions"],
            "pending_review": bool(st["suspended"] or len(st["demotions"]) >= 2),
            "tier_effective": effective_tier(tier, st["suspended"], len(st["demotions"]))}


def apply_monitor(status: dict, mon: dict) -> dict:
    """Lower ND's live tier to the monitors' effective tier. It is never raised."""
    nd = dict(status["ND"])
    nd["monitor"] = mon
    if mon["tier_effective"] < nd.get("tier", 0):
        nd["why"] = nd.get("why", "") + f"; lowered to Tier {mon['tier_effective']} by the live monitor"
        nd["tier"] = mon["tier_effective"]
    return {**status, "ND": nd}
