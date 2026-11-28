"""H_VOL: does model 1's volatility forecast price options? (research/trading/v2/H_VOL.md)

The sample, quantities and tests are fixed in H_VOL.md. This module:
- rebuilds the sample's task list: expiries, entry sessions and strike
  candidates;
- turns the collected Robinhood bars into pairs and applies the exclusions;
- computes, per pair, IV, the total forecast sigma-tilde, the total realized
  vol RV and the net straddle return;
- runs the tests, on dev and, once, on 2026:
  - H_VOL1: OLS of log RV on [1, log IV, log sigma-tilde], with CR1 s.e.
    clustered by expiry month and a one-sided p from t(G-1);
  - H_VOL2: the difference in mean net straddle return, sigma-tilde >= IV
    minus the rest, with expiry months resampled;
  - H_VOL3: the mean net return of the sigma-tilde >= IV group;
- repeats H_VOL1 and H_VOL2 on pairs with no earnings reaction session
  during their life. That run is secondary and reported only.

Bootstrap p-values are null-centered: p = (1 + #{d* - d >= d}) / (1 + B).
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import gates, options, protocol

NAMES = ("AMD", "MU", "INTC", "QCOM", "AMAT", "ORCL", "PLTR", "CRM", "TSM", "MSFT")
FIRST_MONTH, LAST_MONTH = "2024-02", "2026-11"
ENTRY_DAYS = 28
GRIDS = (0.5, 1.0, 2.5, 5.0, 10.0)
RATE = 0.04
COST = 0.05                      # round trip, as a fraction of premium
DEV_LAST_EXPIRY = pd.Timestamp("2025-12-31")
B = 2000
ALPHA = 0.05
TASKS = protocol.RAW / "hvol_tasks.json"
PASS1 = protocol.RAW / "hvol_pass1.jsonl"       # first collection pass (nearest strike ever listed)
PASS2 = protocol.RAW / "hvol_pass2.jsonl"       # amendment 1: nearest strike listed at entry, for placeholders
LOCK = protocol.ROOT / "config" / "hvol_inputs.json"
OUT = protocol.OUT / "v2"
RUNLOG = OUT / "HVOL_RUNLOG.md"
TICK = 0.01
VENDOR_COLS = ("S", "K", "call", "put", "S_T")


# ----------------------------------------------------------------------------- sample

def expiries(sessions: pd.DatetimeIndex, first_month: str = FIRST_MONTH,
             last_month: str = LAST_MONTH) -> list[pd.Timestamp]:
    """Standard monthly expiries: the third Friday, or the session before it when it is a holiday."""
    out = []
    for m in pd.period_range(first_month, last_month, freq="M"):
        d = m.start_time
        e = d + pd.Timedelta(days=(4 - d.weekday()) % 7 + 14)
        if e not in sessions:
            e = sessions[sessions < e][-1]
        out.append(e)
    return out


def entry_session(sessions: pd.DatetimeIndex, expiry: pd.Timestamp) -> pd.Timestamp:
    """The last session at least ENTRY_DAYS calendar days before expiry."""
    return sessions[sessions <= expiry - pd.Timedelta(days=ENTRY_DAYS)][-1]


def strike_candidates(spot: float) -> list[float]:
    """round(S/g)*g for each standard strike spacing g, nearest first (lower strike on ties)."""
    ks = {round(spot / g) * g for g in GRIDS}
    return sorted(ks, key=lambda k: (abs(k - spot), k))


def build_tasks(close: pd.DataFrame, first_month: str = FIRST_MONTH, last_month: str = LAST_MONTH) -> list[dict]:
    sessions = close.index
    rows = []
    for e in expiries(sessions, first_month, last_month):
        ent = entry_session(sessions, e)
        for n in NAMES:
            s = float(close.loc[ent, n])
            rows.append({"name": n, "expiry": str(e.date()), "entry_date": str(ent.date()),
                         "spot_close": round(s, 4),
                         "strike_candidates": [f"{k:.4f}" for k in strike_candidates(s)]})
    return rows


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _num(x) -> bool:
    try:
        return x is not None and bool(np.isfinite(float(x)))
    except (TypeError, ValueError):
        return False


def is_placeholder(rec: dict) -> bool:
    """An entry bar that is gap-fill rather than a price: interpolated, missing, or at the tick floor.

    Works on JSON records and on DataFrame rows (where absent fields are NaN)."""
    if rec["status"] == "missing_bar":
        return True
    if rec["status"] != "ok":
        return False
    if any(not _num(c) or float(c) <= TICK for c in (rec.get("call_close"), rec.get("put_close"))):
        return True
    return any(rec.get(k) == True for k in ("call_interpolated", "put_interpolated"))  # noqa: E712


def merge_passes(pass1: list[dict], pass2: list[dict]) -> pd.DataFrame:
    """Amendment 1: a first-pass placeholder is replaced by its second-pass record."""
    p2 = {}
    for r in pass2:
        p2[(r["name"], r["expiry"])] = r                 # the last record wins
    out = []
    for r in pass1:
        key = (r["name"], r["expiry"])
        if is_placeholder(r):
            if key not in p2:
                raise ValueError(f"no second-pass record for placeholder {key}")
            out.append({**p2[key], "source": "pass2", "first_pass_strike": r["strike"]})
        else:
            out.append({**r, "source": "pass1"})
    extra = set(p2) - {(r["name"], r["expiry"]) for r in pass1 if is_placeholder(r)}
    if extra:
        raise ValueError(f"second-pass records for pairs that were not placeholders: {sorted(extra)}")
    df = pd.DataFrame(out)
    if df.duplicated(["name", "expiry"]).any():
        raise ValueError("duplicate first-pass records")
    return df


def load_bars() -> pd.DataFrame:
    return merge_passes(_jsonl(PASS1), _jsonl(PASS2))


def check_inputs() -> dict:
    """Refuse unless the option inputs match the committed lock (config/hvol_inputs.json)."""
    import hashlib

    from .. import frozen
    from ..config import HoldoutLocked
    frozen.committed_spec_hash(LOCK)
    lock = json.loads(LOCK.read_text())["sha256"]
    now = {f: hashlib.sha256((protocol.RAW / f).read_bytes()).hexdigest() for f in lock}
    bad = sorted(k for k in lock if lock[k] != now[k])
    if bad or "hvol_pass2.jsonl" not in lock:
        raise HoldoutLocked(f"H_VOL inputs differ from config/hvol_inputs.json or are incomplete: {bad}")
    return now


# ----------------------------------------------------------------------------- pairs

def straddle_net(spot_T: float, K: float, call: float, put: float, cost: float = COST) -> float:
    prem = call + put
    return (abs(spot_T - K) - prem) / prem - cost


def pairs(bars: pd.DataFrame, close: pd.DataFrame, r: pd.DataFrame, stilde, kappa20: float,
          earnings: pd.DataFrame | None) -> tuple[pd.DataFrame, dict]:
    """One row per usable pair, and the exclusion counts.

    ``stilde(h)`` returns the unscaled sigma-tilde panel for horizon h. ``r`` holds
    daily log returns and ``close`` raw closes; both are indexed by session."""
    sessions = close.index
    rows, excl = [], {}

    def drop(why):
        excl[why] = excl.get(why, 0) + 1

    for b in bars.itertuples(index=False):
        if b.status != "ok":
            drop(b.status)
            continue
        if is_placeholder(b._asdict()):
            drop("placeholder")
            continue
        e, ent, n = pd.Timestamp(b.expiry), pd.Timestamp(b.entry_date), b.name
        S, K, C, P = float(close.loc[ent, n]), float(b.strike), float(b.call_close), float(b.put_close)
        if abs(S - float(b.spot_close)) > 1e-3 * S:
            raise ValueError(f"spot mismatch for {n} {b.expiry}: {S} vs {b.spot_close}")
        T = (e - ent).days / options.YEAR
        ivc = options.implied_vol(C, S, K, T, "call", r=RATE)
        ivp = options.implied_vol(P, S, K, T, "put", r=RATE)
        if not (np.isfinite(ivc) and np.isfinite(ivp)):
            drop("no_arbitrage")
            continue
        i0, i1 = sessions.get_loc(ent), sessions.get_loc(e)
        h = int(i1 - i0)
        st = kappa20 * float(stilde(h).loc[ent, n])
        life = r[n].iloc[i0 + 1: i1 + 1]
        if not np.isfinite(st) or life.isna().any() or len(life) != h:
            drop("no_forecast_or_returns")
            continue
        iv = 0.5 * (ivc + ivp)
        ST = float(close.loc[e, n])
        earn = bool(earnings[n].iloc[i0 + 1: i1 + 1].any()) if earnings is not None else False
        rows.append({"name": n, "expiry": e, "entry": ent, "month": str(e.to_period("M")),
                     "period": "dev" if e <= DEV_LAST_EXPIRY else "2026", "S": S, "K": K,
                     "call": C, "put": P, "T": T, "h": h, "iv_call": ivc, "iv_put": ivp, "iv": iv,
                     "iv_tot": iv * math.sqrt(T), "st_tot": st, "rv_tot": float(np.sqrt((life ** 2).sum())),
                     "S_T": ST, "net": straddle_net(ST, K, C, P), "hi": bool(st >= iv * math.sqrt(T)),
                     "earnings_in_life": earn,
                     "parity_resid": ((C - P) - (S - K * math.exp(-RATE * T))) / (C + P)})
    return pd.DataFrame(rows), excl


# ----------------------------------------------------------------------------- tests

def h_vol1(df: pd.DataFrame) -> dict:
    y = np.log(df["rv_tot"].to_numpy(float))
    X = np.column_stack([np.ones(len(df)), np.log(df["iv_tot"].to_numpy(float)),
                         np.log(df["st_tot"].to_numpy(float))])
    r = gates.ols_cluster(y, X, df["month"].to_numpy())
    G = r["clusters"]
    t_c = float(r["t"][2])
    return {"n": int(r["n"]), "clusters": G, "a": float(r["b"][0]), "b_log_iv": float(r["b"][1]),
            "c_log_stilde": float(r["b"][2]), "se_c": float(r["se"][2]), "t_c": t_c,
            "p_one_sided": float(stats.t.sf(t_c, G - 1)), "t_b": float(r["t"][1])}


def _month_sums(df: pd.DataFrame):
    g = df.groupby("month")
    hi = df[df["hi"]].groupby("month")["net"]
    lo = df[~df["hi"]].groupby("month")["net"]
    months = sorted(g.groups)
    f = lambda s: s.reindex(months).fillna(0.0).to_numpy(float)
    return months, f(hi.sum()), f(hi.count()), f(lo.sum()), f(lo.count())


def _draws(G: int, B: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.multinomial(G, np.full(G, 1.0 / G), size=B).astype(float)


def h_vol2(df: pd.DataFrame, B: int = B, seed: int = 0) -> dict:
    months, s_hi, n_hi, s_lo, n_lo = _month_sums(df)
    d0 = s_hi.sum() / n_hi.sum() - s_lo.sum() / n_lo.sum() if n_hi.sum() and n_lo.sum() else float("nan")
    W = _draws(len(months), B, seed)
    with np.errstate(invalid="ignore", divide="ignore"):
        d = (W @ s_hi) / (W @ n_hi) - (W @ s_lo) / (W @ n_lo)
    ok = np.isfinite(d)
    p = (1 + np.sum(d[ok] - d0 >= d0)) / (1 + ok.sum()) if np.isfinite(d0) else float("nan")
    return {"n_hi": int(n_hi.sum()), "n_lo": int(n_lo.sum()), "months": len(months),
            "mean_hi": float(s_hi.sum() / n_hi.sum()) if n_hi.sum() else float("nan"),
            "mean_lo": float(s_lo.sum() / n_lo.sum()) if n_lo.sum() else float("nan"),
            "diff": float(d0), "p_one_sided": float(p), "draws_used": int(ok.sum()),
            "ci95": [float(np.quantile(d[ok], 0.025)), float(np.quantile(d[ok], 0.975))] if ok.any() else None}


def h_vol3(df: pd.DataFrame, B: int = B, seed: int = 0) -> dict:
    months, s_hi, n_hi, _, _ = _month_sums(df)
    if not n_hi.sum():
        return {"n_hi": 0}
    m0 = s_hi.sum() / n_hi.sum()
    W = _draws(len(months), B, seed)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = (W @ s_hi) / (W @ n_hi)
    ok = np.isfinite(m)
    return {"n_hi": int(n_hi.sum()), "mean_net": float(m0),
            "p_one_sided": float((1 + np.sum(m[ok] - m0 >= m0)) / (1 + ok.sum())),
            "ci95": [float(np.quantile(m[ok], 0.025)), float(np.quantile(m[ok], 0.975))]}


def describe(df: pd.DataFrame) -> dict:
    return {"n": int(len(df)), "names": int(df["name"].nunique()), "months": int(df["month"].nunique()),
            "share_hi": float(df["hi"].mean()), "share_earnings_in_life": float(df["earnings_in_life"].mean()),
            "median_iv": float(df["iv"].median()),
            "mean_log_rv_over_iv": float(np.log(df["rv_tot"] / df["iv_tot"]).mean()),
            "mean_log_st_over_iv": float(np.log(df["st_tot"] / df["iv_tot"]).mean()),
            "mean_log_rv_over_st": float(np.log(df["rv_tot"] / df["st_tot"]).mean()),
            "mean_net_all": float(df["net"].mean()),
            "median_abs_parity_resid": float(df["parity_resid"].abs().median())}


def analyse(df: pd.DataFrame, seed: int = 0) -> dict:
    out = {"describe": describe(df), "H_VOL1": h_vol1(df), "H_VOL2": h_vol2(df, seed=seed),
           "H_VOL3": h_vol3(df, seed=seed)}
    ne = df[~df["earnings_in_life"]]
    out["secondary_no_earnings"] = ({"describe": describe(ne), "H_VOL1": h_vol1(ne), "H_VOL2": h_vol2(ne, seed=seed)}
                                    if len(ne) >= 20 and ne["month"].nunique() >= 5 and ne["hi"].any()
                                    and (~ne["hi"]).any() else {"n": int(len(ne)), "note": "too few pairs"})
    return out


def _sha256(path) -> str | None:
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest() if Path(path).exists() else None


def posthoc_qlike(dev: pd.DataFrame, y26: pd.DataFrame) -> dict:
    """POST HOC, not pre-registered (H_VOL.md): QLIKE of each forecast's variance against RV at the
    option horizon, and its month-clustered mean differential against raw IV.

    'IV x dev level' scales IV by sqrt(sum RV^2 / sum IV^2) fitted on the dev pairs, so it is
    in-sample on dev."""
    ql = lambda f, y: y ** 2 / f ** 2 - np.log(y ** 2 / f ** 2) - 1
    k_iv = float(np.sqrt((dev["rv_tot"] ** 2).sum() / (dev["iv_tot"] ** 2).sum()))
    out = {"note": "post hoc; not pre-registered; differentials clustered by expiry month", "iv_dev_level": k_iv}
    for name, x in (("dev", dev), ("2026", y26)):
        y = x["rv_tot"]
        base = ql(x["iv_tot"], y)
        fc = {"sigma_tilde": x["st_tot"], "iv_x_dev_level": k_iv * x["iv_tot"],
              "geometric_mean_sigma_tilde_iv": np.sqrt(x["st_tot"] * x["iv_tot"])}
        row = {"iv": {"qlike": float(base.mean())}}
        for k, f in fc.items():
            l = ql(f, y)
            dm = (l - base).groupby(x["month"]).mean()
            row[k] = {"qlike": float(l.mean()), "diff_vs_iv": float(dm.mean()),
                      "t": float(dm.mean() / (dm.std(ddof=1) / np.sqrt(len(dm)))), "months": int(len(dm))}
        out[name] = row
    return out


def posthoc_main() -> dict:
    d = {p: pd.read_csv(OUT / f"hvol_pairs_{p}.csv") for p in ("dev", "2026")}
    res = posthoc_qlike(d["dev"], d["2026"])
    protocol.write_json(OUT / "hvol_posthoc_qlike.json", res)
    return res


# ----------------------------------------------------------------------------- driver

def inputs():
    """Raw closes, daily log returns, the sigma-tilde panels, kappa_20 and the earnings mask."""
    from . import explore as X, market_data as md
    ctx = X.Ctx()
    close = md.load()["close"]
    kappa20 = float(json.loads((protocol.OUT / "sizing_calibration.json").read_text())["kappa"]["20"])
    r = ctx.rf.r.reindex(close.index)

    def stilde(h):
        return ctx.stilde(h).reindex(close.index)

    earn = X.earnings_sessions(ctx).reindex(close.index).fillna(False).astype(bool)
    return close, r, stilde, kappa20, earn


def main(period: str, rerun_reason: str | None = None) -> dict:
    """Read one period, 'dev' or '2026'. Each read runs once; its start is logged first."""
    if period not in ("dev", "2026"):
        raise ValueError(period)
    out_path = OUT / f"hvol_{period}.json"
    if out_path.exists() and not rerun_reason:
        raise RuntimeError(f"the H_VOL {period} read has already run; pass a rerun reason")
    if period == "2026" and not (OUT / "hvol_dev.json").exists():
        raise RuntimeError("read dev first")
    hashes = check_inputs()
    code = protocol.code_state()            # refuses a dirty tree before the start is logged
    manifest = protocol.input_manifest()
    with open(RUNLOG, "a") as fh:
        fh.write(f"| {datetime.now(timezone.utc):%Y-%m-%d %H:%M} | {period} read started | {rerun_reason or 'first read'} |\n")
    close, r, stilde, kappa20, earn = inputs()
    tasks = json.loads(TASKS.read_text())
    if tasks != build_tasks(close):
        raise RuntimeError("the committed task list no longer matches the sample definition")
    bars = load_bars()
    got = set(zip(bars["name"], bars["expiry"]))
    missing = [t for t in tasks if (t["name"], t["expiry"]) not in got]
    in_period = lambda e: (pd.Timestamp(e) <= DEV_LAST_EXPIRY) == (period == "dev")
    # exclusions are counted for this period only (review HV-4)
    df, excl = pairs(bars[[in_period(e) for e in bars["expiry"]]].reset_index(drop=True), close, r, stilde,
                     kappa20, earn)
    sel = df[df["period"] == period].reset_index(drop=True)
    excl_p = {}
    for b in bars.itertuples(index=False):
        if in_period(b.expiry) and b.status != "ok":
            excl_p[b.status] = excl_p.get(b.status, 0) + 1
    n_task = sum(in_period(t["expiry"]) for t in tasks)
    src = bars[[in_period(e) for e in bars["expiry"]]]["source"].value_counts().to_dict()
    res = {"period": period, "n_tasks": n_task, "n_missing_records": sum(in_period(t["expiry"]) for t in missing),
           "records_by_source": {k: int(v) for k, v in src.items()},
           "excluded_by_status": excl_p,
           "excluded_after_collection": {k: v for k, v in excl.items()
                                         if k not in ("missing_bar", "no_contract", "error", "not_listed_at_entry")},
           "n_used": int(len(sel)), "kappa20": kappa20, "rate": RATE, "cost": COST, "inputs": hashes,
           # provenance (review HV-7): every input the read used, and the code it ran
           "price_and_model_inputs": manifest, "code": code,
           "earnings_calendar_sha256": _sha256(protocol.RAW / "earnings_calendar.json"),
           "bootstrap": {"B": B, "seed": 0, "note": "draws with an empty group are dropped (review HV-8)"},
           **analyse(sel)}
    h2 = res["H_VOL2"]
    if period == "dev":
        res["decision"] = {"H_VOL1_claimed": bool(res["H_VOL1"]["p_one_sided"] < ALPHA),
                           "H_VOL2_holds_on_dev": bool(h2["p_one_sided"] < ALPHA)}
    else:
        dev = json.loads((OUT / "hvol_dev.json").read_text())
        res["decision"] = {"H_VOL2_holds_on_dev": dev["decision"]["H_VOL2_holds_on_dev"],
                           "diff_2026_positive": bool(h2["diff"] > 0),
                           "enters_prospective_as_tier1": bool(dev["decision"]["H_VOL2_holds_on_dev"] and h2["diff"] > 0)}
    protocol.write_json(out_path, res)
    # derived quantities only: vendor prices (stock and option closes, strikes) are never committed
    keep = [c for c in sel.columns if c not in VENDOR_COLS]
    sel[keep].assign(expiry=sel["expiry"].dt.date.astype(str), entry=sel["entry"].dt.date.astype(str)).to_csv(
        OUT / f"hvol_pairs_{period}.csv", index=False)
    with open(RUNLOG, "a") as fh:
        fh.write(f"| {datetime.now(timezone.utc):%Y-%m-%d %H:%M} | {period} read completed | n={len(sel)} |\n")
    return res
