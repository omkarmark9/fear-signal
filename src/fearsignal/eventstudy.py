"""Exploratory event study: P/L from buying at extreme-fear points.

This is descriptive, not a test. The pre-registered return test (H5) found no
predictability. Everything here was fixed before any P/L was computed:

* Signal: the v2 frozen spec (fit on 2023-2025), run causally. 2023-2025 is
  in-sample for it; 2026 is out-of-sample.
* Market episode: the first session with M >= threshold after at least
  ``GAP`` sessions below it. Thresholds 2.0 (acute fear or worse), 2.5
  (panic), and high conviction (M_lo >= 2).
* Market trade: buy an equal-dollar basket of every eligible AI stock at that
  session's close and hold H sessions (buy-and-hold, no rebalancing, no
  costs). H = 5, 20 (primary, the H5 horizon) and 60.
* Stock episode: a stock's first Z >= 2.5 (panic band) after ``GAP``
  sessions without one. Buy that stock at the close and hold H sessions.
* Benchmark: the same trade entered on every eligible session of the same
  period (stock trades: the same stock on every one of its sessions). An edge
  is the difference between the two.
* Windows that run past the last available close are reported as open.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

GAP = 10
HORIZONS = (5, 20, 60)


def episodes(flag: pd.Series, gap: int = GAP) -> list[pd.Timestamp]:
    """First True of each run, needing ``gap`` sessions without True before it."""
    f = flag.fillna(False).to_numpy(bool)
    out, last = [], -10**9
    for i, v in enumerate(f):
        if v:
            if i - last > gap:
                out.append(flag.index[i])
            last = i
    return out


def forward_return(px: pd.DataFrame, entry: pd.Timestamp, h: int,
                   tickers: list[str] | None = None) -> tuple[float, int, bool]:
    """Equal-weight buy-and-hold return from the close of ``entry`` for h sessions.

    Returns (return, number of stocks, complete). An incomplete window uses
    the last available close and is flagged.
    """
    i = px.index.get_loc(entry)
    j = min(i + h, len(px) - 1)
    cols = tickers if tickers is not None else px.columns
    a, b = px.iloc[i][cols], px.iloc[j][cols]
    ok = a.notna() & b.notna()
    r = (b[ok] / a[ok] - 1).mean() if ok.any() else np.nan
    return float(r), int(ok.sum()), i + h <= len(px) - 1


def unconditional(px: pd.DataFrame, days: pd.DatetimeIndex, h: int,
                  tickers: list[str] | None = None) -> float:
    """Mean of the same trade entered on every day in ``days`` with a full window."""
    vals = [forward_return(px, d, h, tickers) for d in days]
    return float(np.mean([v for v, n, full in vals if full and n > 0]))


def market_study(px: pd.DataFrame, mk: pd.DataFrame, periods: dict, stake: float = 10_000.0) -> dict:
    rules = {"M>=2 (acute fear+)": mk["M"] >= 2.0,
             "M>=2.5 (panic)": mk["M"] >= 2.5,
             "high conviction (M_lo>=2)": mk["M_lo"] >= 2.0}
    out = {}
    for pname, (lo, hi) in periods.items():
        days = mk.index[(mk.index >= lo) & (mk.index <= hi) & mk["M"].notna()]
        base = {h: unconditional(px, days, h) for h in HORIZONS}
        res = {"benchmark_every_day": base, "rules": {}}
        for rname, flag in rules.items():
            ev = [d for d in episodes(flag) if lo <= d <= hi]
            rows = []
            for d in ev:
                row = {"entry": str(d.date()), "M": float(mk.loc[d, "M"])}
                for h in HORIZONS:
                    r, n, full = forward_return(px, d, h)
                    row[f"r{h}"] = r
                    row[f"pl{h}"] = r * stake
                    row[f"full{h}"] = full
                rows.append(row)
            summ = {}
            for h in HORIZONS:
                rr = np.array([x[f"r{h}"] for x in rows if x[f"full{h}"]])
                if len(rr):
                    summ[h] = {"n": int(len(rr)), "mean": float(rr.mean()), "median": float(np.median(rr)),
                               "hit_rate": float((rr > 0).mean()), "min": float(rr.min()), "max": float(rr.max()),
                               "mean_minus_benchmark": float(rr.mean() - base[h]),
                               "beat_benchmark_rate": float((rr > base[h]).mean())}
            res["rules"][rname] = {"episodes": rows, "summary": summ}
        out[pname] = res
    return out


def stock_study(px: pd.DataFrame, st: pd.DataFrame, periods: dict, stake: float = 1_000.0) -> dict:
    Z = st.pivot(index="date", columns="ticker", values="Z").reindex(px.index)
    out = {}
    for pname, (lo, hi) in periods.items():
        rows = []
        for t in Z.columns:
            for d in episodes(Z[t] >= 2.5):
                if not (lo <= d <= hi):
                    continue
                row = {"ticker": t, "entry": str(d.date())}
                for h in HORIZONS:
                    r, n, full = forward_return(px, d, h, [t])
                    row[f"r{h}"], row[f"full{h}"] = r, full
                rows.append(row)
        df = pd.DataFrame(rows)
        summ = {}
        for h in HORIZONS:
            done = df[df[f"full{h}"] & df[f"r{h}"].notna()]
            if done.empty:
                continue
            # the same stock's unconditional h-day return over the period's eligible days
            base = {}
            for t in done["ticker"].unique():
                days = Z.index[(Z.index >= lo) & (Z.index <= hi) & Z[t].notna()]
                base[t] = unconditional(px, days, h, [t])
            excess = done[f"r{h}"] - done["ticker"].map(base)
            # dates cluster (one market selloff puts many stocks in panic at once):
            # average within entry date first, then across dates
            by_date = done.assign(ex=excess).groupby("entry")[[f"r{h}", "ex"]].mean().sort_index()
            ci = _date_bootstrap_ci(by_date["ex"].to_numpy())
            summ[h] = {"n_trades": int(len(done)), "n_entry_dates": int(len(by_date)),
                       "mean": float(done[f"r{h}"].mean()), "median": float(done[f"r{h}"].median()),
                       "hit_rate": float((done[f"r{h}"] > 0).mean()),
                       "mean_excess_vs_same_stock": float(excess.mean()),
                       "mean_excess_date_weighted": float(by_date["ex"].mean()),
                       "share_dates_with_positive_excess": float((by_date["ex"] > 0).mean()),
                       "date_weighted_excess_ci95": ci,
                       "total_pl": float(done[f"r{h}"].sum() * stake)}
        out[pname] = {"summary": summ, "n_episodes": int(len(df))}
    return out


def _date_bootstrap_ci(x: np.ndarray, block: float = 5.0, n_boot: int = 4000, seed: int = 0) -> list[float]:
    """95% stationary-bootstrap interval for the mean of date-level excess returns.

    The entry dates' holding windows overlap, so even this is likely too narrow
    for 60-session windows.
    """
    from .stats import stationary_bootstrap_indices

    rng = np.random.default_rng(seed)
    boots = [x[stationary_bootstrap_indices(len(x), block, rng)].mean() for _ in range(n_boot)]
    return [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]
