"""Dev selection and the single holdout read, as declared in research/trading/candidates.md.

dev_selection
1. Each variant runs on 2024-01-02 .. 2025-12-31. Signals stop early
   enough that every cohort exits by 2025-12-31.
2. Statistics come from the variant's declared daily excess series.
3. Within a family: a calibrated SPA_c test across variants, then the
   1-SE choice.
4. Across Holm families: Holm step-down at family-wise 0.10, each
   threshold capped at 0.05.
5. The chosen variant must:
   (a) have t >= 1;
   (c) show its family's declared signature;
   (d) keep a positive mean at 25 bps per side and with a one-day delay.
6. A family that passes everything is adopted. The selection is written to
   config/trading_selected.json and committed before the holdout is read.

holdout_read: the selected variants on 2026-01-02 .. 2026-11-27, once.
"""

from __future__ import annotations

from typing import Callable

import pandas as pd

from . import engine, gates, protocol, registry, signals

STRESS_BPS = 25.0
DELAY_CHECK = 2


def _series(v: registry.Variant, res: engine.Result, start, end) -> pd.Series:
    return gates.window(res.daily[v.series], start, end)


def run_window(v: registry.Variant, rf: signals.RuleFeatures, prices: engine.Prices, start, end,
               cost_bps: float = 10.0, delay: int = 1, cache: dict | None = None,
               sig: pd.DataFrame | None = None) -> tuple[engine.Result, pd.Series]:
    """Run a variant with signals in [start, last date whose cohort exits by end]."""
    last = gates.last_signal_date(prices.dates, v.H, delay, end)
    res = registry.run(v, rf, prices, start=start, end=last, cost_bps=cost_bps, delay=delay,
                       cache=cache, sig=sig)
    return res, _series(v, res, start, end)


def evaluate_family(fam: registry.Family, rf, prices, start, end, cache: dict,
                    signature: Callable | None, B: int, R: int, seed: int) -> dict:
    cols, stats, runs, sigs = {}, [], {}, {}
    for v in fam.variants:
        sig = registry.entries(v, rf)
        res, x = run_window(v, rf, prices, start, end, cache=cache, sig=sig)
        cols[v.name] = x
        st = gates.variant_stats(x, start, end)
        st.update(n_events=int(len(res.trades)), n_signal_days=int(res.cohorts["signal_date"].nunique())
                  if len(res.cohorts) else 0,
                  mean_event_excess=float(res.trades["excess_net"].mean()) if len(res.trades) else None)
        stats.append(st)
        runs[v.name], sigs[v.name] = res, sig
    D = pd.DataFrame(cols).fillna(0.0)
    sp = gates.family_spa(D, B=B, R=R, seed=seed)
    k = gates.one_se_choice(stats)
    v = fam.variants[k]
    _, x25 = run_window(v, rf, prices, start, end, cost_bps=STRESS_BPS, cache=cache, sig=sigs[v.name])
    _, xd2 = run_window(v, rf, prices, start, end, delay=DELAY_CHECK, cache=cache, sig=sigs[v.name])
    robust = {"mean_daily_25bps": float(x25.mean()), "mean_daily_delay2": float(xd2.mean()),
              "pass": bool(x25.mean() > 0 and xd2.mean() > 0)}
    sig_out = signature(fam, v, rf, prices, start, end, cache, runs) if signature else {"pass": True}
    return {"title": fam.title, "holm_family": fam.holm,
            "variants": {fv.name: s for fv, s in zip(fam.variants, stats)},
            "spa": sp, "chosen": v.name, "chosen_t": stats[k]["t"], "robustness": robust,
            "signature": sig_out, "daily": D}


def dev_selection(families: list[registry.Family], signatures: dict, rf, prices,
                  B: int = 999, R: int = 999, seed: int = 0) -> dict:
    start, end = gates.DEV
    cache: dict = {}
    fams = {f.id: evaluate_family(f, rf, prices, start, end, cache, signatures.get(f.id), B, R, seed + i)
            for i, f in enumerate(families)}
    holm = gates.holm({k: v["spa"]["p_c_calibrated"] for k, v in fams.items() if v["holm_family"]})
    for k, f in fams.items():
        h = holm.get(k)
        f["holm"] = h
        f["gates"] = {"a_t_ge_1": bool(f["chosen_t"] >= 1.0),
                      "b_spa_holm": bool(h["pass"]) if h else False,
                      "c_signature": bool(f["signature"].get("pass", False)),
                      "d_robustness": bool(f["robustness"]["pass"])}
        f["adopted"] = all(f["gates"].values())
    return {"window": [str(start.date()), str(end.date())], "families": fams,
            "inputs": protocol.input_manifest()}


def holdout_read(selected: list[registry.Variant], rf, prices) -> dict:
    start, end = gates.HOLDOUT
    cache: dict = {}
    out = {}
    for v in selected:
        res, x = run_window(v, rf, prices, start, end, cache=cache)
        st = gates.variant_stats(x, start, end)
        st.update(n_events=int(len(res.trades)),
                  mean_event_excess=float(res.trades["excess_net"].mean()) if len(res.trades) else None)
        out[v.id] = {"stats": st, "agrees": bool(st["mean_daily"] > 0), "t_ge_1": bool(st["t"] >= 1.0),
                     "daily": x, "trades": res.trades}
    return out


def strip(o):
    """Drop frames and series before JSON."""
    if isinstance(o, dict):
        return {k: strip(v) for k, v in o.items() if not isinstance(v, (pd.DataFrame, pd.Series))}
    return o



def same_stock_baseline(rf: signals.RuleFeatures, prices: engine.Prices, H: int, start, end,
                        delay: int = 1, beta: bool = True) -> pd.Series:
    """Each stock's mean H-session excess (R_i - beta_i R_B) over EVERY signal day in
    [start, last day whose window exits by end]: the same-stock baseline that strips
    the hindsight drift of the universe's winners from an event's excess.
    Windows are open(s+delay) to open(s+delay+H), as in the engine."""
    import numpy as np
    dates = prices.dates
    ao = prices.adj_open.to_numpy(float)
    ok = engine.tradable(prices)
    B = rf.beta.reindex(index=dates, columns=prices.adj_close.columns).to_numpy(float)
    last = gates.last_signal_date(dates, H, delay, end)
    S = np.flatnonzero((dates >= pd.Timestamp(start)) & (dates <= last))
    tot = np.zeros(ao.shape[1])
    cnt = np.zeros(ao.shape[1])
    for s in S:
        e, x = s + delay, s + delay + H
        can = ok[s] & np.isfinite(ao[e]) & np.isfinite(ao[x])
        if not can.any():
            continue
        R = ao[x] / ao[e] - 1
        RB = np.nanmean(np.where(can, R, np.nan))
        b = np.nan_to_num(B[s], nan=1.0) if beta else np.ones_like(R)
        ex = np.where(can, R - b * RB, np.nan)
        m = np.isfinite(ex)
        tot[m] += ex[m]
        cnt[m] += 1
    return pd.Series(np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan), index=prices.adj_close.columns)
