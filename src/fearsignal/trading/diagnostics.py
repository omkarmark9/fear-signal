"""Candidate diagnostics for v2 event trials (research/trading/v2/PLAN.md, selection rule).

For an event trial on a window:
- **placebo percentile:** the trial's 1/H mean against random names drawn
  on the same days in the same numbers, with the same beta benchmark;
- **intercept:** OLS of event excess on [1, R_B] with CR1 errors by
  episode. It guards against mis-estimated beta in a trending market;
- **plain basket:** the same rule with beta = 1;
- **by year:** the 1/H mean and t per calendar year;
- **leave-one-episode-out:** the minimum t of the 1/H mean, and the
  per-event mean after dropping the top 5 and top 10 events;
- **side split** for either-sign rules: v >= 0 against v < 0 at the
  signal.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from . import engine, explore as X, gates, placebo, signals


def _signals(tr: X.Trial, ctx: X.Ctx):
    """Exactly the signals run_event uses, including per-name weights."""
    mask, prio, H, cap, side, delay = X.event_rule(tr, ctx)
    sel = signals.suppress_reentry(mask, H, priority=prio, cap=cap)
    w = 1.0 / ctx.stilde(H) if tr.params.get("weight") == "invvol" else None
    sig = signals.to_signals(sel, w)
    if w is not None:
        sig = sig[np.isfinite(sig["weight"]) & (sig["weight"] > 0)]
    return sig, H, side, delay


def _run(sig, ctx, H, side, delay, start, end, beta=True, cost=10.0):
    last = gates.last_signal_date(ctx.prices.dates, H, delay, end)
    res = engine.run_cohorts(sig, ctx.prices, H=H, cost_bps=cost, delay=delay, side=side, start=start,
                             end=last, cache=ctx.cache, beta=ctx.rf.beta if beta else None)
    return res, gates.window(res.daily["excess_net"], start, end)


def event_diagnostics(tr: X.Trial, ctx: X.Ctx, start, end, n_placebo: int = 200, seed: int = 0) -> dict:
    sig, H, side, delay = _signals(tr, ctx)
    res, x = _run(sig, ctx, H, side, delay, start, end)
    out = {"id": tr.id, "mean": float(x.mean()), "t": gates.variant_stats(x, start, end)["t"]}
    tr_ = res.trades.copy()
    if len(tr_) < 5:
        return {**out, "note": "fewer than 5 events"}
    cl = X._episodes(tr_["signal_date"], ctx.prices.dates)
    e = tr_["excess_net"].to_numpy(float)
    X1 = np.column_stack([np.ones(len(e)), tr_["basket_ret"].to_numpy(float)])
    r = gates.ols_cluster(e, X1, cl)
    out["intercept"] = {"a": float(r["b"][0]), "t": float(r["t"][0]), "b_on_basket": float(r["b"][1])}
    _, xp = _run(sig, ctx, H, side, delay, start, end, beta=False)
    out["plain_basket"] = {"mean": float(xp.mean()), "t": gates.variant_stats(xp, start, end)["t"]}
    yrs = {}
    for y in sorted(set(x.index.year)):
        xs = x[x.index.year == y]
        yrs[int(y)] = {"mean": float(xs.mean()), "t": gates.variant_stats(xs, xs.index[0], xs.index[-1])["t"]}
    out["by_year"] = yrs
    loo = []
    for c in np.unique(cl):
        keep = sig.merge(tr_[["signal_date", "ticker"]].assign(cl=cl), left_on=["date", "ticker"],
                         right_on=["signal_date", "ticker"], how="left")
        s2 = sig[(keep["cl"] != c).to_numpy()]
        _, xl = _run(s2, ctx, H, side, delay, start, end)
        loo.append(gates.variant_stats(xl, start, end)["t"])
    out["leave_one_episode_out_min_t"] = float(np.min(loo))
    out["n_episodes"] = int(len(np.unique(cl)))
    srt = np.sort(e)[::-1]
    out["event_mean"] = float(e.mean())
    out["event_mean_drop_top5"] = float(srt[5:].mean())
    out["event_mean_drop_top10"] = float(srt[10:].mean())
    out["event_median"] = float(np.median(e))
    v = ctx.rf.v.reindex(index=pd.to_datetime(tr_["signal_date"])).to_numpy()[
        np.arange(len(tr_)), [ctx.rf.v.columns.get_loc(t) for t in tr_["ticker"]]]
    if np.any(v >= 0) and np.any(v < 0):
        out["side_split"] = {"v_ge_0": {"n": int((v >= 0).sum()), "event_mean": float(e[v >= 0].mean())},
                             "v_lt_0": {"n": int((v < 0).sum()), "event_mean": float(e[v < 0].mean())}}
    rng = np.random.default_rng(seed)
    pm = []
    for _ in range(n_placebo):
        ps = placebo.placebo_signals(sig, ctx.prices, delay, rng)
        _, xq = _run(ps, ctx, H, side, delay, start, end)
        pm.append(float(xq.mean()))
    pm = np.array(pm)
    out["placebo"] = {"percentile": float(np.mean(pm < out["mean"])), "mean": float(pm.mean()),
                      "sd": float(pm.std()), "n": n_placebo}
    return out
