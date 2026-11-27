"""Placebo calibration of the trading model's selection gate.

Question: if no rule had any stock-selection skill, how often would the
gate adopt one anyway? The placebo keeps each variant's real signal TIMING:
the same signal days and the same number of names per day. It replaces
WHICH names with a uniform random draw from the names the basket holds that
day. Excess over the basket then has mean exactly zero at zero cost (the
boundary of the null). It keeps the real dependence structure, including
event clustering and high cross-sectional dispersion on systemic days. The
gate is run on these placebo families exactly as on the real ones. The
share of replications in which any family is adopted is the gate's actual
family-wise false-adoption rate.

No real strategy's outcome is used. Only the signals' timing and counts,
and the prices that form the null world, enter.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import engine, gates


def placebo_signals(sig: pd.DataFrame, prices: engine.Prices, delay: int, rng) -> pd.DataFrame:
    """Same dates and per-day counts as ``sig``; names drawn uniformly from the
    basket's universe that day (tradable at t, with an open at t+delay)."""
    ok = engine.tradable(prices)
    ao = prices.adj_open.to_numpy(float)
    cols = prices.adj_close.columns
    pos = {d: i for i, d in enumerate(prices.dates)}
    T = len(prices.dates)
    rows = []
    counts = sig.groupby(pd.to_datetime(sig["date"])).size()
    for d, k in counts.items():
        t = pos.get(d)
        if t is None or t + delay >= T:
            continue
        univ = np.flatnonzero(ok[t] & np.isfinite(ao[t + delay]))
        if univ.size == 0:
            continue
        pick = rng.choice(univ, size=min(int(k), univ.size), replace=False)
        rows += [(d, cols[j]) for j in pick]
    return pd.DataFrame(rows, columns=["date", "ticker"]).assign(weight=1.0)


def gate_once(families: list[dict], prices: engine.Prices, rng, start, end, B: int = 199, R: int = 199,
              cache: dict | None = None) -> dict:
    """One placebo replication of the dev gate.

    families: [{"id", "side", "variants": [{"name", "signals", "H", "delay", "series"}]}],
    where ``signals`` are the variant's real signal rows (timing only is used) and
    ``series`` is the daily excess series its gate reads.
    Returns per-family SPA p, the Holm outcome, and whether each family would be adopted
    (Holm pass and the 1-SE variant's t >= 1).
    """
    pvals, chosen_t = {}, {}
    for fam in families:
        cols, stats = {}, []
        for v in fam["variants"]:
            ps = placebo_signals(v["signals"], prices, v["delay"], rng)
            last = gates.last_signal_date(prices.dates, v["H"], v["delay"], end)
            r = engine.run_cohorts(ps, prices, H=v["H"], cost_bps=0.0, delay=v["delay"], side=fam["side"],
                                   start=start, end=last, cache=cache)
            # the variant's own gate series; plain-basket benchmark, under which random
            # names have exactly zero expected excess (a beta benchmark would add (1 - mean beta) x R_B)
            x = gates.window(r.daily[v.get("series", "excess_net")], start, end)
            cols[v["name"]] = x
            stats.append(gates.variant_stats(x, start, end))
        D = pd.DataFrame(cols)
        sp = gates.family_spa(D, B=B, R=R, seed=int(rng.integers(2**31)))
        pvals[fam["id"]] = sp["p_c_calibrated"]
        chosen_t[fam["id"]] = stats[gates.one_se_choice(stats)]["t"]
    h = gates.holm(pvals)
    adopted = {k: bool(h[k]["pass"] and chosen_t[k] >= 1.0) for k in pvals}
    return {"p": pvals, "holm": {k: h[k]["pass"] for k in h}, "adopted": adopted,
            "any_adopted": any(adopted.values())}


# ------------------------------------------------------------ parallel runner
_W = {}


def _init_worker():
    _W["prices"] = engine.Prices.load()
    _W["cache"] = {}


def _one(args):
    rep, seed, fams, start, end, B, R = args
    rng = np.random.default_rng([seed, rep])
    return gate_once(fams, _W["prices"], rng, start, end, B=B, R=R, cache=_W["cache"])


def calibrate(families: list[dict], reps: int = 300, workers: int = 4, B: int = 199, R: int = 199,
              seed: int = 20261127, start=None, end=None) -> dict:
    """Run ``reps`` placebo replications of the gate in parallel.

    Gates (c) signature and (d) robustness are NOT applied to placebos. Both can
    only remove adoptions, so the measured rate is an upper bound on the full
    gate's false-adoption rate.
    """
    from concurrent.futures import ProcessPoolExecutor
    start = start or gates.DEV[0]
    end = end or gates.DEV[1]
    jobs = [(r, seed, families, start, end, B, R) for r in range(reps)]
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as ex:
        out = list(ex.map(_one, jobs, chunksize=4))
    ids = [f["id"] for f in families]
    any_rate = float(np.mean([o["any_adopted"] for o in out]))
    per = {k: float(np.mean([o["adopted"][k] for o in out])) for k in ids}
    p_lt_05 = {k: float(np.mean([o["p"][k] <= 0.05 for o in out])) for k in ids}
    se = float(np.sqrt(any_rate * (1 - any_rate) / reps))
    return {"reps": reps, "B": B, "R": R, "seed": seed,
            "familywise_false_adoption": any_rate, "mc_se": se,
            "per_family_false_adoption": per, "per_family_calibrated_p_le_05": p_lt_05,
            "note": "upper bound: placebo adoption ignores the signature and robustness gates"}
