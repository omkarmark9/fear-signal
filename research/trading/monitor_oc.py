"""Operating characteristics of the declared live monitors (research/trading/PROSPECTIVE.md).

Dev data only (2024-01-02..2025-12-31): ND1's 145 dev events, resampled by 5-session
episode (a block bootstrap that keeps the clustering) to sequences of N closed events,
with the mean shifted to each scenario. Each sequence runs through the exact monitor code,
prospective.monitor, from Tier 1 (ND's tier today). It also reports the rule as first coded
(CR1 s.e. by episode), which the per-trade s.e. replaced before the lock.

    .venv/bin/python research/trading/monitor_oc.py        # writes outputs/trading/monitor_oc.json
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from fearsignal.trading import backtest, explore as X, gates, prospective as P, protocol

N, B, SEED = 110, 4000, 0


def cr1_monitor(e, cl, dev_mean):
    """The rule as first coded: CR1 s.e. by episode (for comparison only)."""
    def se(x, c):
        G = len(np.unique(c))
        if len(x) < 3 or G < 2:
            return None
        s = pd.Series(x - x.mean()).groupby(c).sum().to_numpy()
        return float(np.sqrt(G / (G - 1) * (s ** 2).sum()) / len(x))
    susp, dem, nxt = None, [], P.DEMOTE_WINDOW
    for k in range(P.GUARD_EVERY, len(e) + 1, P.GUARD_EVERY):
        s = se(e[:k], cl[:k])
        if s is not None and e[:k].mean() < dev_mean - 2 * s:
            susp = k
            break
    for k in range(P.DEMOTE_WINDOW, len(e) + 1):
        if k < nxt:
            continue
        s = se(e[k - P.DEMOTE_WINDOW:k], cl[k - P.DEMOTE_WINDOW:k])
        if s is not None and e[k - P.DEMOTE_WINDOW:k].mean() < -s:
            dem.append(k)
            nxt = k + P.DEMOTE_WINDOW
    return susp, dem


def main() -> dict:
    rf, prices = backtest.load_inputs()
    res, _ = backtest.run_window(P.nd1_variant(), rf, prices, *gates.DEV)
    tr = res.trades.sort_values(["signal_date", "ticker"]).reset_index(drop=True)
    cl = X._episodes(tr["signal_date"], prices.dates)
    dev_mean = float(json.loads(protocol.SELECTED.read_text())["families"]["ND"]["chosen_stats"]["mean_event_excess"])
    e = tr["excess_net"].to_numpy(float)
    eps = [e[cl == c] for c in np.unique(cl)]
    days = pd.bdate_range("2027-01-04", periods=N * 3)
    out = {"N_events": N, "B": B, "seed": SEED, "dev_events": int(len(e)), "dev_episodes": len(eps),
           "dev_mean": dev_mean, "on_the_dev_sequence": {}, "scenarios": {}}
    m = P.monitor(tr, prices.dates, dev_mean, 1)
    out["on_the_dev_sequence"] = {"per_trade_se": {"suspended_at": m["suspended_at"], "demotions_at": m["demotions_at"]},
                                  "cr1_as_first_coded": dict(zip(("suspended_at", "demotions_at"),
                                                                 cr1_monitor(e, cl, dev_mean)))}
    for name, mu in (("dev effect", dev_mean), ("half the dev effect", dev_mean / 2), ("zero", 0.0),
                     ("-1% per event", -0.01), ("-2% per event", -0.02)):
        rng = np.random.default_rng(SEED)
        cnt = {"per_trade_se": np.zeros(4), "cr1_as_first_coded": np.zeros(4)}
        for _ in range(B):
            seq, lab, i = [], [], 0
            while len(seq) < N:
                g = eps[rng.integers(len(eps))]
                seq.extend(g - dev_mean + mu)
                lab.extend([i] * len(g))
                i += 1
            seq, lab = np.asarray(seq[:N]), np.asarray(lab[:N])
            frame = pd.DataFrame({"signal_date": days[np.arange(N) * 3], "ticker": [f"T{j}" for j in range(N)],
                                  "excess_net": seq})
            mm = P.monitor(frame, days, dev_mean, 1)
            for key, (susp, dem) in (("per_trade_se", (mm["suspended_at"], mm["demotions_at"])),
                                     ("cr1_as_first_coded", cr1_monitor(seq, lab, dev_mean))):
                cnt[key] += [susp is not None, len(dem) >= 1, len(dem) >= 2, susp is not None or len(dem) >= 1]
        out["scenarios"][name] = {k: dict(zip(("P_guard", "P_at_least_one_demotion", "P_two_demotions",
                                               "P_tier0_from_tier1"), np.round(v / B, 3).tolist()))
                                  for k, v in cnt.items()}
        print(name, out["scenarios"][name], flush=True)
    protocol.write_json(protocol.OUT / "monitor_oc.json", out)
    return out


if __name__ == "__main__":
    main()
