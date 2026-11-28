"""Apply v2's declared selection rule (research/trading/v2/PLAN.md) to the full registry.

The rule is mechanical. A trial is a v2 candidate only if all of these hold
on dev:
1. StepM rejects it at FWER 10% across ALL registered trials (block 20).
2. Its deflated Sharpe ratio is >= 0.95, with N = the effective number of
   trials and Var[SR] taken across the trials.
3. Its mean is > 0 at 25 bps per side and with a one-day delay.
4. For event trials: the per-event episode-clustered t is >= 2, the placebo
   percentile is >= 95%, and the same-stock-adjusted mean is > 0.
5. The registry-wide PBO is < 0.5.
Within each family, the candidate with the highest dev Sharpe is chosen.
"""

from __future__ import annotations

import numpy as np

from . import diagnostics as D, explore as X, gates, multitest as mt, protocol

ALPHA, DSR_MIN, EVT_T_MIN, PLACEBO_MIN, PBO_MAX = 0.10, 0.95, 2.0, 0.95, 0.5


def select(B: int = 4000, n_placebo: int = 200, seed: int = 0) -> dict:
    recs, S = X.load_registry()
    M = S.fillna(0.0).to_numpy(float)
    ids = list(S.columns)
    neff = mt.effective_trials(M)
    sm = mt.stepm(M, alpha=ALPHA, block=gates.BLOCK, B=B, seed=seed)
    pbo = mt.pbo_cscv(M, S=16)
    sr = M.mean(axis=0) / M.std(axis=0, ddof=1)
    ctx = X.Ctx()
    rows = {}
    for i in np.flatnonzero(sm["rejected"]):
        cid = ids[i]
        r = recs[recs.id == cid].iloc[0]
        dsr = mt.deflated_sharpe(M[:, i], sr, N=neff)
        crit = {"stepm": True, "dsr": bool(dsr["dsr"] >= DSR_MIN),
                "robust_25bps_delay": bool(r["mean_25bps"] > 0 and r["mean_delay_plus1"] > 0),
                "pbo": bool(pbo["pbo"] < PBO_MAX)}
        diag = None
        if r["kind"] == "event":
            diag = D.event_diagnostics(X.Trial(r["family"], r["kind"], r["params"], int(r["round"])), ctx,
                                       *gates.DEV, n_placebo=n_placebo, seed=seed)
            crit["event_t_episode"] = bool(r.get("event_t_episode", 0) >= EVT_T_MIN)
            crit["placebo"] = bool(diag["placebo"]["percentile"] >= PLACEBO_MIN)
            crit["same_stock"] = bool(r.get("same_stock_adj_mean", -1) > 0)
        rows[cid] = {"family": r["family"], "t": float(sm["t"][i]), "sharpe_annual": dsr["sr_annual"],
                     "dsr": dsr, "criteria": crit, "candidate": all(crit.values()), "diagnostics": diag}
    cands = {k: v for k, v in rows.items() if v["candidate"]}
    chosen = {}
    for k, v in cands.items():
        f = v["family"]
        if f not in chosen or v["sharpe_annual"] > cands[chosen[f]]["sharpe_annual"]:
            chosen[f] = k
    return {"registry": {"n_trials": len(ids), "n_eff": neff, "stepm_first_critical_t": sm["critical_t_first"],
                         "stepm_steps": sm["steps"], "pbo": pbo},
            "stepm_rejected": rows, "candidates": sorted(cands), "chosen_by_family": chosen,
            "rule": {"alpha": ALPHA, "dsr_min": DSR_MIN, "event_t_min": EVT_T_MIN, "placebo_min": PLACEBO_MIN,
                     "pbo_max": PBO_MAX}}


def main() -> dict:
    out = select()
    protocol.write_json(X.REG / "selection.json", out)
    return out
