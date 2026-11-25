"""Development-period evaluation (2023-2025 only).

Walk-forward with expanding training windows and refitting at each fold::

    fold A  train 2023-01-03 .. 2024-06-28   test 2024-07-01 .. 2024-12-31
    fold B  train 2023-01-03 .. 2024-12-31   test 2025-01-02 .. 2025-06-30
    fold C  train 2023-01-03 .. 2025-06-30   test 2025-07-01 .. 2025-12-31

The filter always runs from the start of the panel with the fold's training
parameters; only test-period days are scored, so every scored quantity is a
genuine one-step-ahead forecast. Scores:

1. **Predictive log score** per observation under the common censored,
   zero-inflated likelihood (a strictly proper scoring rule). Differences
   between candidates are summed per day and given Newey-West standard errors.
2. **Brier score** for the event "next session's low is >= 3 own-sigmas below
   today's close", against climatology (training frequency) and two EWMA
   (lambda = 0.94) scale forecasts, of |x| (the stronger) and of x^2.
3. **Construct validity** of the market layer: correlation of 5-session changes
   with 5-session changes in VXN, on the stitched out-of-sample test periods.
   Market candidates: v1's expanding z-score of mean Z, breadth (expected share
   of stocks not calm) for every stock candidate, and a Gaussian HMM on the
   trimmed cross-sectional summary y_t (market.market_observation).

Candidate set and selection rules are fixed here, before the harness is run:

* stock layer: highest out-of-sample log score; the model with the fewest
  parameters among those within one Newey-West standard error of the best is
  selected (coded below as ``selected_stock_model``)
  (S0, the published v1 parameters, was fitted by its author on 2023-2025 and
  so is not out-of-sample on any fold; it is a reference, not a candidate);
* market layer: highest 5-session-change Spearman correlation with VXN.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr

from . import data, features, hmm, market, model as mdl, scoring, stats
from .config import DEV_END
from .features import Obs

FOLDS = (
    ("2024-06-28", "2024-12-31"),
    ("2024-12-31", "2025-06-30"),
    ("2025-06-30", "2025-12-31"),
)
EWMA_LAMBDA = 0.94


def load_vxn(path=None, dev_only: bool = True) -> pd.Series:
    """VXN closes (validation only). Truncated at DEV_END unless dev_only=False,
    which only the frozen holdout run passes."""
    path = path or (data.RAW / "vxn_dev.csv")
    v = pd.read_csv(path, parse_dates=["observation_date"]).set_index("observation_date")["VXNCLS"]
    v = pd.to_numeric(v, errors="coerce").dropna()
    v.index.name = "date"
    if dev_only:
        v = v[v.index <= DEV_END]
    return v


def ewma_tail_forecast(obs: Obs, zero_rate: float, power: int = 1, lam: float = EWMA_LAMBDA,
                       threshold: float = mdl.TAIL_THRESHOLD) -> np.ndarray:
    """Baseline: q_t from an EWMA of |x|^power over each stock's past eligible days.

    The EWMA estimates E|x|^p under a zero-inflated half-normal with scale s,
    E|x| = (1 - z) s sqrt(2/pi) and E x^2 = (1 - z) s^2, which is inverted for
    s and mapped to (1 - z) 2 Phi(-threshold / s). power=1 is the stronger
    baseline (one censored day contributes 10, not 100).
    """
    T, N = obs.x.shape
    q = np.full((T, N), np.nan)
    v = np.full(N, np.nan)
    c = (1 - zero_rate) * (np.sqrt(2 / np.pi) if power == 1 else 1.0)
    seed = 0.8 if power == 1 else 1.0
    for t in range(T):
        have = np.isfinite(v)
        s = (v / c) if power == 1 else np.sqrt(v / c)
        q[t] = np.where(have, (1 - zero_rate) * 2 * ndtr(-threshold / s), np.nan)
        el = obs.eligible[t]
        xp = np.where(el, np.abs(obs.x[t]) ** power, np.nan)
        v = np.where(el & have, lam * v + (1 - lam) * xp, v)
        v = np.where(el & ~have, xp + seed, v)   # seed at the first eligible day
    return q


def d5_within_blocks(s: np.ndarray, v: np.ndarray, block: np.ndarray) -> pd.DataFrame:
    """5-session changes that never straddle two fold models."""
    df = pd.DataFrame({"s": s, "v": v})
    d = df.diff(5)
    b = pd.Series(block)
    same = b.eq(b.shift(5)) & (b >= 0)
    return d[same.to_numpy()].dropna()


def _fold_masks(dates: pd.DatetimeIndex):
    for tr_end, te_end in FOLDS:
        tr_end, te_end = pd.Timestamp(tr_end), pd.Timestamp(te_end)
        yield tr_end, (dates > tr_end) & (dates <= te_end)


def run(n_random: int = 2, seed: int = 0, out_path=None) -> dict:
    out_path = Path(out_path) if out_path else None
    panel = data.load_dev_panel()
    feat = features.compute(panel)
    obs = feat.obs
    dates = obs.dates
    T, N = obs.x.shape
    y_tail = mdl.tail_event(obs)
    results = {"folds": [], "candidates": [asdict(c) for c in mdl.CANDIDATES]}
    daily_logscore = {c.name: np.full(T, np.nan) for c in mdl.CANDIDATES}
    nobs_day = np.zeros(T)
    stitched = {c.name: {"B": np.full(T, np.nan), "Mleg": np.full(T, np.nan),
                         "q": np.full((T, N), np.nan)} for c in mdl.CANDIDATES}
    q_ewma = {1: np.full((T, N), np.nan), 2: np.full((T, N), np.nan)}
    clim = np.full((T, N), np.nan)
    block = np.full(T, -1)
    mobs = market.market_observation(obs)
    mkt_hmm_L = np.full(T, np.nan)

    for b, (tr_end, test) in enumerate(_fold_masks(dates)):
        block[test] = b
        train_obs = obs.slice_time(tr_end)
        z0 = mdl.zero_rate(train_obs)
        test_idx = np.flatnonzero(test)
        nobs_day[test_idx] = obs.eligible[test_idx].sum(axis=1)
        ytr = y_tail[: len(train_obs.dates)]
        clim[test_idx] = np.where(obs.eligible[test_idx], np.nanmean(ytr), np.nan)
        for pw in q_ewma:
            q_ewma[pw][test_idx] = ewma_tail_forecast(obs, z0, power=pw)[test_idx]
        fold = {"train_end": str(tr_end.date()), "test_days": int(test.sum()), "models": {}}
        for spec in mdl.CANDIDATES:
            m = mdl.fit_stock_model(spec, train_obs, n_random=n_random, seed=seed)
            sm = mdl.as_scoring_model(m, z0)
            fr = sm.filter(obs)
            daily_logscore[spec.name][test_idx] = fr.logc[test_idx].sum(axis=1)
            z_map, _ = mdl.fit_maps(m, train_obs)
            sig = mdl.compute_signal(m, obs, z_map, None, z0)
            # q_next[t] forecasts day t+1; store it on the day it is scored
            q_on_day = np.vstack([np.full((1, N), np.nan), sig.q_next[:-1]])
            stitched[spec.name]["q"][test_idx] = q_on_day[test_idx]
            stitched[spec.name]["B"][test_idx] = sig.breadth[test_idx]
            stitched[spec.name]["Mleg"][test_idx] = sig.M_legacy[test_idx]
            e = m.emission
            fold["models"][spec.name] = {
                "loglik_train": m.loglik, "n_params": m.n_params(),
                "s": e.s.tolist(), "pi": None if e.pi is None else e.pi.tolist(),
                "A": m.A.tolist(), "iters": len(m.history), "converged": m.converged,
                "test_logscore_per_obs": float(fr.logc[test_idx].sum() / obs.eligible[test_idx].sum()),
            }
        # market-level Gaussian HMM on the trimmed cross-sectional summary
        y_tr = mobs.slice_time(tr_end)
        yq = np.nanquantile(y_tr.x, [0.3, 0.8, 0.98])
        g0 = hmm.HMM(hmm.GaussianEmission(yq, np.full(3, np.nanstd(y_tr.x) / 2)),
                     np.full((3, 3), 0.05) + np.eye(3) * 0.85)
        g = hmm.em(g0, y_tr, max_iter=3000, tol=1e-10)
        gsev = scoring.severity(g.filter(mobs).alpha[:, 0], g.emission.mu)
        mkt_hmm_L[test_idx] = scoring.logit(gsev)[test_idx]
        fold["market_hmm"] = {"mu": g.emission.mu.tolist(), "sd": g.emission.sd.tolist(),
                              "A": g.A.tolist()}
        results["folds"].append(fold)

    scored = np.isfinite(daily_logscore[mdl.CANDIDATES[0].name])
    tot_obs = nobs_day[scored].sum()
    summ = {spec.name: {"logscore_per_obs": float(daily_logscore[spec.name][scored].sum() / tot_obs),
                        "n_params": results["folds"][-1]["models"][spec.name]["n_params"]}
            for spec in mdl.CANDIDATES}
    ok = np.isfinite(clim) & np.isfinite(y_tail) & np.isfinite(q_ewma[1]) & np.isfinite(q_ewma[2])
    b_clim = float(np.mean((clim[ok] - y_tail[ok]) ** 2))
    b_ewma = {pw: float(np.mean((q[ok] - y_tail[ok]) ** 2)) for pw, q in q_ewma.items()}
    day = np.broadcast_to(np.arange(T)[:, None], (T, N))
    for name, d in summ.items():
        q = stitched[name]["q"]
        ok2 = ok & np.isfinite(q)
        d["brier"] = float(np.mean((q[ok2] - y_tail[ok2]) ** 2))
        d["brier_skill_vs_climatology"] = 1 - d["brier"] / b_clim
        d["brier_skill_vs_ewma_abs"] = 1 - d["brier"] / b_ewma[1]
        d["brier_skill_vs_ewma_sq"] = 1 - d["brier"] / b_ewma[2]
        d["calibration"] = stats.logistic_calibration(q[ok2], y_tail[ok2], groups=day[ok2])
    # The selection rule compares against the best *out-of-sample* candidate.
    # S0 carries v1's published parameters, which were estimated on 2023-2025
    # and so have already seen every test fold: its scores are not
    # out-of-sample and it is reported for reference only.
    fair = [c for c in mdl.CANDIDATES if not c.fixed]
    best = max(fair, key=lambda c: summ[c.name]["logscore_per_obs"])
    for spec in mdl.CANDIDATES:
        diff = daily_logscore[spec.name][scored] - daily_logscore[best.name][scored]
        summ[spec.name]["diff_vs_best_per_obs"] = float(diff.sum() / tot_obs)
        summ[spec.name]["diff_vs_best_se_per_obs"] = float(
            stats.newey_west_se_mean(diff, 10) * len(diff) / tot_obs)
    # one-standard-error rule: the fewest parameters among candidates within 1 s.e. of the best
    within = [c for c in fair if summ[c.name]["diff_vs_best_per_obs"]
              >= -summ[c.name]["diff_vs_best_se_per_obs"]]
    selected = min(within, key=lambda c: (summ[c.name]["n_params"], -summ[c.name]["logscore_per_obs"]))
    results["best_out_of_sample"] = best.name
    results["selected_stock_model"] = selected.name
    results["summary"] = summ
    results["baselines"] = {"brier_climatology": b_clim, "brier_ewma_abs": b_ewma[1],
                            "brier_ewma_sq": b_ewma[2], "n_scored_stock_days": int(ok.sum()),
                            "n_scored_days": int(scored.sum()),
                            "event_rate_test": float(np.nanmean(y_tail[ok]))}

    # ---- market layer, stitched out-of-sample, against VXN
    # 5-session changes are taken within each fold's test block only, so no
    # change mixes two fold models.
    vxn = load_vxn().reindex(dates).to_numpy()
    mk = {}
    series_list = [("market_hmm", "y", mkt_hmm_L)]
    for spec in mdl.CANDIDATES:
        series_list.append((spec.name, "breadth", scoring.logit(stitched[spec.name]["B"])))
        series_list.append((spec.name, "M_legacy", stitched[spec.name]["Mleg"]))
    for name, label, series in series_list:
        lv = pd.DataFrame({"s": series, "v": vxn})[scored].dropna()
        d5 = d5_within_blocks(series, vxn, block)
        mk[f"{name}:{label}"] = {
            "level_spearman": stats.corr_ci(lv["s"], lv["v"], "spearman", block=60, n_boot=500),
            "d5_pearson": stats.corr_ci(d5["s"], d5["v"], "pearson", n_boot=500),
            "d5_spearman": stats.corr_ci(d5["s"], d5["v"], "spearman", n_boot=500),
        }
    for spec in mdl.CANDIDATES:
        a = d5_within_blocks(scoring.logit(stitched[spec.name]["B"]), vxn, block)
        b = d5_within_blocks(stitched[spec.name]["Mleg"], vxn, block)
        j = a.index.intersection(b.index)
        mk[f"{spec.name}:breadth_minus_legacy_d5_spearman"] = stats.corr_diff_ci(
            a.loc[j, "s"], b.loc[j, "s"], a.loc[j, "v"], n_boot=1000)
    results["market_vs_vxn"] = mk
    np.savez_compressed(
        (out_path.parent if out_path else data.ROOT / "outputs" / "dev") / "walkforward_daily.npz",
        dates=dates.values.astype("datetime64[D]").astype(str), scored=scored, nobs=nobs_day,
        block=block,
        **{f"logscore__{k}": v for k, v in daily_logscore.items()},
        **{f"breadth__{k}": v["B"] for k, v in stitched.items()},
        **{f"mlegacy__{k}": v["Mleg"] for k, v in stitched.items()},
        market_hmm_L=mkt_hmm_L)
    results["dev_end"] = str(DEV_END.date())
    if out_path:
        with open(out_path, "w") as fh:
            json.dump(results, fh, indent=2, default=float)
    return results


def compare_family(candidate: "mdl.StockSpec", incumbent_name: str = "S1_v1_refit",
                   n_random: int = 2, seed: int = 0, B: int = 999, R: int = 999) -> dict:
    """Score one declared stock-layer candidate against the incumbent on the dev
    walk-forward, with exactly the folds, seeds and scoring of ``run``.

    Returns the daily log-score differential and its NW(10) s.e., the
    calibrated SPA_c p-value (K = 1), per-fold gains, the fitted transition
    matrices (for the mechanism check), the market-layer check, and secondary
    tail-forecast diagnostics.
    """
    from . import spa

    panel = data.load_dev_panel()
    obs = features.compute(panel).obs
    dates = obs.dates
    T, N = obs.x.shape
    y_tail = mdl.tail_event(obs)
    inc_spec = next(c for c in mdl.CANDIDATES if c.name == incumbent_name)
    ls = {k: np.full(T, np.nan) for k in ("inc", "cand")}
    q = {k: np.full((T, N), np.nan) for k in ("inc", "cand")}
    BL = {k: np.full(T, np.nan) for k in ("inc", "cand")}
    nobs = np.zeros(T)
    block = np.full(T, -1)
    folds = []
    for b, (tr_end, test) in enumerate(_fold_masks(dates)):
        block[test] = b
        tr = obs.slice_time(tr_end)
        z0 = mdl.zero_rate(tr)
        idx = np.flatnonzero(test)
        nobs[idx] = obs.eligible[idx].sum(axis=1)
        fold = {"train_end": str(tr_end.date())}
        for key, spec in (("inc", inc_spec), ("cand", candidate)):
            m = mdl.fit_stock_model(spec, tr, n_random=n_random, seed=seed)
            fr = mdl.as_scoring_model(m, z0).filter(obs)
            ls[key][idx] = fr.logc[idx].sum(axis=1)
            sig = mdl.compute_signal(m, obs, None, None, z0)
            qd = np.vstack([np.full((1, N), np.nan), sig.q_next[:-1]])
            q[key][idx] = qd[idx]
            BL[key][idx] = sig.breadth_L[idx]
            fold[key] = {"A": m.A.tolist(), "s": m.emission.s.tolist(), "converged": m.converged,
                         "loglik_train": m.loglik, "n_params": m.n_params()}
        fold["gain_per_obs"] = float((ls["cand"][idx] - ls["inc"][idx]).sum() / nobs[idx].sum())
        folds.append(fold)
    scored = np.isfinite(ls["inc"])
    tot = nobs[scored].sum()
    d = (ls["cand"] - ls["inc"])[scored] / nobs[scored]         # per-observation daily gain
    gain = float((ls["cand"] - ls["inc"])[scored].sum() / tot)
    se = float(stats.newey_west_se_mean((ls["cand"] - ls["inc"])[scored], 10) * scored.sum() / tot)
    gate = spa.spa_calibrated(d[:, None], block=10, B=B, R=R, seed=seed)
    vxn = load_vxn().reindex(dates).to_numpy()
    a = d5_within_blocks(BL["cand"], vxn, block)
    bb = d5_within_blocks(BL["inc"], vxn, block)
    j = a.index.intersection(bb.index)
    mkt = stats.corr_diff_ci(a.loc[j, "s"], bb.loc[j, "s"], a.loc[j, "v"], n_boot=2000)
    sec = {}
    day = np.broadcast_to(np.arange(T)[:, None], (T, N))
    for key in ("inc", "cand"):
        ok = np.isfinite(q[key]) & np.isfinite(y_tail)
        sec[key] = {"brier": float(np.mean((q[key][ok] - y_tail[ok]) ** 2)),
                    "calibration": stats.logistic_calibration(q[key][ok], y_tail[ok], groups=day[ok]),
                    "within_between": stats.within_between_calibration(q[key][ok], y_tail[ok], day[ok])}
    return {"candidate": candidate.name, "incumbent": incumbent_name,
            "n_days": int(scored.sum()), "n_obs": int(tot),
            "gain_per_obs": gain, "nw10_se_per_obs": se, "gain_over_se": gain / se,
            "spa_calibrated": {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in gate.items()},
            "folds": folds, "market_d5_spearman_cand_minus_inc": mkt, "tail_secondary": sec}
