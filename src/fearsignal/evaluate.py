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
   today's close", against climatology (training frequency) and an EWMA
   (RiskMetrics, lambda = 0.94) scale forecast.
3. **Construct validity** of the market layer: correlation of 5-session changes
   with 5-session changes in VXN, on the stitched out-of-sample test periods.
   Market candidates: v1's expanding z-score of mean Z, breadth (expected share
   of stocks not calm) for every stock candidate, and a Gaussian HMM on the
   trimmed cross-sectional summary y_t (market.market_observation).

Candidate set and selection rules are fixed here, before the harness is run:

* stock layer: highest out-of-sample log score; a model with fewer parameters
  is preferred when it is within one Newey-West standard error of the best
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


def load_vxn(path=None) -> pd.Series:
    path = path or (data.RAW / "vxn_dev.csv")
    v = pd.read_csv(path, parse_dates=["observation_date"]).set_index("observation_date")["VXNCLS"]
    v = pd.to_numeric(v, errors="coerce").dropna()
    v.index.name = "date"
    return v


def ewma_tail_forecast(obs: Obs, zero_rate: float, lam: float = EWMA_LAMBDA,
                       threshold: float = mdl.TAIL_THRESHOLD) -> np.ndarray:
    """Baseline: q_t from an EWMA of x^2 over each stock's past eligible days."""
    T, N = obs.x.shape
    q = np.full((T, N), np.nan)
    v = np.full(N, np.nan)
    for t in range(T):
        have = np.isfinite(v)
        s = np.sqrt(v / max(1 - zero_rate, 1e-6))
        q[t] = np.where(have, (1 - zero_rate) * 2 * ndtr(-threshold / s), np.nan)
        el = obs.eligible[t]
        x2 = np.where(el, obs.x[t] ** 2, np.nan)
        v = np.where(el & have, lam * v + (1 - lam) * x2, v)
        v = np.where(el & ~have, x2 + 1.0, v)   # seed at the first eligible day
    return q


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
    q_ewma = np.full((T, N), np.nan)
    clim = np.full((T, N), np.nan)
    mobs = market.market_observation(obs)
    mkt_hmm_L = np.full(T, np.nan)

    for tr_end, test in _fold_masks(dates):
        train_obs = obs.slice_time(tr_end)
        z0 = mdl.zero_rate(train_obs)
        test_idx = np.flatnonzero(test)
        nobs_day[test_idx] = obs.eligible[test_idx].sum(axis=1)
        ytr = y_tail[: len(train_obs.dates)]
        clim[test_idx] = np.where(obs.eligible[test_idx], np.nanmean(ytr), np.nan)
        qe = ewma_tail_forecast(obs, z0)
        q_ewma[test_idx] = qe[test_idx]
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
                "A": m.A.tolist(), "iters": len(m.history),
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
    ref = "S3_state_K3"
    summ = {}
    for spec in mdl.CANDIDATES:
        ls = daily_logscore[spec.name][scored]
        diff = ls - daily_logscore[ref][scored]
        q = stitched[spec.name]["q"]
        ok = np.isfinite(q) & np.isfinite(y_tail)
        brier = float(np.mean((q[ok] - y_tail[ok]) ** 2))
        summ[spec.name] = {
            "logscore_per_obs": float(ls.sum() / tot_obs),
            "diff_vs_ref_per_obs": float(diff.sum() / tot_obs),
            "diff_vs_ref_se_per_obs": float(stats.newey_west_se_mean(diff, 10) * len(diff) / tot_obs),
            "brier": brier,
        }
    ok = np.isfinite(clim) & np.isfinite(y_tail) & np.isfinite(q_ewma)
    b_clim = float(np.mean((clim[ok] - y_tail[ok]) ** 2))
    b_ewma = float(np.mean((q_ewma[ok] - y_tail[ok]) ** 2))
    for name, d in summ.items():
        d["brier_skill_vs_climatology"] = 1 - d["brier"] / b_clim
        d["brier_skill_vs_ewma"] = 1 - d["brier"] / b_ewma
        q = stitched[name]["q"]
        ok2 = np.isfinite(q) & np.isfinite(y_tail)
        d["calibration"] = stats.logistic_calibration(q[ok2], y_tail[ok2])
    # The selection rule compares against the best *out-of-sample* candidate.
    # S0 carries v1's published parameters, which were estimated on 2023-2025
    # and so have already seen every test fold: its scores are not
    # out-of-sample and it is reported for reference only.
    fair = [c.name for c in mdl.CANDIDATES if not c.fixed]
    best = max(fair, key=lambda n: summ[n]["logscore_per_obs"])
    for spec in mdl.CANDIDATES:
        diff = daily_logscore[spec.name][scored] - daily_logscore[best][scored]
        summ[spec.name]["diff_vs_best_per_obs"] = float(diff.sum() / tot_obs)
        summ[spec.name]["diff_vs_best_se_per_obs"] = float(
            stats.newey_west_se_mean(diff, 10) * len(diff) / tot_obs)
    results["best_out_of_sample"] = best
    results["summary"] = summ
    results["baselines"] = {"brier_climatology": b_clim, "brier_ewma": b_ewma,
                            "event_rate_test": float(np.nanmean(y_tail[scored]))}

    # ---- market layer, stitched out-of-sample, against VXN
    vxn = load_vxn().reindex(dates)
    mk = {}
    series_list = [("market_hmm", "y", mkt_hmm_L)]
    for spec in mdl.CANDIDATES:
        series_list.append((spec.name, "breadth", scoring.logit(stitched[spec.name]["B"])))
        series_list.append((spec.name, "M_legacy", stitched[spec.name]["Mleg"]))
    for name, label, series in series_list:
        df = pd.DataFrame({"s": series, "v": vxn.to_numpy()}, index=dates)[scored].dropna()
        d5 = df.diff(5).dropna()
        mk[f"{name}:{label}"] = {
            "level_spearman": stats.corr_ci(df["s"], df["v"], "spearman", n_boot=500),
            "d5_pearson": stats.corr_ci(d5["s"], d5["v"], "pearson", n_boot=500),
            "d5_spearman": stats.corr_ci(d5["s"], d5["v"], "spearman", n_boot=500),
        }
    for spec in mdl.CANDIDATES:
        df = pd.DataFrame({"a": scoring.logit(stitched[spec.name]["B"]),
                           "b": stitched[spec.name]["Mleg"], "v": vxn.to_numpy()},
                          index=dates)[scored].dropna()
        d5 = df.diff(5).dropna()
        mk[f"{spec.name}:breadth_minus_legacy_d5_spearman"] = stats.corr_diff_ci(
            d5["a"], d5["b"], d5["v"], n_boot=1000)
    results["market_vs_vxn"] = mk
    np.savez_compressed(
        (out_path.parent if out_path else data.ROOT / "outputs" / "dev") / "walkforward_daily.npz",
        dates=dates.values.astype("datetime64[D]").astype(str), scored=scored, nobs=nobs_day,
        **{f"logscore__{k}": v for k, v in daily_logscore.items()},
        **{f"breadth__{k}": v["B"] for k, v in stitched.items()},
        **{f"mlegacy__{k}": v["Mleg"] for k, v in stitched.items()},
        market_hmm_L=mkt_hmm_L)
    results["dev_end"] = str(DEV_END.date())
    if out_path:
        with open(out_path, "w") as fh:
            json.dump(results, fh, indent=2, default=float)
    return results
