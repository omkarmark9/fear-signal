"""Shared helpers for the market-aggregation study (scratch only)."""
import os
import numpy as np
import pandas as pd
from scipy import stats as sst

os.environ["GIT_OPTIONAL_LOCKS"] = "0"
from fearsignal import evaluate, stats, data, scoring

OUT = "/tmp/claude-0/v3/market-aggregation"
DEV_END = pd.Timestamp("2025-12-31")


def rowmean(a):
    a = np.asarray(a, float)
    n = np.isfinite(a).sum(axis=1)
    s = np.nansum(a, axis=1)
    return np.where(n > 0, s / np.maximum(n, 1), np.nan)


def load_dev():
    d = dict(np.load(f"{OUT}/dev_oos.npz"))
    dates = pd.to_datetime(d["dates"])
    vxn = evaluate.load_vxn().reindex(dates).to_numpy()
    folds = [dict(np.load(f"{OUT}/dev_fold{b}_full.npz")) for b in range(3)]
    return d, dates, vxn, folds


def load_full():
    f = dict(np.load(f"{OUT}/full_frozen.npz"))
    dates = pd.to_datetime(f["dates"])
    vxn = evaluate.load_vxn(data.RAW / "vxn_full.csv", dev_only=False).reindex(dates).to_numpy()
    return f, dates, vxn


def aggregates(arr):
    """Raw (unstandardised) daily aggregates from a dict of per-stock arrays."""
    return {
        "B": np.asarray(arr["B"], float),
        "BL": np.asarray(arr["BL"], float),
        "sev": rowmean(arr["sev"]),
        "Zbar": rowmean(arr["Z"]),
        "Lbar": rowmean(arr["L"]),
    }


def standardise(x, calib_mask):
    v = x[calib_mask & np.isfinite(x)]
    return (x - v.mean()) / v.std(ddof=1), float(v.mean()), float(v.std(ddof=1))


def d5_full(s, v, dates, mask):
    """5-session changes on the full calendar, restricted to mask (validate._d5)."""
    df = pd.DataFrame({"s": s, "v": v}, index=dates)
    return df.diff(5)[mask].dropna()


def boot_pair(a, b, v, block=20.0, n_boot=2000, seed=0):
    """Paired stationary bootstrap of corr(a,v)-corr(b,v), Spearman and Pearson.

    Same resampling scheme and seed as stats.corr_diff_ci, so the Spearman
    interval reproduces it; additionally returns bootstrap s.d. (s.e.).
    """
    a, b, v = (np.asarray(z, float) for z in (a, b, v))
    ok = np.isfinite(a) & np.isfinite(b) & np.isfinite(v)
    a, b, v = a[ok], b[ok], v[ok]
    sp = lambda u, w: sst.spearmanr(u, w)[0]
    pe = lambda u, w: np.corrcoef(u, w)[0, 1]
    est = {"spearman": sp(a, v) - sp(b, v), "pearson": pe(a, v) - pe(b, v)}
    rng = np.random.default_rng(seed)
    bs = {"spearman": np.empty(n_boot), "pearson": np.empty(n_boot)}
    for j in range(n_boot):
        ix = stats.stationary_bootstrap_indices(len(a), block, rng)
        bs["spearman"][j] = sp(a[ix], v[ix]) - sp(b[ix], v[ix])
        bs["pearson"][j] = pe(a[ix], v[ix]) - pe(b[ix], v[ix])
    out = {}
    for k in est:
        lo, hi = np.quantile(bs[k], [0.025, 0.975])
        out[k] = {"diff": float(est[k]), "se": float(bs[k].std(ddof=1)), "lo": float(lo), "hi": float(hi),
                  "z": float(est[k] / bs[k].std(ddof=1)), "p_le0": float((bs[k] <= 0).mean())}
    out["n"] = int(len(a))
    return out


def corr_pair(s, v):
    return {"spearman": float(sst.spearmanr(s, v)[0]), "pearson": float(np.corrcoef(s, v)[0, 1]),
            "n": int(len(s))}
