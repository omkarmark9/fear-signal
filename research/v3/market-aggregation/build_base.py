"""Build the per-day / per-stock base arrays needed for the market-aggregation study.

Outputs (all under /tmp/claude-0/v3/market-aggregation/):
  dev_oos.npz   stitched walk-forward out-of-sample arrays (from the cached S1 fold fits)
  full_frozen.npz  frozen v2 point model applied to the full panel (2023..2026-11-23)

No VXN correlation is computed here (no scoring).
"""
import os
import pickle
import json
import numpy as np
import pandas as pd

os.environ["GIT_OPTIONAL_LOCKS"] = "0"
from fearsignal import data, features, model as mdl, scoring, frozen, pipeline, market, evaluate

OUT = "/tmp/claude-0/v3/market-aggregation"
CACHE = "/tmp/claude-0/v3/shared/fold_models_S1.pkl"


def per_day(sig, model, obs):
    """Everything an aggregate might need, kept per stock-day (float32 to save space)."""
    el = obs.eligible
    a = sig.alpha  # NaN when ineligible
    return dict(
        P_stress=a[:, :, 1].astype(np.float32),
        P_panic=a[:, :, 2].astype(np.float32),
        P_calm=a[:, :, 0].astype(np.float32),
        sev=sig.severity.astype(np.float32),
        L=sig.L.astype(np.float32),
        Z=sig.Z.astype(np.float32),
        el=el,
        B=sig.breadth, BL=sig.breadth_L, M=sig.M, Mleg=sig.M_legacy,
    )


def dev_oos():
    panel = data.load_dev_panel()
    feat = features.compute(panel)
    obs = feat.obs
    T, N = obs.x.shape
    cache = pickle.load(open(CACHE, "rb"))
    keys2d = ["P_stress", "P_panic", "P_calm", "sev", "L", "Z"]
    keys1d = ["B", "BL", "M", "Mleg"]
    st = {k: np.full((T, N), np.nan, np.float32) for k in keys2d}
    st.update({k: np.full(T, np.nan) for k in keys1d})
    block = np.full(T, -1)
    # per-fold "training-window" arrays are needed for any frozen per-fold map a candidate uses;
    # store each fold's full-panel arrays of the 2d quantities compactly via fold index
    fold_full = []
    for b, f in enumerate(cache["folds"]):
        tr_end, te_end = pd.Timestamp(f["train_end"]), pd.Timestamp(f["test_end"])
        test = (obs.dates > tr_end) & (obs.dates <= te_end)
        train = obs.dates <= tr_end
        block[test] = b
        sig = mdl.compute_signal(f["model"], obs, f["z_map"], f["m_map"], f["zero_rate"])
        d = per_day(sig, f["model"], obs)
        for k in keys2d:
            st[k][test] = d[k][test]
        for k in keys1d:
            st[k][test] = d[k][test]
        fold_full.append({k: d[k] for k in keys2d + keys1d})
        fold_full[-1]["train"] = train
    # check against the committed walk-forward
    wf = np.load(data.ROOT / "outputs" / "dev" / "walkforward_daily.npz")
    okB = np.isfinite(wf["breadth__S1_v1_refit"])
    print("dev check breadth max|diff|:", np.nanmax(np.abs(wf["breadth__S1_v1_refit"][okB] - st["B"][okB])))
    okM = np.isfinite(wf["mlegacy__S1_v1_refit"])
    print("dev check Mleg max|diff|:", np.nanmax(np.abs(wf["mlegacy__S1_v1_refit"][okM] - st["Mleg"][okM])))
    print("dev check block equal:", np.array_equal(wf["block"], block))
    np.savez_compressed(f"{OUT}/dev_oos.npz", dates=obs.dates.values.astype("datetime64[D]").astype(str),
                        block=block, el=obs.eligible, tickers=np.array(obs.tickers), **st)
    # fold-full arrays (needed for training-window calibrations of candidates)
    for b, ff in enumerate(fold_full):
        np.savez_compressed(f"{OUT}/dev_fold{b}_full.npz", **ff)


def full_frozen():
    spec = frozen.load()
    sha = frozen.committed_spec_hash()
    panel = data.load_holdout_panel(sha)
    feat = features.compute(panel)
    obs = feat.obs
    point = pipeline.model_from_dict(spec["stock_model"])
    z_map = scoring.QuantileMap.from_dict(spec["z_map"])
    m_map = scoring.QuantileMap.from_dict(spec["m_map"])
    sig = mdl.compute_signal(point, obs, z_map, m_map, spec["dev_zero_rate"])
    d = per_day(sig, point, obs)
    # v1 as published, frozen v1 maps (the H1b comparator)
    vm = spec["v1_reference_maps"]
    v1 = mdl.compute_signal(pipeline._v1_model(), obs, scoring.QuantileMap.from_dict(vm["z_map"]),
                            scoring.QuantileMap.from_dict(vm["m_map"]), spec["dev_zero_rate"])
    mf = pd.read_csv(data.ROOT / "outputs" / "holdout" / "market_fear.csv", index_col=0, parse_dates=True)
    mf = mf.reindex(obs.dates)
    ok = np.isfinite(mf["M"].to_numpy())
    print("2026 check M max|diff|:", np.nanmax(np.abs(mf["M"].to_numpy()[ok] - sig.M[ok])))
    print("2026 check Mleg max|diff|:", np.nanmax(np.abs(mf["M_legacy"].to_numpy()[ok] - sig.M_legacy[ok])))
    np.savez_compressed(f"{OUT}/full_frozen.npz", dates=obs.dates.values.astype("datetime64[D]").astype(str),
                        tickers=np.array(obs.tickers), x=obs.x.astype(np.float32), kind=obs.kind,
                        Mleg_v1=v1.M_legacy, Z_v1=v1.Z.astype(np.float32), **d)


if __name__ == "__main__":
    dev_oos()
    full_frozen()
