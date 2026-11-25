"""Build the data cache for the H4 (volatility increment) work. Read-only on the repo.

Outputs /tmp/claude-0/v3/volatility-increment/cache.npz with
  dev panel:  dates_dev, R_dev, and per fold f in 0..2 (shared cache models):
              M_f, BL_f (logit breadth), MS_f (mean logit severity)  -- full dev span
  full panel: dates_full, R_full, frozen M, BL, MS, and published market_fear.csv M (check)
"""
import os
os.environ["GIT_OPTIONAL_LOCKS"] = "0"
import pickle
import numpy as np
import pandas as pd

from fearsignal import data, features, frozen, model as mdl, validate, pipeline, scoring

OUT = "/tmp/claude-0/v3/volatility-increment"


def market_vars(sig, obs):
    el = obs.eligible
    L = np.where(el, sig.L, np.nan)
    n = np.isfinite(L).sum(1)
    ms = np.where(n > 0, np.nansum(L, 1) / np.maximum(n, 1), np.nan)
    return sig.M, sig.breadth_L, ms


def main():
    res = {}
    # ---------------- dev panel, per-fold models from the shared cache
    panel = data.load_dev_panel()
    feat = features.compute(panel)
    obs = feat.obs
    res["dates_dev"] = obs.dates.values.astype("datetime64[D]").astype(str)
    res["R_dev"] = validate.basket_returns(feat)
    res["el_dev"] = obs.eligible.any(axis=1)
    with open("/tmp/claude-0/v3/shared/fold_models_S1.pkl", "rb") as fh:
        cache = pickle.load(fh)
    for f, fd in enumerate(cache["folds"]):
        sig = mdl.compute_signal(fd["model"], obs, fd["z_map"], fd["m_map"], fd["zero_rate"])
        M, BL, MS = market_vars(sig, obs)
        res[f"M_{f}"], res[f"BL_{f}"], res[f"MS_{f}"] = M, BL, MS
        res[f"train_end_{f}"] = str(pd.Timestamp(fd["train_end"]).date())
        print("fold", f, fd["train_end"], fd["test_end"])
    # check: stitched OOS breadth equals committed walkforward_daily
    wf = np.load("/home/user/fear-signal/outputs/dev/walkforward_daily.npz")
    B = wf["breadth__S1_v1_refit"]
    blk = wf["block"]
    for f in range(3):
        m = blk == f
        Bf = 1 / (1 + np.exp(-res[f"BL_{f}"][m]))
        print("fold", f, "max |breadth - committed|", np.nanmax(np.abs(Bf - B[m])))
    res["block_dev"] = blk

    # ---------------- full panel, frozen spec (no bootstrap bands: point signal only)
    spec = frozen.load()
    h = frozen.committed_spec_hash()
    fpanel = data.load_holdout_panel(h)
    ffeat = features.compute(fpanel)
    fobs = ffeat.obs
    point = pipeline.model_from_dict(spec["stock_model"])
    z_map = scoring.QuantileMap.from_dict(spec["z_map"])
    m_map = scoring.QuantileMap.from_dict(spec["m_map"])
    sig = mdl.compute_signal(point, fobs, z_map, m_map, spec["dev_zero_rate"])
    M, BL, MS = market_vars(sig, fobs)
    res["dates_full"] = fobs.dates.values.astype("datetime64[D]").astype(str)
    res["R_full"] = validate.basket_returns(ffeat)
    res["el_full"] = fobs.eligible.any(axis=1)
    res["M_frz"], res["BL_frz"], res["MS_frz"] = M, BL, MS
    # the published file, as a check
    mf = pd.read_csv("/home/user/fear-signal/outputs/holdout/market_fear.csv", parse_dates=["date"]).set_index("date")
    Mpub = mf["M"].reindex(fobs.dates).to_numpy()
    print("max |M - published|", np.nanmax(np.abs(Mpub - M)))
    # frozen M on the dev panel too (in-sample dev reference)
    sigd = mdl.compute_signal(point, obs, z_map, m_map, spec["dev_zero_rate"])
    Md, BLd, MSd = market_vars(sigd, obs)
    res["M_frz_dev"], res["BL_frz_dev"], res["MS_frz_dev"] = Md, BLd, MSd
    np.savez_compressed(f"{OUT}/cache.npz", **res)
    print("saved", len(res["dates_dev"]), len(res["dates_full"]))


if __name__ == "__main__":
    main()
