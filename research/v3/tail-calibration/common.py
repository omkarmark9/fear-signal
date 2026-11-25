"""Shared data for the tail-calibration study (scratch; never writes to the repo).

build() caches, in /tmp/claude-0/v3/tail-calibration/cache.npz:
  dev OOS (stitched walk-forward test days, per-fold S1 params from the shared cache):
    dev_q      (T,N) q scored on day t (forecast made at t-1)
    dev_pred   (T,N,K) one-step predictive state law used for that q
    dev_tailk  (3,K) per-fold state tail probs (1-z0) 2Phi(-3/s_k)
    dev_y      (T,N) tail event
    dev_block  (T,) fold id of test days (-1 = not scored)
    dev_qewma  (T,N) |x| EWMA baseline with the fold's z0
    dev_clim   (T,N) fold training climatology
  2026 (frozen spec, diagnostics only): h_q, h_pred, h_y, h_qewma, h_x, h_kind, h_sigma, h_r, mask
"""
import os
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("GIT_OPTIONAL_LOCKS", "0")
import sys
sys.dont_write_bytecode = True
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from fearsignal import data, evaluate, features, frozen, model as mdl, pipeline

D = Path("/tmp/claude-0/v3/tail-calibration")
CACHE = D / "cache.npz"
FOLDS_PKL = Path("/tmp/claude-0/v3/shared/fold_models_S1.pkl")


def shift_to_scored_day(a):
    """value at t-1 placed on day t (the day the forecast is scored)."""
    pad = np.full((1,) + a.shape[1:], np.nan)
    return np.concatenate([pad, a[:-1]], axis=0)


def q_and_pred(model, obs, z0):
    """q_next and predictive law, both aligned to the scored day."""
    fr = model.filter(obs)
    el = obs.eligible
    alpha = np.where(el[:, :, None], fr.alpha, np.nan)
    pred_next = np.nan_to_num(alpha) @ model.A                       # P(S_{t+1} | x_<=t)
    pred_next = np.where(el[:, :, None], pred_next, np.nan)
    tk = mdl.as_scoring_model(model, z0).emission.tail_prob(mdl.TAIL_THRESHOLD)
    q = pred_next @ tk
    return shift_to_scored_day(q), shift_to_scored_day(pred_next), tk


def build(force=False):
    if CACHE.exists() and not force:
        return dict(np.load(CACHE, allow_pickle=True))
    out = {}
    # ---------------- dev
    panel = data.load_dev_panel()
    feat = features.compute(panel)
    obs = feat.obs
    T, N = obs.x.shape
    y = mdl.tail_event(obs)
    folds = pickle.load(open(FOLDS_PKL, "rb"))["folds"]
    q = np.full((T, N), np.nan)
    pred = np.full((T, N, 3), np.nan)
    qe = np.full((T, N), np.nan)
    clim = np.full((T, N), np.nan)
    block = np.full(T, -1)
    tailk = np.zeros((3, 3))
    for b, f in enumerate(folds):
        tr_end, te_end = pd.Timestamp(f["train_end"]), pd.Timestamp(f["test_end"])
        test = (obs.dates > tr_end) & (obs.dates <= te_end)
        block[test] = b
        qq, pp, tk = q_and_pred(f["model"], obs, f["zero_rate"])
        tailk[b] = tk
        q[test], pred[test] = qq[test], pp[test]
        qe[test] = evaluate.ewma_tail_forecast(obs, f["zero_rate"], power=1)[test]
        ntr = int((obs.dates <= tr_end).sum())
        clim[test] = np.where(obs.eligible[test], np.nanmean(y[:ntr]), np.nan)
    out.update(dev_q=q, dev_pred=pred, dev_tailk=tailk, dev_y=y, dev_block=block, dev_qewma=qe,
               dev_clim=clim, dev_x=obs.x, dev_kind=obs.kind, dev_sigma=feat.sigma, dev_r=feat.r,
               dev_dates=obs.dates.values.astype("datetime64[D]").astype(str),
               tickers=np.array(obs.tickers))
    # ---------------- 2026 (diagnostic only; frozen spec, nothing refit)
    hp = data.load_holdout_panel(frozen.committed_spec_hash())
    hf = features.compute(hp)
    ho = hf.obs
    spec = frozen.load()
    m = pipeline.model_from_dict(spec["stock_model"])
    z0 = spec["dev_zero_rate"]
    hq, hpred, htk = q_and_pred(m, ho, z0)
    hy = mdl.tail_event(ho)
    hqe = evaluate.ewma_tail_forecast(ho, z0, power=1)
    # the frozen run's tail_forecast mask also requires both EWMA baselines finite
    hqe2 = evaluate.ewma_tail_forecast(ho, z0, power=2)
    mask = (ho.dates >= pd.Timestamp(spec["holdout_start"]))
    out.update(h_q=hq, h_pred=hpred, h_tailk=htk, h_y=hy, h_qewma=hqe, h_qewma2=hqe2,
               h_x=ho.x, h_kind=ho.kind, h_sigma=hf.sigma, h_r=hf.r, h_mask=mask,
               h_dates=ho.dates.values.astype("datetime64[D]").astype(str),
               h_tickers=np.array(ho.tickers), h_clim=np.array(spec["dev_tail_climatology"]),
               frozen_s=np.array(spec["stock_model"]["s"]), frozen_A=np.array(spec["stock_model"]["A"]),
               frozen_z0=np.array(z0))
    np.savez_compressed(CACHE, **out)
    return dict(np.load(CACHE, allow_pickle=True))


if __name__ == "__main__":
    c = build(force=True)
    print({k: getattr(v, "shape", v) for k, v in c.items()})
