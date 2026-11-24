"""Freeze on 2023-2025, then run the frozen specification once on 2026.

``freeze``  fits everything on the development panel and writes
            ``config/frozen.json``: model parameters, quantile maps, bootstrap
            replicates (with their own maps), the dev climatology, the HAR
            coefficients and the VXN "elevated" threshold used by the tests.
``holdout`` refuses to run until that file is committed; it then loads the
            full panel, applies the frozen specification without refitting
            anything, and writes results to ``outputs/holdout``.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import bootstrap, data, evaluate, features, frozen, hmm, model as mdl, scoring, validate
from .config import DEV_END, ERA_START, HOLDOUT_START, STATE_NAMES

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"
BAND_LEVEL = 0.90


# ----------------------------------------------------------------------------- (de)serialise

def model_to_dict(m: hmm.HMM, decimals: int = 10) -> dict:
    e = m.emission
    return {"s": np.round(e.s, decimals).tolist(),
            "pi": None if e.pi is None else np.round(e.pi, decimals).tolist(),
            "zero_mode": e.zero_mode, "censor": e.censor,
            "A": np.round(m.A, decimals).tolist()}


def model_from_dict(d: dict) -> hmm.HMM:
    A = np.asarray(d["A"], float)
    A = A / A.sum(axis=1, keepdims=True)
    pi = None if d["pi"] is None else np.asarray(d["pi"], float)
    return hmm.HMM(hmm.HalfNormalEmission(np.asarray(d["s"], float), pi, d["zero_mode"], d["censor"]), A)


def _spec(name: str) -> mdl.StockSpec:
    return next(c for c in mdl.CANDIDATES if c.name == name)


# ----------------------------------------------------------------------------- freeze

def freeze(stock_spec: str, n_boot: int = 200, seed: int = 20260101,
           path: Path = frozen.FROZEN_PATH) -> dict:
    panel = data.load_dev_panel()
    feat = features.compute(panel)
    obs = feat.obs
    spec = _spec(stock_spec)
    if spec.fixed:
        raise ValueError(f"{stock_spec} has fixed published parameters; it cannot be frozen as v2")
    point = mdl.fit_stock_model(spec, obs, n_random=4, seed=seed, max_iter=6000, tol=1e-11)
    if not point.converged:
        raise RuntimeError("point estimate did not converge; refusing to freeze")
    z_map, m_map = mdl.fit_maps(point, obs)
    z0 = mdl.zero_rate(obs)
    reps = bootstrap.replicate_fits(point, obs, n_boot=n_boot, seed=seed, max_iter=3000, tol=1e-9)
    v1_z_map, v1_m_map = mdl.fit_maps(_v1_model(), obs)
    rep_entries = []
    for r in reps:
        L, BL = _calibration_values(r, obs)
        rep_entries.append({"model": model_to_dict(r, 6),
                            "z_map": scoring.QuantileMap.fit(L, dz=0.05).to_dict(4),
                            "m_map": scoring.QuantileMap.fit(BL, dz=0.05).to_dict(4)})
    sig = mdl.compute_signal(point, obs, z_map, m_map, z0)
    y_tail = mdl.tail_event(obs)
    vxn = evaluate.load_vxn()
    el_days = obs.eligible.any(axis=1)
    vxn_dev = vxn.reindex(obs.dates[el_days]).dropna()
    R = validate.basket_returns(feat)
    har = validate.har_fit(sig, R, el_days)
    spec_doc = {
        "version": "2.0.0",
        "era_start": str(ERA_START.date()), "dev_end": str(DEV_END.date()),
        "holdout_start": str(HOLDOUT_START.date()),
        "data_manifest_sha256": json.loads((ROOT / "data" / "manifest_dev.json").read_text())["sha256_csv_6dp"],
        "universe": list(panel.tickers),
        "stock_spec": asdict(spec),
        "stock_model": model_to_dict(point),
        "stock_model_fit": {"loglik": point.loglik, "n_obs": point.n_obs,
                            "iters": len(point.history), "converged": point.converged},
        "v1_reference_maps": {"z_map": v1_z_map.to_dict(), "m_map": v1_m_map.to_dict()},
        "state_names": list(STATE_NAMES[: spec.K]) if spec.K == 3 else [f"state{k}" for k in range(spec.K)],
        "dev_zero_rate": z0,
        "dev_tail_climatology": float(np.nanmean(y_tail)),
        "z_map": z_map.to_dict(),
        "m_map": m_map.to_dict(),
        "bootstrap": {"block": bootstrap.BLOCK, "n_boot": n_boot, "seed": seed,
                      "band_level": BAND_LEVEL,
                      "n_converged": int(sum(r.converged for r in reps)),
                      "parameter_table": bootstrap.parameter_table(point, reps),
                      "replicates": rep_entries},
        "har": har,
        "vxn_elevated_threshold": float(np.quantile(vxn_dev, 0.80)),
        "alert_threshold_M": 2.0,
    }
    path.write_text(json.dumps(spec_doc, indent=1, default=float) + "\n")
    return spec_doc


def _calibration_values(m: hmm.HMM, obs) -> tuple[np.ndarray, np.ndarray]:
    """Dev-period logit severities (stock-days) and logit breadths (days)."""
    s = mdl.compute_signal(m, obs, None, None, mdl.zero_rate(obs))
    return s.L[obs.eligible], s.breadth_L[np.isfinite(s.breadth_L)]


# ----------------------------------------------------------------------------- apply frozen spec

def apply(spec_doc: dict, feat: features.Features):
    """Signal and replicate bands from a frozen spec (no fitting of any kind)."""
    obs = feat.obs
    point = model_from_dict(spec_doc["stock_model"])
    z_map = scoring.QuantileMap.from_dict(spec_doc["z_map"])
    m_map = scoring.QuantileMap.from_dict(spec_doc["m_map"])
    z0 = spec_doc["dev_zero_rate"]
    sig = mdl.compute_signal(point, obs, z_map, m_map, z0)
    reps = spec_doc["bootstrap"]["replicates"]
    M_reps = np.empty((len(reps), len(obs.dates)))
    Z_reps = np.empty((len(reps),) + obs.x.shape, dtype=np.float32)
    for j, r in enumerate(reps):
        rs = mdl.compute_signal(model_from_dict(r["model"]), obs,
                                scoring.QuantileMap.from_dict(r["z_map"]),
                                scoring.QuantileMap.from_dict(r["m_map"]), z0)
        M_reps[j] = rs.M
        Z_reps[j] = rs.Z
    lvl = spec_doc["bootstrap"]["band_level"]
    qs = [(1 - lvl) / 2, 0.5, (1 + lvl) / 2]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # ineligible cells are all-NaN
        M_band = np.nanquantile(M_reps, qs, axis=0)
        Z_band = np.nanquantile(Z_reps, qs, axis=0)
    return sig, M_band, Z_band


def signal_frames(sig: mdl.Signal, M_band, Z_band, obs, state_names) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = obs.dates
    mk = pd.DataFrame({
        "M": sig.M, "M_lo": M_band[0], "M_hi": M_band[2], "F_M": scoring.fear_score(sig.M),
        "breadth": sig.breadth, "M_legacy": sig.M_legacy,
        "band": scoring.band(np.nan_to_num(sig.M, nan=-9)),
        "n_stocks": obs.eligible.sum(axis=1),
    }, index=dates)
    mk.loc[~np.isfinite(sig.M), "band"] = ""
    mk.index.name = "date"
    T, N = obs.x.shape
    tt, nn = np.nonzero(obs.eligible)
    st = pd.DataFrame({
        "date": dates[tt], "ticker": np.asarray(obs.tickers)[nn],
        "x": obs.x[tt, nn], "Z": sig.Z[tt, nn], "Z_lo": Z_band[0][tt, nn], "Z_hi": Z_band[2][tt, nn],
        "F": sig.F[tt, nn], "band": scoring.band(sig.Z[tt, nn]), "I": sig.I[tt, nn],
        "q_next": sig.q_next[tt, nn],
        **{f"P_{s}": sig.alpha[tt, nn, k] for k, s in enumerate(state_names)},
    })
    return mk, st


def dev_reference(spec_doc: dict) -> dict:
    """The pre-registered metrics computed in-sample on dev, for reference."""
    panel = data.load_dev_panel()
    feat = features.compute(panel)
    return _metrics(spec_doc, feat, mask_from=None, vxn=evaluate.load_vxn())


def _metrics(spec_doc, feat, mask_from, vxn, applied=None) -> dict:
    obs = feat.obs
    sig, M_band, _ = applied if applied is not None else apply(spec_doc, feat)
    el_days = obs.eligible.any(axis=1)
    mask = el_days if mask_from is None else el_days & (obs.dates >= pd.Timestamp(mask_from))
    R = validate.basket_returns(feat)
    q_abs = evaluate.ewma_tail_forecast(obs, spec_doc["dev_zero_rate"], power=1)
    q_sq = evaluate.ewma_tail_forecast(obs, spec_doc["dev_zero_rate"], power=2)
    baselines = {"ewma_abs": q_abs, "ewma_sq": q_sq}
    vm = spec_doc["v1_reference_maps"]
    v1 = mdl.compute_signal(  # the published v1 model, scored identically with its frozen dev maps
        _v1_model(), obs, scoring.QuantileMap.from_dict(vm["z_map"]),
        scoring.QuantileMap.from_dict(vm["m_map"]), spec_doc["dev_zero_rate"])
    return {
        "days": int(mask.sum()),
        "first_day": str(obs.dates[mask][0].date()), "last_day": str(obs.dates[mask][-1].date()),
        "vxn": validate.vxn_construct(sig, vxn, obs.dates, mask, v1=v1),
        "vxn_v1_published": validate.vxn_construct(v1, vxn, obs.dates, mask),
        "thermometer": validate.thermometer(sig, feat, mask),
        "thermometer_v1_published": validate.thermometer(v1, feat, mask),
        "tail_forecast": validate.tail_forecast(sig, feat, mask, spec_doc["dev_tail_climatology"], baselines),
        "tail_forecast_v1_published": validate.tail_forecast(v1, feat, mask, spec_doc["dev_tail_climatology"], baselines),
        "har_oos": validate.har_oos(sig, R, mask, spec_doc["har"]),
        "forward_returns_20d": validate.forward_returns(sig, R, mask, 20),
        "forward_returns_5d": validate.forward_returns(sig, R, mask, 5),
        "alerts": validate.alerts(sig, M_band[0], vxn, obs.dates, mask,
                                  spec_doc["vxn_elevated_threshold"], spec_doc["alert_threshold_M"]),
    }


def _v1_model() -> hmm.HMM:
    """The v1 starting point exactly as published (no refit)."""
    return hmm.HMM(hmm.HalfNormalEmission(mdl.V1_S.copy(), None, "none", False), mdl.V1_A.copy())


def holdout(presented_sha256: str, rerun_reason: str | None = None) -> dict:
    """The single frozen run on 2026. Writes outputs/holdout/*.

    Refuses to run a second time unless a reason is given; every run is
    appended to outputs/holdout/RUNLOG.md with the spec hash and the reason.
    """
    out = OUT / "holdout"
    log = out / "RUNLOG.md"
    if (out / "holdout_results.json").exists() and not rerun_reason:
        raise RuntimeError("the holdout has already been run; pass a rerun reason to run it again")
    if not data.verify("dev"):
        raise RuntimeError("data/raw/dev.parquet does not match data/manifest_dev.json")
    panel = data.load_holdout_panel(presented_sha256)
    spec_doc = frozen.load()
    feat = features.compute(panel)
    vxn = evaluate.load_vxn(data.RAW / "vxn_full.csv", dev_only=False)
    applied = apply(spec_doc, feat)
    res = _metrics(spec_doc, feat, mask_from=spec_doc["holdout_start"], vxn=vxn, applied=applied)
    res["frozen_spec_sha256"] = presented_sha256
    res["holdout_manifest_sha256"] = json.loads(
        (ROOT / "data" / "manifest_holdout.json").read_text())["sha256_csv_6dp"]
    sig, M_band, Z_band = applied
    mk, st = signal_frames(sig, M_band, Z_band, feat.obs, spec_doc["state_names"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "holdout_results.json").write_text(json.dumps(res, indent=2, default=float) + "\n")
    mk.round(4).to_csv(out / "market_fear.csv")
    num = st.select_dtypes("number").columns
    st.assign(**st[num].round(4)).to_csv(out / "stock_fear.csv.gz", index=False, compression="gzip")
    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M UTC")
    with open(log, "a") as fh:
        if fh.tell() == 0:
            fh.write("# Holdout run log\n\n| when | frozen spec sha256 | last session | reason |\n|---|---|---|---|\n")
        fh.write(f"| {stamp} | `{presented_sha256[:16]}…` | {res['last_day']} | "
                 f"{rerun_reason or 'first and only planned run'} |\n")
    return res
