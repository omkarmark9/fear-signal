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
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from . import bootstrap, data, evaluate, features, frozen, hmm, model as mdl, scoring, stats, validate
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
    A = A / A.sum(axis=-1, keepdims=True)
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

@dataclass
class Applied:
    """A frozen spec applied to data: the signal and its parameter bands.

    ``M_band`` is the band of the headline M; ``Mpct_band`` the band of the
    percentile score M_pct that alerts and conviction use. For a v2 spec the
    two are the same object.
    """

    sig: mdl.Signal
    M_band: np.ndarray
    Z_band: np.ndarray
    Mpct_band: np.ndarray

    def __iter__(self):  # (sig, M_band, Z_band), as v2 callers unpacked it
        return iter((self.sig, self.M_band, self.Z_band))


def _linear(d: dict | None) -> tuple[float, float] | None:
    return None if d is None else (float(d["mu"]), float(d["sd"]))


def apply(spec_doc: dict, feat: features.Features) -> Applied:
    """Signal and replicate bands from a frozen spec (no fitting of any kind)."""
    obs = feat.obs
    point = model_from_dict(spec_doc["stock_model"])
    z_map = scoring.QuantileMap.from_dict(spec_doc["z_map"])
    m_map = scoring.QuantileMap.from_dict(spec_doc["m_map"])
    z0 = spec_doc["dev_zero_rate"]
    sig = mdl.compute_signal(point, obs, z_map, m_map, z0, _linear(spec_doc.get("market_scale")))
    reps = spec_doc["bootstrap"]["replicates"]
    M_reps = np.empty((len(reps), len(obs.dates)))
    P_reps = np.empty((len(reps), len(obs.dates)))
    Z_reps = np.empty((len(reps),) + obs.x.shape, dtype=np.float32)
    for j, r in enumerate(reps):
        rs = mdl.compute_signal(model_from_dict(r["model"]), obs,
                                scoring.QuantileMap.from_dict(r["z_map"]),
                                scoring.QuantileMap.from_dict(r["m_map"]), z0,
                                _linear(r.get("market_scale")))
        M_reps[j] = rs.M
        P_reps[j] = rs.M_pct
        Z_reps[j] = rs.Z
    lvl = spec_doc["bootstrap"]["band_level"]
    qs = [(1 - lvl) / 2, 0.5, (1 + lvl) / 2]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # ineligible cells are all-NaN
        M_band = np.nanquantile(M_reps, qs, axis=0)
        P_band = np.nanquantile(P_reps, qs, axis=0)
        Z_band = np.nanquantile(Z_reps, qs, axis=0)
    return Applied(sig, M_band, Z_band, P_band)


def signal_frames(sig: mdl.Signal, M_band, Z_band, obs, state_names,
                  Mpct_band=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Market and stock output tables. With ``Mpct_band`` (v3) the market table
    carries the linear headline M and the percentile score separately."""
    dates = obs.dates
    if Mpct_band is None:   # v2 layout, unchanged
        mk = pd.DataFrame({
            "M": sig.M, "M_lo": M_band[0], "M_hi": M_band[2], "F_M": scoring.fear_score(sig.M),
            "breadth": sig.breadth, "M_legacy": sig.M_legacy,
            "band": scoring.band(np.nan_to_num(sig.M, nan=-9)),
            "n_stocks": obs.eligible.sum(axis=1),
        }, index=dates)
    else:
        mk = pd.DataFrame({
            "M": sig.M, "M_lo": M_band[0], "M_hi": M_band[2],
            "M_pct": sig.M_pct, "M_pct_lo": Mpct_band[0], "M_pct_hi": Mpct_band[2],
            "F_M": scoring.fear_score(sig.M_pct),
            "band": scoring.band(np.nan_to_num(sig.M_pct, nan=-9)),
            "high_conviction": np.nan_to_num(Mpct_band[0], nan=-9) >= 2.0,
            "breadth": sig.breadth, "M_legacy": sig.M_legacy,
            "n_stocks": obs.eligible.sum(axis=1),
        }, index=dates)
    mk.loc[~np.isfinite(sig.M_pct), "band"] = ""
    mk.index.name = "date"
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


def _metrics(spec_doc, feat, mask_from, vxn, applied=None, mask_to=None) -> dict:
    obs = feat.obs
    applied = applied if applied is not None else apply(spec_doc, feat)
    sig = applied.sig
    el_days = obs.eligible.any(axis=1)
    mask = el_days if mask_from is None else el_days & (obs.dates >= pd.Timestamp(mask_from))
    if mask_to is not None:
        mask &= obs.dates <= pd.Timestamp(mask_to)
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
        "alerts": validate.alerts(sig, applied.Mpct_band[0], vxn, obs.dates, mask,
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


# ============================================================================= v3
# docs/V3.md: the 2026 holdout is spent. v3 changes one thing selected on dev
# (the market scale), is re-estimated on everything through 2026-11-23, and is
# judged once, prospectively, at a fixed sample size.

V3_CHANGES = [
    {"id": "v3-1", "layer": "market",
     "change": "headline M = (logit breadth - mu) / sd on a fixed linear yardstick, replacing "
               "v2's percentile score Phi^-1(F_dev(logit breadth)); the percentile score is kept "
               "as M_pct and still defines bands, alerts and conviction",
     "evidence": "dev walk-forward OOS, 363 within-fold 5-session changes: v2's shipped percentile "
                 "score minus the dev-selected logit breadth = -0.043 Spearman with VXN (s.e. 0.015; "
                 "all three folds negative; calibrated SPA family gate p < 0.0034)"},
    {"id": "v3-2", "layer": "evaluation",
     "change": "H4' uses HAR and HAR+M re-estimated daily on an expanding window (rows s <= t-5), "
               "tested with Clark-West; the frozen-coefficient design is reported as secondary",
     "evidence": "standard recursive practice and the correct test for nested models; dev evidence "
                 "for a gain in the deployed configuration is null (folds B+C t 1.25)"},
]


def _market_scale(BL: np.ndarray) -> dict:
    v = BL[np.isfinite(BL)]
    return {"mu": float(v.mean()), "sd": float(v.std(ddof=1)), "n_days": int(len(v))}


def freeze_v3(n_boot: int = 200, seed: int = 20261123, path: Path = frozen.FROZEN_V3_PATH) -> dict:
    """Re-estimate the selected v3 specification on 2023-01-03 .. 2026-11-23.

    Estimation only: every choice was made on dev (docs/V3.md), so using 2026
    here fits parameters, it does not select anything.
    """
    from .config import PROSPECTIVE_SESSIONS, PROSPECTIVE_START, V3_ESTIMATION_END

    v2 = frozen.load()
    panel = data.load_v3_estimation_panel()
    feat = features.compute(panel)
    obs = feat.obs
    spec = _spec("S1_v1_refit")
    point = mdl.fit_stock_model(spec, obs, n_random=4, seed=seed, max_iter=6000, tol=1e-11)
    if not point.converged:
        raise RuntimeError("point estimate did not converge; refusing to freeze")
    z_map, m_map = mdl.fit_maps(point, obs)
    z0 = mdl.zero_rate(obs)
    L, BL = _calibration_values(point, obs)
    scale = _market_scale(BL)
    reps = bootstrap.replicate_fits(point, obs, n_boot=n_boot, seed=seed, max_iter=3000, tol=1e-9)
    rep_entries = []
    for r in reps:
        Lr, BLr = _calibration_values(r, obs)
        rep_entries.append({"model": model_to_dict(r, 6),
                            "z_map": scoring.QuantileMap.fit(Lr, dz=0.05).to_dict(4),
                            "m_map": scoring.QuantileMap.fit(BLr, dz=0.05).to_dict(4),
                            "market_scale": _market_scale(BLr)})
    sig = mdl.compute_signal(point, obs, z_map, m_map, z0, (scale["mu"], scale["sd"]))
    y_tail = mdl.tail_event(obs)
    vxn = evaluate.load_vxn(data.RAW / "vxn_full.csv", dev_only=False)
    vxn = vxn[vxn.index <= V3_ESTIMATION_END]
    el_days = obs.eligible.any(axis=1)
    R = validate.basket_returns(feat)
    doc = {
        "version": "3.0.0",
        "era_start": str(ERA_START.date()),
        "estimation_end": str(V3_ESTIMATION_END.date()),
        "prospective_start": str(PROSPECTIVE_START.date()),
        "prospective_sessions": PROSPECTIVE_SESSIONS,
        "comparators": {"v2_frozen_sha256": frozen.committed_spec_hash(), "v1": "published parameters"},
        "changes_from_v2": V3_CHANGES,
        "data_manifest_sha256": {
            n: json.loads((ROOT / "data" / f"manifest_{n}.json").read_text())["sha256_csv_6dp"]
            for n in ("dev", "holdout")},
        "universe": list(panel.tickers),
        "stock_spec": asdict(spec),
        "stock_model": model_to_dict(point),
        "stock_model_fit": {"loglik": point.loglik, "n_obs": point.n_obs,
                            "iters": len(point.history), "converged": point.converged},
        "state_names": list(STATE_NAMES),
        "dev_zero_rate": z0,
        "dev_tail_climatology": float(np.nanmean(y_tail)),
        "z_map": z_map.to_dict(),
        "m_map": m_map.to_dict(),
        "market_scale": scale,
        "v1_reference_maps": v2["v1_reference_maps"],
        "bootstrap": {"block": bootstrap.BLOCK, "n_boot": n_boot, "seed": seed,
                      "band_level": BAND_LEVEL,
                      "n_converged": int(sum(r.converged for r in reps)),
                      "parameter_table": bootstrap.parameter_table(point, reps),
                      "replicates": rep_entries},
        "har": validate.har_fit(sig, R, el_days),
        "vxn_elevated_threshold": float(np.quantile(vxn.reindex(obs.dates[el_days]).dropna(), 0.80)),
        "alert_threshold_M": 2.0,
    }
    path.write_text(json.dumps(doc, indent=1, default=float) + "\n")
    return doc


def posthoc_2026_market_scale() -> dict:
    """Labelled post-hoc diagnostic of change v3-1 on the spent 2026 holdout.

    Uses v2's frozen parameters and maps with the linear yardstick estimated on
    v2's dev period, so the only difference from v2 is the scale itself.
    Not out-of-sample evidence for v3: the 2026 result was known when v3 was built.
    """
    v2 = frozen.load()
    dev_feat = features.compute(data.load_dev_panel())
    point = model_from_dict(v2["stock_model"])
    L, BL = _calibration_values(point, dev_feat.obs)
    scale = _market_scale(BL)
    panel = data.load_holdout_panel(frozen.committed_spec_hash())
    feat = features.compute(panel)
    sig_lin = mdl.compute_signal(point, feat.obs, scoring.QuantileMap.from_dict(v2["z_map"]),
                                 scoring.QuantileMap.from_dict(v2["m_map"]), v2["dev_zero_rate"],
                                 (scale["mu"], scale["sd"]))
    vm = v2["v1_reference_maps"]
    v1 = mdl.compute_signal(_v1_model(), feat.obs, scoring.QuantileMap.from_dict(vm["z_map"]),
                            scoring.QuantileMap.from_dict(vm["m_map"]), v2["dev_zero_rate"])
    vxn = evaluate.load_vxn(data.RAW / "vxn_full.csv", dev_only=False)
    mask = feat.obs.eligible.any(axis=1) & (feat.obs.dates >= HOLDOUT_START)
    v = vxn.reindex(feat.obs.dates).to_numpy()
    d = {k: validate._d5(s, v, feat.obs.dates, mask)
         for k, s in (("v3_linear", sig_lin.M), ("v2_percentile", sig_lin.M_pct), ("v1", v1.M_legacy))}
    out = {"label": "2026, spent holdout, post-hoc diagnostic, not out-of-sample evidence",
           "dev_market_scale": scale}
    for k, df in d.items():
        out[k] = {"d5_spearman": stats.corr_ci(df.s, df.v, "spearman"),
                  "d5_pearson": stats.corr_ci(df.s, df.v, "pearson")}
    j = d["v3_linear"].index.intersection(d["v1"].index)
    out["v3_linear_minus_v1"] = stats.corr_diff_ci(d["v3_linear"].loc[j, "s"], d["v1"].loc[j, "s"],
                                                   d["v3_linear"].loc[j, "v"])
    out["v3_linear_minus_v2_percentile"] = stats.corr_diff_ci(
        d["v3_linear"].loc[j, "s"], d["v2_percentile"].loc[j, "s"], d["v3_linear"].loc[j, "v"])
    out["alert_days_identical"] = bool(np.array_equal(
        np.nan_to_num(sig_lin.M_pct, nan=-9) >= 2,
        np.nan_to_num(apply(v2, feat).sig.M, nan=-9) >= 2))
    return out


def v3_reference(spec3: dict) -> dict:
    """In-sample reference metrics for v3 on its estimation period (not evidence)."""
    feat = features.compute(data.load_v3_estimation_panel())
    vxn = evaluate.load_vxn(data.RAW / "vxn_full.csv", dev_only=False)
    applied = apply(spec3, feat)
    res = _metrics(spec3, feat, mask_from=None, vxn=vxn, applied=applied)
    R = validate.basket_returns(feat)
    res["har_recursive"] = validate.har_recursive(applied.sig, R, feat.obs.eligible.any(axis=1))
    return res


def prospective(presented_sha256: str, rerun_reason: str | None = None) -> dict:
    """The single pre-registered read of v3 (docs/PREREGISTRATION_v3.md).

    Scores the first PROSPECTIVE_SESSIONS sessions after 2026-11-23 for v3,
    and re-reads v2 (frozen) and v1 (published) on the same sample.
    """
    from .config import PREREGISTERED_SHA256 as PRE, PROSPECTIVE_START

    out = OUT / "prospective"
    log = out / "RUNLOG.md"
    attempted = log.exists() and "started" in log.read_text()
    if ((out / "prospective_results.json").exists() or attempted) and not rerun_reason:
        raise RuntimeError("the prospective read has already been attempted; pass a reason to repeat it")
    # every spec must be the committed, pre-registered one
    if presented_sha256 != PRE["frozen_v3.json"]:
        raise data.HoldoutLocked("presented hash is not the pre-registered v3 spec")
    if frozen.committed_spec_hash() != PRE["frozen.json"]:
        raise data.HoldoutLocked("v2 comparator spec differs from the pre-registered one")
    chal_present = CHALLENGER_F5_PATH.exists()
    if chal_present and frozen.committed_spec_hash(CHALLENGER_F5_PATH) != PRE["frozen_v3_challenger_F5.json"]:
        raise data.HoldoutLocked("F5 challenger spec differs from the pre-registered one")
    for name in ("dev", "holdout", "prospective"):
        if not data.verify(name):
            raise RuntimeError(f"data/raw/{name}.parquet does not match data/manifest_{name}.json")
    spec3 = json.loads(frozen.FROZEN_V3_PATH.read_text())
    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M UTC")
    out.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as fh:      # logged before any prospective bar is loaded
        if fh.tell() == 0:
            fh.write("# Prospective read log\n\n| when | event | frozen v3 sha256 | reason |\n|---|---|---|---|\n")
        fh.write(f"| {stamp} | started | `{presented_sha256[:16]}…` | "
                 f"{rerun_reason or 'the single pre-registered read'} |\n")
    panel = data.load_prospective_panel(presented_sha256)
    if list(panel.tickers) != list(spec3["universe"]):
        raise ValueError("prospective universe differs from the frozen v3 universe")
    feat = features.compute(panel)
    vxn = evaluate.load_vxn(data.RAW / "vxn_prospective.csv", dev_only=False)
    a3 = apply(spec3, feat)
    res = {"v3": _metrics(spec3, feat, mask_from=PROSPECTIVE_START, vxn=vxn, applied=a3)}
    a2 = apply(frozen.load(), feat)
    res["v2_frozen"] = _metrics(frozen.load(), feat, mask_from=PROSPECTIVE_START, vxn=vxn, applied=a2)
    mask = feat.obs.eligible.any(axis=1) & (feat.obs.dates >= PROSPECTIVE_START)
    v = vxn.reindex(feat.obs.dates).to_numpy()
    a = validate._d5(a3.sig.M, v, feat.obs.dates, mask)
    b = validate._d5(a2.sig.M, v, feat.obs.dates, mask)
    j = a.index.intersection(b.index)
    res["H1b_v3_minus_v2"] = stats.corr_diff_ci(a.loc[j, "s"], b.loc[j, "s"], a.loc[j, "v"])
    # P-H1's pass rule (PREREGISTRATION_v3.md): lower bound of the 97.5% interval > 0.30
    ci975 = stats.corr_ci(a["s"], a["v"], "spearman", n_boot=2000, level=0.975)
    res["P_H1"] = {"d5_spearman_ci975": ci975, "pass": bool(ci975["lo"] > 0.30)}
    # P-H1b against v1 under an unambiguous name (the shared _metrics key says "v2M")
    res["P_H1b_v3_minus_v1"] = res["v3"]["vxn"]["H1b_d5_spearman_v2M_minus_v1legacy"]
    res["P_H2_detail"] = validate.tail_calibration_detail(a3.sig, feat, mask)
    res["H4_recursive"] = validate.har_recursive(a3.sig, validate.basket_returns(feat), mask)
    if chal_present:
        res["P_F5_challenger"] = challenger_comparison(
            spec3, json.loads(CHALLENGER_F5_PATH.read_text()), feat, mask, vxn)
    res["frozen_v3_sha256"] = presented_sha256
    res["data_manifest_sha256"] = {
        n: json.loads((ROOT / "data" / f"manifest_{n}.json").read_text())["sha256_csv_6dp"]
        for n in ("dev", "holdout", "prospective")}
    (out / "prospective_results.json").write_text(json.dumps(res, indent=2, default=float) + "\n")
    mk, st = signal_frames(a3.sig, a3.M_band, a3.Z_band, feat.obs, spec3["state_names"], a3.Mpct_band)
    mk.round(4).to_csv(out / "market_fear_v3.csv")
    num = st.select_dtypes("number").columns
    st.assign(**st[num].round(4)).to_csv(out / "stock_fear_v3.csv.gz", index=False, compression="gzip")
    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M UTC")
    with open(log, "a") as fh:
        fh.write(f"| {stamp} | completed | `{presented_sha256[:16]}…` | {res['v3']['last_day']} |\n")
    return res


CHALLENGER_F5_PATH = ROOT / "config" / "frozen_v3_challenger_F5.json"


def freeze_challenger_f5(seed: int = 20261123) -> dict:
    """Freeze family F5 (shock-type-dependent persistence) as a prospective challenger.

    F5 passed the dev log-score gate (+0.0024 nats/obs, 2.2 s.e., calibrated
    SPA p = 0.006) but failed two pre-declared adoption criteria: the
    mechanism signature held only in part, and the market layer's VXN
    tracking got worse. It is therefore not adopted. It is frozen on the same
    estimation period and pre-registered as a challenger to v3's stock layer
    on the prospective sample (docs/PREREGISTRATION_v3.md, P-F5).
    """
    panel = data.load_v3_estimation_panel()
    obs = features.compute(panel).obs
    m = mdl.fit_stock_model(mdl.F5_SPEC, obs, n_random=4, seed=seed, max_iter=6000, tol=1e-11)
    if not m.converged:
        raise RuntimeError("challenger did not converge; refusing to freeze")
    doc = {"family": "F5", "spec": asdict(mdl.F5_SPEC), "model": model_to_dict(m),
           "fit": {"loglik": m.loglik, "iters": len(m.history), "converged": m.converged,
                   "n_obs": m.n_obs},
           "regime_rule": {"x": features.SYSTEMIC_X, "share": features.SYSTEMIC_SHARE},
           "estimation_zero_rate": mdl.zero_rate(obs),
           "dev_evidence": "outputs/dev/family_F5.json"}
    CHALLENGER_F5_PATH.write_text(json.dumps(doc, indent=1, default=float) + "\n")
    return doc


def challenger_comparison(spec3: dict, chal: dict, feat: features.Features, mask: np.ndarray,
                          vxn: pd.Series | None = None) -> dict:
    """P-F5: daily one-step log-score gain of the challenger over v3's stock model,
    and (with ``vxn``) the paired difference in five-session-change Spearman of
    logit breadth with VXN, the market-layer condition."""
    obs = feat.obs
    z0 = spec3["dev_zero_rate"]
    inc = mdl.as_scoring_model(model_from_dict(spec3["stock_model"]), z0).filter(obs)
    ch = mdl.as_scoring_model(model_from_dict(chal["model"]), z0).filter(obs)
    days = mask & obs.eligible.any(axis=1)
    n = obs.eligible[days].sum(axis=1)
    d = (ch.logc[days].sum(axis=1) - inc.logc[days].sum(axis=1))
    gain = float(d.sum() / n.sum())
    se = float(stats.newey_west_se_mean(d, 10) * len(d) / n.sum())
    t = gain / se if se > 0 else 0.0      # identical models: no evidence either way
    out = {"gain_per_obs": gain, "nw10_se_per_obs": se, "t": t,
           "p_one_sided": float(sp_stats.norm.sf(t)),
           "n_days": int(days.sum()), "n_obs": int(n.sum())}
    if vxn is not None:
        v = vxn.reindex(obs.dates).to_numpy()
        bl = {k: mdl.compute_signal(model_from_dict(d), obs, None, None, z0).breadth_L
              for k, d in (("chal", chal["model"]), ("v3", spec3["stock_model"]))}
        a = validate._d5(bl["chal"], v, obs.dates, days)
        b = validate._d5(bl["v3"], v, obs.dates, days)
        j = a.index.intersection(b.index)
        diff = stats.corr_diff_ci(a.loc[j, "s"], b.loc[j, "s"], a.loc[j, "v"])
        se_d = (diff["hi"] - diff["lo"]) / (2 * 1.96)
        out["market_d5_spearman_chal_minus_v3"] = diff
        out["market_condition_met"] = bool(diff["diff"] >= -se_d)
        out["claim_F5_better"] = bool(out["p_one_sided"] < 0.05 and out["market_condition_met"])
    return out


def posthoc_2026_f5() -> dict:
    """Labelled post-hoc 2026 diagnostic for F5, after its decision was final.

    F5 fitted on dev (as in the walk-forward: n_random=2, seed=0) against v2's
    frozen S1 on 2026: log-score gain and the market-layer difference.
    """
    v2 = frozen.load()
    dev = features.compute(data.load_dev_panel()).obs
    f5 = mdl.fit_stock_model(mdl.F5_SPEC, dev, n_random=2, seed=0)
    feat = features.compute(data.load_holdout_panel(frozen.committed_spec_hash()))
    mask = feat.obs.dates >= HOLDOUT_START
    ll = challenger_comparison({"stock_model": v2["stock_model"], "dev_zero_rate": v2["dev_zero_rate"]},
                               {"model": model_to_dict(f5)}, feat, mask)
    vxn = evaluate.load_vxn(data.RAW / "vxn_full.csv", dev_only=False).reindex(feat.obs.dates).to_numpy()
    bl = {k: mdl.compute_signal(m, feat.obs, None, None, v2["dev_zero_rate"]).breadth_L
          for k, m in (("f5", f5), ("s1", model_from_dict(v2["stock_model"])))}
    el = feat.obs.eligible.any(axis=1) & mask
    a = validate._d5(bl["f5"], vxn, feat.obs.dates, el)
    b = validate._d5(bl["s1"], vxn, feat.obs.dates, el)
    j = a.index.intersection(b.index)
    return {"label": "2026, spent holdout, post-hoc diagnostic for family F5 (decision already final: not adopted)",
            "f5_fit_on_dev": {"converged": f5.converged, "A": f5.A.tolist(), "s": f5.emission.s.tolist()},
            "log_score_gain_vs_v2_frozen_S1": ll,
            "market_logit_breadth_d5_spearman_f5_minus_s1": stats.corr_diff_ci(
                a.loc[j, "s"], b.loc[j, "s"], a.loc[j, "v"])}


def v3_insample_outputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """v3's market and stock tables on its own estimation window (in-sample)."""
    spec3 = json.loads(frozen.FROZEN_V3_PATH.read_text())
    feat = features.compute(data.load_v3_estimation_panel())
    ap = apply(spec3, feat)
    return signal_frames(ap.sig, ap.M_band, ap.Z_band, feat.obs, spec3["state_names"], ap.Mpct_band)
