"""Render the holdout section of docs/RESULTS.md from the results JSON.

Every number in the holdout write-up comes from here, so the prose cannot
drift from ``outputs/holdout/holdout_results.json``.
"""

from __future__ import annotations

import json
from pathlib import Path


def _ci(d: dict, key: str = "r") -> str:
    return f"{d[key]:.3f} ({d['lo']:.2f}, {d['hi']:.2f})"


def verdicts(res: dict) -> list[tuple[str, str, str, bool | None]]:
    """(id, statistic, value, passed) for each pre-registered hypothesis."""
    v = res["vxn"]
    h1 = v["M"]["d5_spearman"]
    h1b = v["H1b_d5_spearman_v2M_minus_v1legacy"]
    tf = res["tail_forecast"]
    cal = tf["calibration"]
    th = res["thermometer"]
    har = res["har_oos"]
    fr = res["forward_returns_20d"]
    dev_sign = res.get("_dev_forward_slope_sign")
    h5_pass = fr["hodrick_1b_p"] < 0.05 and (dev_sign is None or
                                              (fr["slope_per_M"] > 0) == (dev_sign > 0))
    rows = [
        ("H1", "Spearman ρ(ΔM, ΔVXN), 5-session", _ci(h1), h1["lo"] > 0.30),
        ("H1b", "ρ(ΔM_v2) − ρ(ΔM_v1), paired", f"{h1b['diff']:+.3f} ({h1b['lo']:+.2f}, {h1b['hi']:+.2f})",
         h1b["lo"] > 0),
        ("H2", "Brier skill vs climatology; calibration slope",
         f"{tf['bss_vs_climatology']:+.2%}; slope {cal['slope']:.2f} (s.e. {cal['se_slope']:.2f}); "
         f"gain t = {tf['brier_gain_vs_clim_t']:.2f}",
         tf["bss_vs_climatology"] > 0 and 0.8 <= cal["slope"] <= 1.25),
        ("H2b", "Brier skill vs |x| EWMA (x² EWMA)",
         f"{tf['bss_vs_ewma_abs']:+.2%} ({tf['bss_vs_ewma_sq']:+.2%})", tf["bss_vs_ewma_abs"] > 0),
        ("H3", "3-sigma loss days with Z ≥ 1.5",
         f"{th['share_3sigma_loss_days_Z_ge_1.5']:.3f} of {th['n_3sigma_loss_days']}",
         th["share_3sigma_loss_days_Z_ge_1.5"] >= 0.95),
        ("H3b", "Z ≥ 2.5 readings with no 2-sigma dip in 5 sessions",
         f"{th['share_panic_readings_without_2sigma_dip_in_5d']:.3f} of {th['n_panic_readings']}",
         th["share_panic_readings_without_2sigma_dip_in_5d"] <= 0.05),
        ("H4", "HAR+M vs HAR, OOS R²; DM-HLN",
         f"{har['oos_r2_har']:.3f} → {har['oos_r2_har_M']:.3f}; t = {har['dm_hln_t']:.2f}, "
         f"p = {har['dm_hln_p_one_sided']:.3f}",
         har["mse_gain_mean"] > 0 and har["dm_hln_p_one_sided"] < 0.05),
        ("H5", "20-day return slope on M; Hodrick 1B",
         f"{fr['slope_per_M']:+.4f} per unit M; t = {fr['hodrick_1b_t']:.2f}, p = {fr['hodrick_1b_p']:.3f}",
         h5_pass),
    ]
    return rows


def render(res: dict, dev: dict | None = None, period: str = "2026") -> str:
    if dev is not None:
        res = dict(res, _dev_forward_slope_sign=dev["forward_returns_20d"]["slope_per_M"])
    v = res["vxn"]
    v1 = res["vxn_v1_published"]
    al = res["alerts"]
    out = [f"Scored sessions: {res['days']} ({res['first_day']} .. {res['last_day']}). "
           f"Frozen spec SHA-256 `{res['frozen_spec_sha256']}`.", ""]
    out += ["### Pre-registered tests", "",
            f"| id | statistic | {period} value | pass |", "|---|---|---|---|"]
    for hid, stat, val, ok in verdicts(res):
        out.append(f"| {hid} | {stat} | {val} | {'**pass**' if ok else 'fail'} |")
    out += ["", f"### Market fear against VXN, {period}", "",
            "| reading | Δ5 Spearman (95% CI) | Δ5 Pearson (95% CI) | level Spearman (95% CI) |",
            "|---|---|---|---|"]
    for label, d in (("v2 M (breadth)", v["M"]), ("v1 expanding z-score, v2 filter", v["M_legacy"]),
                     ("v1 as published (expanding z-score)", v1["M_legacy"])):
        out.append(f"| {label} | {_ci(d['d5_spearman'])} | {_ci(d['d5_pearson'])} | {_ci(d['level_spearman'])} |")
    out += ["", f"### Alerts, {period}", "",
            f"VXN 'elevated' = at or above its dev 80th percentile ({al['vxn_elevated_threshold']:.2f}); "
            f"base rate {al['base_rate_vxn_elevated']:.1%}, mean VXN {al['mean_vxn_all']:.1f}.", "",
            "| rule | days | VXN elevated on those days | mean VXN |", "|---|---|---|---|"]
    for key, label in (("M_ge_2.0", "M ≥ 2"), ("conviction_M_lo_ge_2.0", "high conviction: M_lo ≥ 2")):
        a = al[key]
        prec = "–" if a["precision_vxn_elevated"] is None else f"{a['precision_vxn_elevated']:.1%}"
        mv = "–" if a["mean_vxn"] is None else f"{a['mean_vxn']:.1f}"
        out.append(f"| {label} | {a['n_days']} | {prec} | {mv} |")
    tf, tf1 = res["tail_forecast"], res["tail_forecast_v1_published"]
    out += ["", f"### Tail forecast, {period}", "",
            f"Event rate {tf['event_rate']:.2%} over {tf['n']:,} stock-days; mean forecast "
            f"{tf['mean_forecast']:.2%}. Brier skill vs climatology: v2 {tf['bss_vs_climatology']:+.2%}, "
            f"v1 published {tf1['bss_vs_climatology']:+.2%}.", "",
            "| forecast bin | stock-days | mean forecast | observed |", "|---|---|---|---|"]
    for r in tf["reliability"]:
        out.append(f"| {r['bin']} | {r['n']:,} | {r['mean_forecast']:.3%} | {r['observed']:.3%} |")
    fr5 = res["forward_returns_5d"]
    fr20 = res["forward_returns_20d"]
    out += ["", "### Forward basket returns (secondary)", "",
            f"20-day: slope {fr20['slope_per_M']:+.4f} log-return per unit of M, Hodrick 1B "
            f"t = {fr20['hodrick_1b_t']:.2f}; mean 20-day return {fr20['mean_fwd_all']:+.2%} on all days and "
            + ("–" if fr20["mean_fwd_M_ge_2"] is None else f"{fr20['mean_fwd_M_ge_2']:+.2%}")
            + f" on the {fr20['n_days_M_ge_2']} days with M ≥ 2. "
            f"5-day: slope {fr5['slope_per_M']:+.4f}, 1B t = {fr5['hodrick_1b_t']:.2f}."]
    return "\n".join(out) + "\n"


def write(results_json: Path, results_md: Path, dev_json: Path | None = None) -> None:
    res = json.loads(Path(results_json).read_text())
    dev = json.loads(Path(dev_json).read_text()) if dev_json else None
    md = Path(results_md).read_text()
    head = md[: md.index("## 2. Holdout")]
    section = "## 2. Holdout: the single frozen run on 2026\n\n" + render(res, dev)
    Path(results_md).write_text(head + section)
