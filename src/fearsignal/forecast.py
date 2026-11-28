"""Next-session forecasts from the frozen v3 model, with honest error ranges.

The model's own outputs are probabilities: q_next per stock, and its filter
state for the market. This module turns them into checkable forecasts for
the next session. Error ranges are calibrated on how similar situations
actually turned out in the estimation window (2023-07 .. 2026-11-23), where the
frozen v3 model's in-sample outputs are available. No prospective data is
used to make a forecast.

The forecasts:
- count: the number of stocks whose low is >= 3 own-sigmas below the
  previous close. Model mean = sum of q_next. Because one market shock hits
  every stock at once, the realized count is very over-dispersed (variance
  about 25x the mean), so the range comes from historical days with a
  similar model mean.
- stocks: q_next for every stock (the model's per-stock probability).
- market: the next-session percentile score M_pct and its band, from
  historical days with a similar M_pct.
- persistence: chance the market is back to calm (M_pct < 1) within 5
  sessions, from the same days.
- volatility: annualized 5-session realized volatility of the equal-weight
  basket. HAR + M is re-estimated recursively on rows whose targets are
  known, and the range comes from recursive out-of-sample residuals.

score() compares a stored forecast with what happened. The course-correction
rule is fixed here, before any outcome is seen (see docs/LIVE.md): a session
is "flagged" when the realized count or the next-session M_pct falls outside
its 90% range. Flags go to a scoreboard. No single session changes the
model. Five or more flags within 20 sessions (p about 0.04 if the ranges are
right) triggers a dev-only investigation under the v3 protocol. Any fix
becomes a new version with its own prospective sample; v3 stays frozen.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import scoring, validate

ROOT = Path(__file__).resolve().parents[2]
WINDOW = 0.25          # "similar" count days: model mean within +-25%
M_WINDOW = 0.30        # "similar" market days: M_pct within +-0.30
FLAG_LEVEL = 0.90      # central range used for flags
ALARM = (5, 20)        # 5+ flags within 20 sessions -> investigate

# NYSE early-close (1 pm) sessions, 2023-2026. 2026-07-02 is NOT one: July 4, 2026 is a
# Saturday, NYSE closed Friday July 3 and traded a full session on July 2 (only the bond
# market closed early); its volume was 1.11x normal against 0.32-0.63x on every true
# half-day. Corrected 2026-11-27 after the trading model's pre-backtest audit.
HALF_DAYS = pd.to_datetime(["2023-07-03", "2023-11-24", "2024-07-03", "2024-11-29", "2024-12-24",
                            "2025-07-03", "2025-11-28", "2025-12-24", "2026-11-27", "2026-12-24",
                            # 2027: only the day after Thanksgiving (July 4 is a Sunday and
                            # Christmas a Saturday, observed Monday July 5 and Friday Dec 24)
                            "2027-11-26",
                            # 2028: July 3 and the day after Thanksgiving (Christmas Eve is a Sunday)
                            "2028-07-03", "2028-11-24"])
HALF_DAYS_COVER_THROUGH = pd.Timestamp("2028-12-31")     # live code refuses dates beyond this


def history() -> tuple[pd.DataFrame, pd.Series]:
    """Daily (model mean count, realized count) and M_pct, from v3's in-sample outputs."""
    st = pd.read_csv(ROOT / "outputs/v3/stock_fear_v3_insample.csv.gz", parse_dates=["date"])
    mk = pd.read_csv(ROOT / "outputs/v3/market_fear_v3_insample.csv", index_col=0, parse_dates=True)
    q = st.pivot(index="date", columns="ticker", values="q_next")
    x = st.pivot(index="date", columns="ticker", values="x")
    pred = q.shift(1).where(x.notna()).sum(axis=1, min_count=1)
    real = (x >= 3).sum(axis=1).where(x.notna().any(axis=1))
    counts = pd.DataFrame({"pred": pred, "real": real}).dropna()
    counts = counts[counts.index > counts.index.min()]
    return counts, mk["M_pct"].dropna()


def _q(v, ps=(0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)) -> dict:
    return {f"q{int(p * 100):02d}": float(np.quantile(v, p)) for p in ps}


def count_forecast(model_mean: float, counts: pd.DataFrame) -> dict:
    sim = counts[(counts.pred >= model_mean * (1 - WINDOW)) & (counts.pred <= model_mean * (1 + WINDOW))]
    half = counts.loc[counts.index.intersection(HALF_DAYS)]
    return {"model_mean": model_mean, "similar_days": int(len(sim)),
            "realized_on_similar_days": {**_q(sim.real), "mean": float(sim.real.mean()),
                                         "p_zero": float((sim.real == 0).mean()),
                                         "p_at_least_double": float((sim.real >= 2 * model_mean).mean())},
            "half_days_history": {str(d.date()): {"model_mean": round(float(r.pred), 2), "realized": int(r.real)}
                                  for d, r in half.iterrows()}}


def market_forecast(m_pct: float, mpct: pd.Series) -> dict:
    idx = np.flatnonzero((mpct.values >= m_pct - M_WINDOW) & (mpct.values <= m_pct + M_WINDOW))
    idx = idx[idx + 5 < len(mpct)]
    nxt = mpct.values[idx + 1]
    calm5 = np.array([(mpct.values[i + 1:i + 6] < 1.0).any() for i in idx])
    bands = pd.Series(scoring.band(nxt)).value_counts(normalize=True)
    return {"m_pct_today": m_pct, "similar_days": int(len(idx)),
            "next_m_pct": _q(nxt),
            "next_band_probabilities": {k: float(v) for k, v in bands.items()},
            "p_next_acute_or_worse": float((nxt >= 2.0).mean()),
            "p_calm_within_5_sessions": float(calm5.mean())}


def vol_forecast(sig, R: np.ndarray) -> dict:
    """Recursive HAR+M forecast of annualized 5-session basket vol from the last session."""
    y, X = validate.har_design(R, sig.M)
    good = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    t = len(R) - 1
    first = int(np.argmax(good))
    res = []
    for s in np.flatnonzero(good):
        rows = np.flatnonzero(good[: max(s - 4, 0)])
        rows = rows[rows >= first]
        if len(rows) >= 60:
            b = np.linalg.lstsq(X[rows], y[rows], rcond=None)[0]
            res.append(y[s] - X[s] @ b)
    rows = np.flatnonzero(good[: max(t - 4, 0)])
    b = np.linalg.lstsq(X[rows], y[rows], rcond=None)[0]
    f = float(X[t] @ b)
    qs = np.quantile(res, [0.10, 0.50, 0.90])
    ann = lambda v: float(np.exp(v) * np.sqrt(252))
    return {"point_annualized": ann(f),
            "range_80": [ann(f + qs[0]), ann(f + qs[2])],
            "median_adjusted": ann(f + qs[1]),
            "past_5_session_realized_annualized": ann(0.5 * np.log(np.mean(R[-5:] ** 2)))}


def make(mk: pd.DataFrame, st: pd.DataFrame, sig, R: np.ndarray, target: str, data_report: dict) -> dict:
    asof = st["date"].max()
    t = st[st["date"] == asof]
    counts, mpct = history()
    m_pct = float(mk.loc[asof, "M_pct"])
    return {
        "made_from_close_of": str(asof.date()), "target_session": target,
        "target_is_half_day": pd.Timestamp(target) in HALF_DAYS,
        "model": "v3 frozen (config/frozen_v3.json), not refit", "data": data_report,
        "count_ge_3sigma": count_forecast(float(t["q_next"].sum()), counts),
        "market": market_forecast(m_pct, mpct),
        "stocks_q_next": {r.ticker: round(float(r.q_next), 4) for r in t.sort_values("q_next", ascending=False).itertuples()},
        "volatility_5_sessions": vol_forecast(sig, R),
        "flag_rule": {"level": FLAG_LEVEL, "alarm": {"flags": ALARM[0], "within_sessions": ALARM[1]}},
    }


def score(fc: dict, mk: pd.DataFrame, st: pd.DataFrame) -> dict:
    """Score a stored forecast against the realized target session."""
    target = pd.Timestamp(fc["target_session"])
    t = st[st["date"] == target]
    if t.empty:
        raise ValueError(f"no data for {target.date()} yet")
    lo_p, hi_p = (1 - FLAG_LEVEL) / 2, (1 + FLAG_LEVEL) / 2
    key = lambda p: f"q{int(round(p * 100)):02d}"
    real = int((t["x"] >= 3).sum())
    cr = fc["count_ge_3sigma"]["realized_on_similar_days"]
    m_next = float(mk.loc[target, "M_pct"])
    mr = fc["market"]["next_m_pct"]
    q = pd.Series(fc["stocks_q_next"])
    y = (t.set_index("ticker")["x"] >= 3).astype(float).reindex(q.index).dropna()
    qq = q.reindex(y.index)
    clim = 0.0338
    flags = {"count_outside_90": not (cr[key(lo_p)] <= real <= cr[key(hi_p)]),
             "market_outside_90": not (mr[key(lo_p)] <= m_next <= mr[key(hi_p)])}
    return {"target_session": str(target.date()), "realized_count_ge_3sigma": real,
            "count_range_90": [cr[key(lo_p)], cr[key(hi_p)]], "model_mean": fc["count_ge_3sigma"]["model_mean"],
            "realized_m_pct": m_next, "m_pct_range_90": [mr[key(lo_p)], mr[key(hi_p)]],
            "realized_band": str(scoring.band(np.array([m_next]))[0]),
            "stocks_hit": sorted(y.index[y == 1].tolist()),
            "brier_model": float(((qq - y) ** 2).mean()), "brier_climatology": float(((clim - y) ** 2).mean()),
            "flags": flags, "flagged": any(flags.values())}
