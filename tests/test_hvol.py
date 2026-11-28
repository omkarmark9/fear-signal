import math

import numpy as np
import pandas as pd
import pytest

from fearsignal.trading import hvol, options


def _sessions():
    d = pd.bdate_range("2024-01-02", "2026-12-31")
    hol = pd.to_datetime(["2025-04-18", "2026-06-19", "2024-03-29", "2026-04-03"])
    return d[~d.isin(hol)]


def test_expiries_follow_holidays():
    e = hvol.expiries(_sessions())
    assert len(e) == 34 and e[0] == pd.Timestamp("2024-02-16") and e[-1] == pd.Timestamp("2026-11-20")
    moved = [x for x in e if x.weekday() != 4]
    assert moved == [pd.Timestamp("2025-04-17"), pd.Timestamp("2026-06-18")]


def test_entry_session_is_last_session_28_days_before():
    s = _sessions()
    assert hvol.entry_session(s, pd.Timestamp("2024-02-16")) == pd.Timestamp("2024-01-19")
    # 2026-06-18 - 28 days = 2026-05-21 (a Thursday)
    assert hvol.entry_session(s, pd.Timestamp("2026-06-18")) == pd.Timestamp("2026-05-21")


def test_strike_candidates_nearest_first():
    assert hvol.strike_candidates(174.23) == [174.0, 175.0, 170.0]
    assert hvol.strike_candidates(412.7) == [412.5, 413.0, 415.0, 410.0]
    c = hvol.strike_candidates(21.26)
    assert c[0] == 21.5 and 21.0 in c and 20.0 in c


def test_straddle_net():
    assert hvol.straddle_net(110.0, 100.0, 6.0, 4.0) == pytest.approx(-0.05)
    assert hvol.straddle_net(80.0, 100.0, 6.0, 4.0) == pytest.approx(1.0 - 0.05)


def _panel(n_months=24, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for m in range(n_months):
        for k in range(10):
            liv = math.log(0.12) + 0.3 * rng.standard_normal()
            lst = liv + 0.2 * rng.standard_normal()
            lrv = 0.0 + 0.5 * liv + 0.5 * lst + 0.15 * rng.standard_normal()
            rows.append({"month": f"m{m:02d}", "name": f"N{k}", "iv_tot": math.exp(liv), "st_tot": math.exp(lst),
                         "rv_tot": math.exp(lrv), "hi": lst >= liv, "earnings_in_life": k < 3})
    return pd.DataFrame(rows)


def test_h_vol1_recovers_the_forecast_coefficient():
    df = _panel()
    r = hvol.h_vol1(df)
    assert r["c_log_stilde"] == pytest.approx(0.5, abs=0.1)
    assert r["p_one_sided"] < 0.01 and r["clusters"] == 24


def test_h_vol2_planted_difference_and_null_direction():
    df = _panel()
    rng = np.random.default_rng(1)
    df["net"] = np.where(df["hi"], 0.25, -0.10) + 0.3 * rng.standard_normal(len(df))
    r = hvol.h_vol2(df, B=1000)
    assert r["diff"] > 0.25 and r["p_one_sided"] < 0.01
    df["net"] = np.where(df["hi"], -0.10, 0.10) + 0.3 * rng.standard_normal(len(df))
    assert hvol.h_vol2(df, B=1000)["p_one_sided"] > 0.5


def test_h_vol3_sign():
    df = _panel()
    df["net"] = 0.3 + 0.1 * np.random.default_rng(2).standard_normal(len(df))
    r = hvol.h_vol3(df, B=500)
    assert r["mean_net"] == pytest.approx(0.3, abs=0.05) and r["p_one_sided"] < 0.01


def test_pairs_end_to_end():
    s = _sessions()
    close = pd.DataFrame(100.0, index=s, columns=["AMD"])
    e, ent = pd.Timestamp("2024-02-16"), pd.Timestamp("2024-01-19")
    close.loc[e, "AMD"] = 112.0
    r = pd.DataFrame(0.01, index=s, columns=["AMD"])
    T = (e - ent).days / 365
    C = options.price(100.0, 100.0, T, 0.40, "call", r=hvol.RATE)
    P = options.price(100.0, 100.0, T, 0.40, "put", r=hvol.RATE)
    st_panel = pd.DataFrame(0.20, index=s, columns=["AMD"])
    bars = pd.DataFrame([{"name": "AMD", "expiry": "2024-02-16", "entry_date": "2024-01-19", "spot_close": 100.0,
                          "strike": "100.0000", "call_close": C, "put_close": P, "status": "ok"},
                         {"name": "AMD", "expiry": "2024-03-15", "entry_date": "2024-02-16", "spot_close": 100.0,
                          "strike": None, "call_close": None, "put_close": None, "status": "no_contract"}])
    df, excl = hvol.pairs(bars, close, r, lambda h: st_panel, 1.0, None)
    assert excl == {"no_contract": 1} and len(df) == 1
    row = df.iloc[0]
    assert row["iv"] == pytest.approx(0.40, abs=1e-4)
    assert row["h"] == 20 and row["rv_tot"] == pytest.approx(0.01 * math.sqrt(20))
    assert row["iv_tot"] == pytest.approx(0.40 * math.sqrt(T))
    assert bool(row["hi"]) is (0.20 >= 0.40 * math.sqrt(T))
    assert row["net"] == pytest.approx((12.0 - C - P) / (C + P) - hvol.COST)
    assert row["period"] == "dev" and abs(row["parity_resid"]) < 1e-9


def _rec(name, expiry, status="ok", c=5.0, p=5.0, **kw):
    return {"name": name, "expiry": expiry, "entry_date": "2024-01-19", "spot_close": 100.0, "strike": "100.0000",
            "call_close": c, "put_close": p, "status": status, **kw}


def test_placeholder_detection():
    assert hvol.is_placeholder(_rec("A", "x", c=0.01))
    assert hvol.is_placeholder(_rec("A", "x", status="missing_bar"))
    assert hvol.is_placeholder(_rec("A", "x", call_interpolated=True))
    assert not hvol.is_placeholder(_rec("A", "x", call_interpolated=False))
    assert not hvol.is_placeholder(_rec("A", "x", status="no_contract", c=None, p=None))
    # DataFrame rows: absent flags are NaN and must not count as interpolated
    assert not hvol.is_placeholder(_rec("A", "x", call_interpolated=float("nan")))
    assert hvol.is_placeholder(_rec("A", "x", c=float("nan")))


def test_merge_passes_replaces_placeholders_only():
    p1 = [_rec("A", "e1"), _rec("B", "e1", c=0.01, p=0.01), _rec("C", "e1", status="missing_bar")]
    p2 = [_rec("B", "e1", strike="105.0000"), _rec("C", "e1", status="not_listed_at_entry", c=None, p=None)]
    m = hvol.merge_passes(p1, p2).set_index("name")
    assert list(m["source"]) == ["pass1", "pass2", "pass2"]
    assert m.loc["B", "strike"] == "105.0000" and m.loc["B", "first_pass_strike"] == "100.0000"
    assert m.loc["C", "status"] == "not_listed_at_entry"
    with pytest.raises(ValueError):
        hvol.merge_passes(p1, p2[:1])                    # a placeholder without a second-pass record
    with pytest.raises(ValueError):
        hvol.merge_passes(p1, p2 + [_rec("A", "e1")])    # a second-pass record for a good pair
