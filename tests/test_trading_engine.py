"""Trading engine: exact accounting on synthetic prices with known answers."""

import numpy as np
import pandas as pd
import pytest

from fearsignal.trading import engine


def _prices(close: np.ndarray, open_: np.ndarray | None = None, vol: float = 1e7):
    T, N = close.shape
    idx = pd.bdate_range("2024-01-01", periods=T)
    cols = [f"S{j}" for j in range(N)]
    c = pd.DataFrame(close, index=idx, columns=cols)
    o = pd.DataFrame(open_ if open_ is not None else close, index=idx, columns=cols)
    dv = pd.DataFrame(vol, index=idx, columns=cols)
    return engine.Prices(o, c, dv)


def _flat(T=60, N=3):
    return np.full((T, N), 100.0)


def test_single_cohort_return_entry_at_next_open_exit_after_H_sessions():
    close = _flat()
    open_ = close.copy()
    open_[31, 0] = 110.0          # entry open of S0 (signal at t=30, delay 1)
    close[31:, 0] = 121.0         # S0 rises after entry
    open_[32:, 0] = 121.0
    p = _prices(close, open_)
    sig = pd.DataFrame({"date": [p.dates[30]], "ticker": ["S0"]})
    r = engine.run_cohorts(sig, p, H=5, cost_bps=0.0)
    co = r.cohorts.iloc[0]
    assert co["entry"] == p.dates[31] and co["exit"] == p.dates[36]
    assert co["ret_net"] == pytest.approx(121 / 110 - 1)
    # basket: S0 plus two flat names, all bought at the same open
    assert co["basket_ret"] == pytest.approx((121 / 110 - 1) / 3)
    # the daily series compounds to the cohort return x 1/H at H=1-equivalent scale
    assert r.daily["net"].sum() == pytest.approx((121 / 110 - 1) / 5)


def test_round_trip_costs_on_flat_prices():
    p = _prices(_flat())
    sig = pd.DataFrame({"date": [p.dates[20]], "ticker": ["S1"]})
    r = engine.run_cohorts(sig, p, H=3, cost_bps=10.0)
    co = r.cohorts.iloc[0]
    assert co["ret_net"] == pytest.approx((1 - 1e-3) ** 2 - 1)
    assert co["excess_net"] == pytest.approx((1 - 1e-3) ** 2 - 1)
    # costs make the bearish information measure worse too, never better
    rs = engine.run_cohorts(sig, p, H=3, cost_bps=10.0, side="short_info")
    assert rs.cohorts.iloc[0]["excess_net"] == pytest.approx((1 - 1e-3) ** 2 - 1)
    assert rs.daily["excess_net"].sum() < 0


def test_short_info_is_basket_minus_names():
    close = _flat()
    close[25:, 2] = 80.0          # S2 falls 20% after entry day's open
    p = _prices(close)
    sig = pd.DataFrame({"date": [p.dates[23]], "ticker": ["S2"]})
    r = engine.run_cohorts(sig, p, H=4, cost_bps=0.0, side="short_info")
    co = r.cohorts.iloc[0]
    assert co["ret_net"] == pytest.approx(-0.2)
    assert co["basket_ret"] == pytest.approx(-0.2 / 3)
    assert co["excess_net"] == pytest.approx(0.2 - 0.2 / 3)


def test_overlapping_cohorts_average_with_one_over_H_capital():
    rng = np.random.default_rng(0)
    T, N = 80, 4
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, (T, N)), axis=0))
    open_ = close * np.exp(rng.normal(0, 0.005, (T, N)))
    p = _prices(close, open_)
    H = 3
    sig = pd.DataFrame({"date": p.dates[10:40], "ticker": "S0"})
    r = engine.run_cohorts(sig, p, H=H, cost_bps=0.0)
    # every cohort's own value path, summed with weight 1/H, equals the daily series
    total = r.daily["net"]
    manual = np.zeros(T)
    for t in range(10, 40):
        e, x = t + 1, t + 1 + H
        v = np.concatenate([[1.0], close[e:x, 0] / open_[e, 0], [open_[x, 0] / open_[e, 0]]])
        manual[e:x + 1] += (v[1:] / v[:-1] - 1) / H
    np.testing.assert_allclose(total.to_numpy(), manual, atol=1e-12)
    # in steady state the sleeve is fully invested: H cohorts overnight and H intraday
    assert r.daily["n_cohorts_active"].iloc[20] == H + 1


def test_no_look_ahead_signal_ignores_prices_after_exit_and_illiquid_names():
    rng = np.random.default_rng(1)
    T, N = 70, 3
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, (T, N)), axis=0))
    p1 = _prices(close)
    close2 = close.copy()
    close2[50:] *= 3.0                       # change only prices after the exit
    p2 = _prices(close2)
    sig = pd.DataFrame({"date": [p1.dates[30]], "ticker": ["S1"]})
    a = engine.run_cohorts(sig, p1, H=5).cohorts
    b = engine.run_cohorts(sig, p2, H=5).cohorts
    pd.testing.assert_frame_equal(a, b)
    # an illiquid name is not traded
    p3 = _prices(close, vol=1e5)
    assert engine.run_cohorts(sig, p3, H=5).cohorts.empty


def test_cohort_without_observable_exit_is_skipped():
    p = _prices(_flat(T=30))
    sig = pd.DataFrame({"date": [p.dates[26]], "ticker": ["S0"]})
    assert engine.run_cohorts(sig, p, H=5).cohorts.empty


def test_basket_legs_compound_to_close_to_close():
    rng = np.random.default_rng(2)
    T, N = 50, 5
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, (T, N)), axis=0))
    open_ = close * np.exp(rng.normal(0, 0.01, (T, N)))
    p = _prices(close, open_)
    legs = engine.basket_legs(p)
    cc = (close[1:] / close[:-1]).mean(axis=1) - 1
    np.testing.assert_allclose(((1 + legs["on"]) * (1 + legs["intraday"]) - 1).to_numpy()[15:],
                               cc[14:], atol=1e-12)


def test_exposure_timing_and_costs():
    rng = np.random.default_rng(3)
    T, N = 40, 4
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, (T, N)), axis=0))
    open_ = close * np.exp(rng.normal(0, 0.01, (T, N)))
    p = _prices(close, open_)
    legs = engine.basket_legs(p)
    w = pd.Series(0.0, index=p.dates)
    w.iloc[20:] = 1.0                          # decided at the close of day 20
    out = engine.run_exposure(w, legs, cost_bps=10.0)
    assert out["w_intraday"].iloc[20] == 0 and out["w_intraday"].iloc[21] == 1
    assert out["w_overnight"].iloc[21] == 0 and out["w_overnight"].iloc[22] == 1
    assert out["net"].iloc[21] == pytest.approx(legs["intraday"].iloc[21] - 1e-3)
    assert out["net"].iloc[25] == pytest.approx(legs["cc"].iloc[25])
    assert out["net"].iloc[:21].abs().max() == 0.0


def test_holm_step_down_with_cap():
    from fearsignal.trading import gates
    r = gates.holm({"a": 0.01, "b": 0.04, "c": 0.20})
    # thresholds: min(.05, .10/3)=.0333, min(.05, .10/2)=.05, min(.05, .10/1)=.05
    assert r["a"]["pass"] and r["a"]["threshold"] == pytest.approx(0.1 / 3)
    assert r["b"]["pass"] and r["b"]["threshold"] == pytest.approx(0.05)
    assert not r["c"]["pass"]
    # step-down stops at the first failure
    r2 = gates.holm({"a": 0.04, "b": 0.045, "c": 0.049})
    assert not any(v["pass"] for v in r2.values())   # 0.04 > 0.10/3, so nothing after it passes


def test_one_se_rule_prefers_earlier_declared_variant():
    from fearsignal.trading import gates
    stats = [{"mean_daily": 0.0010, "se_daily": 0.0004}, {"mean_daily": 0.0013, "se_daily": 0.0005},
             {"mean_daily": 0.0002, "se_daily": 0.0004}]
    assert gates.one_se_choice(stats) == 0          # 0.0010 >= 0.0013 - 0.0005
    stats[0]["mean_daily"] = 0.0005
    assert gates.one_se_choice(stats) == 1


def test_last_signal_date_keeps_dev_cohorts_inside_dev():
    from fearsignal.trading import gates
    d = pd.bdate_range("2025-12-01", "2026-01-30")
    s = gates.last_signal_date(d, H=5, delay=1, end="2025-12-31")
    i = d.get_loc(s)
    assert d[i + 1 + 5] <= pd.Timestamp("2025-12-31") < d[i + 1 + 5 + 1]


def test_black_scholes_parity_iv_round_trip_and_delta_strike():
    from fearsignal.trading import options as op
    S, K, T, v = 100.0, 105.0, 30 / 365, 0.55
    c, p = op.price(S, K, T, v, "call"), op.price(S, K, T, v, "put")
    assert c - p == pytest.approx(S - K * np.exp(-op.RATE * T), abs=1e-10)       # put-call parity
    assert op.implied_vol(c, S, K, T, "call") == pytest.approx(v, abs=1e-6)
    k30 = op.strike_for_delta(S, T, v, "put", -0.30)
    assert op.delta(S, k30, T, v, "put") == pytest.approx(-0.30, abs=1e-6)
    assert np.isnan(op.implied_vol(0.0, S, K, T, "call"))
