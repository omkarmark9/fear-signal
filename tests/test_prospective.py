import json

import numpy as np
import pandas as pd
import pytest

from fearsignal.config import HoldoutLocked
from fearsignal.trading import gates, prospective as P, protocol


def test_window_refuses_until_400_sessions_exist():
    d = pd.bdate_range("2026-01-02", periods=300 + 399)
    with pytest.raises(HoldoutLocked):
        P.window(d[d <= pd.Timestamp("2027-06-01")])
    full = pd.bdate_range("2026-11-02", "2028-12-29")
    a, b = P.window(full)
    after = full[full > P.LAST_KNOWN]
    assert a == pd.Timestamp("2026-11-30") and b == after[399]


def _trades(means, sd=0.02, per_day=1, seed=0):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2026-12-01", periods=len(means) * 7)
    rows = []
    for i, m in enumerate(means):
        rows.append({"signal_date": days[i * 7], "ticker": f"T{i}", "excess_net": m + sd * rng.standard_normal()})
    return pd.DataFrame(rows), days


def test_monitor_guard_and_demotion():
    tr, days = _trades([0.02] * 60)
    m = P.monitor(tr, days, dev_mean=0.023, tier=1)
    assert m["suspended_at"] is None and m["demotions_at"] == [] and m["tier_effective"] == 1
    tr, days = _trades([-0.03] * 60)
    m = P.monitor(tr, days, dev_mean=0.023, tier=1)
    assert m["suspended_at"] == 25 and m["tier_effective"] == 0
    # slightly below dev but within 2 s.e.: neither rule fires
    tr, days = _trades([0.0008] * 80, sd=0.002)
    m = P.monitor(tr, days, dev_mean=0.001, tier=2)
    assert m["suspended_at"] is None and m["demotions_at"] == [] and m["tier_effective"] == 2
    # demotion alone (guard disarmed by a very low dev mean): each block of 40 below -1 s.e.
    tr, days = _trades([-0.001] * 80, sd=0.002)
    m = P.monitor(tr, days, dev_mean=-1.0, tier=2)
    assert m["suspended_at"] is None and m["demotions_at"] == [40, 80] and m["tier_effective"] == 0
    m = P.monitor(tr.iloc[:79], days, dev_mean=-1.0, tier=2)
    assert m["demotions_at"] == [40] and m["tier_effective"] == 1


def test_decide():
    r2 = {"claimed": True, "switch_to_ewma": False}
    good = {"claimed": True, "per_event": {"mean": 0.02}, "gate_series": {"mean_daily": 0.001}}
    assert P.decide(good, r2, 1, 0.01)["ND_tier_after"] == 2
    assert P.decide(good, r2, 1, -0.01)["ND_tier_after"] == 1
    bad = {"claimed": False, "per_event": {"mean": -0.01}, "gate_series": {"mean_daily": 0.0002}}
    assert P.decide(bad, r2, 1, 0.01)["ND_tier_after"] == 0
    mid = {"claimed": False, "per_event": {"mean": 0.01}, "gate_series": {"mean_daily": 0.0002}}
    d = P.decide(mid, {"claimed": False, "switch_to_ewma": True}, 1, 0.01)
    assert d["ND_tier_after"] == 1 and d["risk_yardstick"] == "ewma"


@pytest.mark.skipif(not (protocol.OUT / "holdout" / "holdout_results.json").exists()
                    or not (protocol.RAW / "trade_ohlcv.parquet").exists(), reason="needs the locked inputs")
def test_p21_reproduces_the_v1_holdout_read():
    """P2-1 is the v1 gate statistic: on the 2026 window it must equal the recorded holdout read."""
    from fearsignal.trading import backtest
    rf, prices = backtest.load_inputs()
    r = P.p21(rf, prices, *gates.HOLDOUT)
    h = json.loads((protocol.OUT / "holdout" / "holdout_results.json").read_text())["ND.ND1"]["stats"]
    assert r["gate_series"]["t"] == pytest.approx(h["t"], rel=1e-9)
    assert r["gate_series"]["mean_daily"] == pytest.approx(h["mean_daily"], rel=1e-9)
    assert r["per_event"]["n_events"] == h["n_events"]
