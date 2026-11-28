import json

import numpy as np
import pandas as pd
import pytest

from fearsignal.config import HoldoutLocked
from fearsignal.trading import calendar, gates, prospective as P, protocol

HAVE_INPUTS = (protocol.OUT / "holdout" / "holdout_results.json").exists() and \
    (protocol.RAW / "trade_ohlcv.parquet").exists()


def _dates(n_after: int) -> pd.DatetimeIndex:
    before = pd.bdate_range("2026-09-01", P.LAST_KNOWN)
    before = before[~before.isin(calendar.NYSE_HOLIDAYS)]
    return before.append(P.exchange_sessions(P.LAST_KNOWN, n_after))


def test_window_is_the_first_400_exchange_sessions():
    d = _dates(410)
    a, b, sess = P.window(d)
    assert a == pd.Timestamp("2026-11-30") and len(sess) == 400 and b == sess[-1]
    assert not sess.isin(calendar.NYSE_HOLIDAYS).any() and (sess.dayofweek < 5).all()
    assert pd.Timestamp("2026-12-25") not in sess and pd.Timestamp("2027-11-25") not in sess


def test_window_refusals():
    with pytest.raises(HoldoutLocked, match="only"):
        P.window(_dates(300))
    d = _dates(410)
    with pytest.raises(HoldoutLocked, match="disagree"):          # a vendor bar on a holiday
        P.window(d.append(pd.DatetimeIndex(["2026-12-25"])).sort_values())
    with pytest.raises(HoldoutLocked, match="disagree"):          # a missing session
        P.window(d.delete(d.get_loc(pd.Timestamp("2027-03-10"))))
    counts = pd.Series(130, index=d)
    counts[pd.Timestamp("2027-05-05")] = 40
    with pytest.raises(HoldoutLocked, match="too few names"):
        P.window(d, counts)


def _trades(means, sd=0.02, seed=0):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2026-12-01", periods=len(means) * 7)
    rows = [{"signal_date": days[i * 7], "ticker": f"T{i}", "excess_net": m + sd * rng.standard_normal()}
            for i, m in enumerate(means)]
    return pd.DataFrame(rows), days


def test_guard_and_rolling_demotion():
    tr, days = _trades([0.02] * 60)
    m = P.monitor(tr, days, dev_mean=0.023, tier=1)
    assert m["suspended_at"] is None and m["demotions_at"] == [] and m["tier_effective"] == 1
    tr, days = _trades([-0.03] * 60)
    m = P.monitor(tr, days, dev_mean=0.023, tier=1)
    assert m["suspended_at"] == 25 and m["tier_effective"] == 0
    tr, days = _trades([0.0008] * 80, sd=0.002)                   # within 2 s.e. of dev: nothing fires
    m = P.monitor(tr, days, dev_mean=0.001, tier=2)
    assert m["suspended_at"] is None and m["demotions_at"] == [] and m["tier_effective"] == 2
    tr, days = _trades([-0.001] * 80, sd=0.002)                   # guard disarmed by a very low dev mean
    m = P.monitor(tr, days, dev_mean=-1.0, tier=2)
    assert m["demotions_at"] == [40, 80] and m["tier_effective"] == 0
    assert P.monitor(tr.iloc[:79], days, dev_mean=-1.0, tier=2)["demotions_at"] == [40]
    # rolling, not blocks: 40 good events then a bad run fires before the 80th event
    tr, days = _trades([0.01] * 40 + [-0.03] * 35, sd=0.002)
    m = P.monitor(tr, days, dev_mean=-1.0, tier=1)
    assert m["demotions_at"] and 40 < m["demotions_at"][0] < 76 and m["tier_effective"] == 0


def test_decide_follows_the_declaration():
    r2 = {"claimed": True, "switch_to_ewma": False}
    good = {"claimed": True, "per_event": {"mean": 0.02}, "gate_series": {"mean_daily": 0.001}}
    assert P.decide(good, r2, 1, 0.01, False)["ND_tier_after"] == 2
    assert P.decide(good, r2, 1, -0.01, False)["ND_tier_after"] == 0          # IR_post <= 0 -> Tier 0
    d = P.decide(good, r2, 0, 0.01, True)
    assert d["ND_tier_after"] == 2 and "review" in d["why"]
    bad = {"claimed": False, "per_event": {"mean": -0.01}, "gate_series": {"mean_daily": 0.0002}}
    assert P.decide(bad, r2, 1, 0.01, False)["ND_tier_after"] == 0
    mid = {"claimed": False, "per_event": {"mean": 0.01}, "gate_series": {"mean_daily": 0.0002}}
    assert P.decide(mid, r2, 1, 0.01, False)["ND_tier_after"] == 1
    assert P.decide(mid, r2, 0, 0.01, True)["ND_tier_after"] == 0            # a suspension is not undone
    assert P.decide(mid, {"claimed": False, "switch_to_ewma": True}, 1, 0.01, False)["risk_yardstick"] == "ewma"


def test_apply_monitor_only_lowers():
    st = {"ND": {"tier": 1, "why": "x"}, "SR": {"tier": 0}}
    assert P.apply_monitor(st, {"tier_effective": 2})["ND"]["tier"] == 1
    out = P.apply_monitor(st, {"tier_effective": 0})
    assert out["ND"]["tier"] == 0 and "lowered" in out["ND"]["why"] and st["ND"]["tier"] == 1


def test_firings_persist_and_stand(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "MONITOR_LOG", tmp_path / "firings.jsonl")
    bad, days = _trades([-0.03] * 30)
    good, _ = _trades([0.03] * 30)

    class Prices:
        dates = days
    monkeypatch.setattr(P, "closed_nd1_events", lambda rf, prices, as_of=None: bad)
    m1 = P.nd_monitor(None, Prices, 1)
    assert m1["tier_effective"] == 0 and (tmp_path / "firings.jsonl").exists()
    monkeypatch.setattr(P, "closed_nd1_events", lambda rf, prices, as_of=None: good)   # a vendor revision
    m2 = P.nd_monitor(None, Prices, 1)
    assert m2["suspended_at"] is None and m2["tier_effective"] == 0                    # the firing stands


def test_preflight_refusals(monkeypatch):
    env = P.environment()
    decl = {"src_tree": "aaa", "committed_inputs_sha256": {f: P._sha(f) for f in P.LOCKED_FILES},
            "environment": env}
    monkeypatch.setattr(P, "prior_reads", lambda: [])
    with pytest.raises(HoldoutLocked, match="tree"):
        P.preflight(decl, {"src_tree": "bbb"}, None, None)
    assert P.preflight(decl, {"src_tree": "aaa"}, None, None)["prior_reads"] == []
    bad = {**decl, "committed_inputs_sha256": {**decl["committed_inputs_sha256"], P.LOCKED_FILES[0]: "0" * 64}}
    with pytest.raises(HoldoutLocked, match="committed inputs"):
        P.preflight(bad, {"src_tree": "aaa"}, None, None)
    monkeypatch.setattr(P, "prior_reads", lambda: ["outputs/trading/prospective/RUNLOG.md"])
    with pytest.raises(HoldoutLocked, match="earlier prospective read"):
        P.preflight(decl, {"src_tree": "aaa"}, None, None)
    assert P.preflight(decl, {"src_tree": "aaa"}, "logged reason", None)
    other = {**decl, "environment": {**env, "packages": {**env["packages"], "pandas": "0.0"}}}
    monkeypatch.setattr(P, "prior_reads", lambda: [])
    with pytest.raises(HoldoutLocked, match="environment"):
        P.preflight(other, {"src_tree": "aaa"}, None, None)
    assert P.preflight(other, {"src_tree": "aaa"}, None, "rebuilt venv, logged")


def test_prior_reads_sees_runlog_and_marker(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "RUNLOG", P.ROOT / "outputs" / "trading" / "prospective" / "RUNLOG.md")
    monkeypatch.setattr(P, "MARKER", tmp_path / "marker.jsonl")
    base = P.prior_reads()
    (tmp_path / "marker.jsonl").write_text("{}\n")
    assert len(P.prior_reads()) == len(base) + 1


def test_calendar_coverage(tmp_path):
    f = tmp_path / "cal.json"
    months = [str(m.start_time.date()) for m in pd.period_range("2026-08", "2028-07", freq="M")]
    f.write_text(json.dumps({"windows": months, "rows": []}))
    assert P.calendar_covers(f, pd.Timestamp("2026-11-30"), pd.Timestamp("2028-07-03"))[0]
    f.write_text(json.dumps({"windows": [m for m in months if m != "2027-06-01"], "rows": []}))
    ok, why = P.calendar_covers(f, pd.Timestamp("2026-11-30"), pd.Timestamp("2028-07-03"))
    assert not ok and "2027-06" in why
    assert not P.calendar_covers(tmp_path / "missing.json", pd.Timestamp("2026-11-30"), pd.Timestamp("2028-07-03"))[0]


@pytest.mark.skipif(not HAVE_INPUTS, reason="needs the locked inputs")
def test_p21_reproduces_the_v1_holdout_read():
    """P2-1 is the v1 gate statistic: on the 2026 window it must equal the recorded holdout read."""
    from fearsignal.trading import backtest
    rf, prices = backtest.load_inputs()
    r = P.p21(rf, prices, *gates.HOLDOUT)
    h = json.loads((protocol.OUT / "holdout" / "holdout_results.json").read_text())["ND.ND1"]["stats"]
    assert r["gate_series"]["t"] == pytest.approx(h["t"], rel=1e-9)
    assert r["gate_series"]["mean_daily"] == pytest.approx(h["mean_daily"], rel=1e-9)
    assert r["per_event"]["n_events"] == h["n_events"]


@pytest.mark.skipif(not HAVE_INPUTS, reason="needs the locked inputs")
def test_p22_path_reproduces_the_2026_qlike_check():
    from fearsignal.trading import backtest, sizing
    rf, _ = backtest.load_inputs()
    cal = json.loads((protocol.OUT / "sizing_calibration.json").read_text())
    q = sizing.qlike_oos(rf, sizing.load_refits(), cal["kappa"], cal["kappa_ewma"], *gates.HOLDOUT, hs=(20,))[20]
    want = json.loads((protocol.OUT / "sizing_qlike_2026.json").read_text())
    w = want["oos_2026"]["20"]
    assert q["diff_mean"] == pytest.approx(w["diff_mean"], rel=1e-9) and q["n_days"] == w["n_days"]


@pytest.mark.skipif(not HAVE_INPUTS, reason="needs the locked inputs")
def test_ir_post_all_refuses_overlapping_windows():
    from fearsignal.trading import backtest
    rf, prices = backtest.load_inputs()
    with pytest.raises(RuntimeError, match="two windows"):
        P.ir_post_all(rf, prices, pd.Timestamp("2026-06-01"), gates.HOLDOUT[1])
