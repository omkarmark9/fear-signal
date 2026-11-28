import json

import numpy as np
import pandas as pd
import pytest

from fearsignal.config import HoldoutLocked
from fearsignal.trading import calendar, engine, gates, prospective as P, protocol

HAVE_INPUTS = (protocol.OUT / "holdout" / "holdout_results.json").exists() and \
    (protocol.RAW / "trade_ohlcv.parquet").exists()


def _dates(n_after: int) -> pd.DatetimeIndex:
    before = pd.bdate_range("2026-09-01", P.LAST_KNOWN)
    before = before[~before.isin(calendar.NYSE_HOLIDAYS)]
    return before.append(P.exchange_sessions(P.LAST_KNOWN, n_after))


def test_window_is_the_first_400_exchange_sessions():
    d = _dates(410)
    a, b, sess = P.window(d)
    assert a == pd.Timestamp("2026-11-30") and len(sess) == 400 and b == sess[-1] == pd.Timestamp("2028-07-03")
    assert not sess.isin(calendar.NYSE_HOLIDAYS).any() and (sess.day_of_week < 5).all()
    assert pd.Timestamp("2026-12-25") not in sess and pd.Timestamp("2027-11-25") not in sess


def test_window_refusals():
    with pytest.raises(HoldoutLocked, match="only"):
        P.window(_dates(300))
    d = _dates(410)
    with pytest.raises(HoldoutLocked, match="disagree"):          # a vendor bar on a holiday
        P.window(d.append(pd.DatetimeIndex(["2026-12-25"])).sort_values())
    with pytest.raises(HoldoutLocked, match="disagree.*--closed"):  # a missing session
        P.window(d.delete(d.get_loc(pd.Timestamp("2027-03-10"))))
    counts = pd.Series(130, index=d)
    counts[pd.Timestamp("2027-05-05")] = 40
    with pytest.raises(HoldoutLocked, match="too few names"):
        P.window(d, counts)


def test_window_floor_is_the_universe_not_the_sample_median():
    """A shortfall spread evenly over the sample moved the old relative floor with it."""
    d = _dates(410)
    assert P.REF_NAMES == 134
    with pytest.raises(HoldoutLocked, match="too few names"):
        P.window(d, pd.Series(110, index=d))
    assert P.window(d, pd.Series(121, index=d))[2].size == 400
    counts = pd.Series(130, index=d)
    opens = pd.Series(130, index=d)
    opens[pd.Timestamp("2027-08-02")] = 60                          # opens checked as well as closes
    with pytest.raises(HoldoutLocked, match="opens"):
        P.window(d, {"closes": counts, "opens": opens})


def test_window_absorbs_a_declared_closure():
    d = _dates(410)
    shut = pd.Timestamp("2027-06-08")
    d2 = d.delete(d.get_loc(shut))
    with pytest.raises(HoldoutLocked, match="missing \\['2027-06-08'\\]"):
        P.window(d2)
    a, b, sess = P.window(d2, closed=["2027-06-08"])
    assert len(sess) == 400 and shut not in sess and b == pd.Timestamp("2028-07-05")
    assert sess[-2] == pd.Timestamp("2028-07-03")
    with pytest.raises(HoldoutLocked, match="exchange was open"):    # a closure cannot delete a real session
        P.window(d, closed=["2027-06-08"])
    with pytest.raises(HoldoutLocked, match="not scheduled"):        # nor be a day the table already closes
        P.window(d2, closed=["2027-12-25"])
    counts = pd.Series(130, index=d)                                  # an all-empty date counts as absent
    counts[shut] = 0
    assert P.window(d, counts, closed=["2027-06-08"])[1] == pd.Timestamp("2028-07-05")


def test_read_refuses_a_closure_without_a_reason():
    with pytest.raises(HoldoutLocked, match="calendar-reason"):
        P.read(closed=["2027-06-08"])


def _trades(means, sd=0.02, seed=0, start="2026-12-01"):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, periods=len(means) * 7)
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


def test_monitors_use_the_per_trade_se():
    e = np.array([0.03, -0.01, 0.02, 0.0, 0.05, -0.02])
    s = P.trade_stats(e)
    assert s["se"] == pytest.approx(e.std(ddof=1) / np.sqrt(len(e)))
    # 40 events in 4 tight clusters: CR1 by episode is tiny, the per-trade s.e. is not
    days = pd.bdate_range("2026-12-01", periods=400)
    vals = np.r_[np.full(10, -0.004), np.full(10, 0.002), np.full(10, -0.003), np.full(10, 0.001)]
    vals = vals + np.tile(np.linspace(-0.05, 0.05, 10), 4)
    tr = pd.DataFrame({"signal_date": np.repeat(days[[0, 50, 100, 150]], 10),
                       "ticker": [f"T{i}" for i in range(40)], "excess_net": vals})
    m = P.monitor(tr, days, dev_mean=-1.0, tier=1)
    assert m["demotions_at"] == []                                 # mean -0.001 is within one per-trade s.e.
    assert P.episode_stats(tr, days)["se"] < P.trade_stats(vals)["se"] / 2


def test_decide_follows_the_declaration():
    r2 = {"claimed": True, "switch_to_ewma": False}
    good = {"claimed": True, "per_event": {"mean": 0.02}, "gate_series": {"mean_daily": 0.001}}
    assert P.decide(good, r2, 1, 0.01, False)["ND_tier_after"] == 2
    assert P.decide(good, r2, 1, -0.01, False)["ND_tier_after"] == 0          # IR_post <= 0 -> Tier 0
    d = P.decide(good, r2, 0, 0.01, True)                                     # claimed, but pending review
    assert d["ND_tier_after"] == 0 and d["ND_tier_after_review"] == 2 and "review" in d["why"]
    assert P.decide(good, r2, 0, 0.01, False)["ND_tier_after"] == 2           # one demotion: the claim promotes
    bad = {"claimed": False, "per_event": {"mean": -0.01}, "gate_series": {"mean_daily": 0.0002}}
    assert P.decide(bad, r2, 1, 0.01, False)["ND_tier_after"] == 0
    mid = {"claimed": False, "per_event": {"mean": 0.01}, "gate_series": {"mean_daily": 0.0002}}
    assert P.decide(mid, r2, 1, 0.01, False)["ND_tier_after"] == 1
    assert P.decide(mid, r2, 0, 0.01, True)["ND_tier_after"] == 0            # a suspension is not undone
    assert P.decide(mid, {"claimed": False, "switch_to_ewma": True}, 1, 0.01, False)["risk_yardstick"] == "ewma"
    none = {"claimed": False, "per_event": {"mean": None}, "gate_series": {"mean_daily": 0.0}}
    assert P.decide(none, r2, 1, 0.01, False)["ND_tier_after"] == 0          # no events at all


def test_apply_monitor_only_lowers():
    st = {"ND": {"tier": 1, "why": "x"}, "SR": {"tier": 0}}
    assert P.apply_monitor(st, {"tier_effective": 2})["ND"]["tier"] == 1
    out = P.apply_monitor(st, {"tier_effective": 0})
    assert out["ND"]["tier"] == 0 and "lowered" in out["ND"]["why"] and st["ND"]["tier"] == 1


class _Prices:
    def __init__(self, dates):
        self.dates = dates


def _patch_events(monkeypatch, frame, seen=None):
    def fake(rf, prices, as_of=None, after=P.LAST_KNOWN):
        if seen is not None:
            seen.append(pd.Timestamp(after))
        return frame() if callable(frame) else frame
    monkeypatch.setattr(P, "closed_nd1_events", fake)


def test_firings_persist_and_stand(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "MONITOR_LOG", tmp_path / "firings.jsonl")
    bad, days = _trades([-0.03] * 30)
    good, _ = _trades([0.03] * 30)
    _patch_events(monkeypatch, bad)
    m1 = P.nd_monitor(None, _Prices(days), 1)
    assert m1["tier_effective"] == 0 and (tmp_path / "firings.jsonl").exists()
    f = json.loads((tmp_path / "firings.jsonl").read_text().splitlines()[0])
    assert f["rule"] == "guard" and f["event"] == {"signal_date": str(bad.signal_date[24].date()), "ticker": "T24"}
    _patch_events(monkeypatch, good)                                                    # a vendor revision
    m2 = P.nd_monitor(None, _Prices(days), 1)
    assert m2["suspended_at"] is None and m2["suspended"] and m2["tier_effective"] == 0  # the firing stands
    assert m2["pending_review"]


def test_the_monitor_log_survives_checkouts():
    assert P.MONITOR_LOG.parent == protocol.RAW == P.MARKER.parent      # git-ignored, like the marker


def _demotion_case():
    tr, days = _trades([0.0] * 30 + [-0.02] * 30, sd=0.002)
    return tr, days


def _dev_mean(monkeypatch, tmp_path, value):
    f = tmp_path / "selected.json"
    f.write_text(json.dumps({"families": {"ND": {"chosen_stats": {"mean_event_excess": value}}}}))
    monkeypatch.setattr(protocol, "SELECTED", f)


def _exact(values, extra_at=None):
    days = pd.bdate_range("2026-12-01", periods=len(values) * 3 + 10)
    rows = [{"signal_date": days[3 * i + 1], "ticker": f"T{i}", "excess_net": v} for i, v in enumerate(values)]
    if extra_at is not None:                       # an earlier event a backfilled bar made tradable
        rows.insert(extra_at, {"signal_date": days[3 * extra_at], "ticker": "EXTRA", "excess_net": 0.0})
    return pd.DataFrame(rows), days


def test_a_shifted_index_is_still_one_demotion(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "MONITOR_LOG", tmp_path / "firings.jsonl")
    _dev_mean(monkeypatch, tmp_path, -1.0)                       # guard disarmed
    vals = [0.0] * 39 + [-0.01] * 21
    tr, days = _exact(vals)
    _patch_events(monkeypatch, tr)
    m1 = P.nd_monitor(None, _Prices(days), 2)
    assert m1["demotions_at"] == [41] and m1["tier_effective"] == 1
    tr2, _ = _exact(vals, extra_at=1)
    assert P.monitor(tr2, days, -1.0, 2)["demotions_at"] == [42]    # the same demotion, one index later
    _patch_events(monkeypatch, tr2)
    m2 = P.nd_monitor(None, _Prices(days), 2)
    logged = m2["logged_firings"][0]
    assert logged["event"] == {"signal_date": str(tr.signal_date[40].date()), "ticker": "T40"}
    assert P._position(logged, P._event_index(tr2)) == 42
    assert m2["new_firings"] == [] and m2["demotions_counted"] == [42] and m2["tier_effective"] == 1
    assert len((tmp_path / "firings.jsonl").read_text().splitlines()) == 1


def test_close_firings_count_once_and_spaced_ones_twice():
    tr, _ = _demotion_case()
    m = {"suspended_at": None, "demotions_at": [48]}
    near = [{"rule": "demotion", "at_event": 46, "event": None, "as_of": "2027-01-01"}]
    assert P.merge_firings(m, tr, near)["demotions"] == [46]
    far = [{"rule": "demotion", "at_event": 5, "event": None, "as_of": "2027-01-01"}]
    assert P.merge_firings(m, tr, far)["demotions"] == [5, 48]
    assert P._new_firings(m, tr, near) == [] and len(P._new_firings(m, tr, far)) == 1


def test_firings_after_as_of_or_from_another_count_are_not_merged(tmp_path, monkeypatch):
    log = tmp_path / "firings.jsonl"
    monkeypatch.setattr(P, "MONITOR_LOG", log)
    good, days = _trades([0.03] * 30)
    rows = [{"rule": "guard", "at_event": 25, "event": None, "after": "2026-11-27", "as_of": "2028-07-06"},
            {"rule": "guard", "at_event": 25, "event": None, "after": "2028-06-01", "as_of": "2028-06-20"}]
    log.write_text("".join(json.dumps(r) + "\n" for r in rows))
    _patch_events(monkeypatch, good)
    m = P.nd_monitor(None, _Prices(days), 1, as_of=pd.Timestamp("2028-07-03"), persist=False)
    assert not m["suspended"] and m["tier_effective"] == 1 and m["logged_firings"] == []
    seen = []
    late, late_days = _trades([0.03] * 30, start="2028-06-02")
    _patch_events(monkeypatch, late, seen)
    m = P.nd_monitor(None, _Prices(late_days), 2, after=pd.Timestamp("2028-06-01"), persist=False)
    assert seen == [pd.Timestamp("2028-06-01")] and m["suspended"] and m["tier_effective"] == 0


def test_live_after_a_read_counts_only_post_sample_events(tmp_path, monkeypatch):
    """The read's tier already counts every in-sample firing; live must not apply them again."""
    from fearsignal.trading import live
    log = tmp_path / "firings.jsonl"
    monkeypatch.setattr(P, "MONITOR_LOG", log)
    log.write_text(json.dumps({"rule": "demotion", "at_event": 40, "event": None, "after": "2026-11-27",
                               "as_of": "2028-01-10"}) + "\n")
    seen = []
    good, gdays = _trades([0.03] * 10, start="2028-06-02")
    _patch_events(monkeypatch, good, seen)
    status = {"ND": {"tier": 2, "why": "prospective read: P2-1 claimed",
                     "prospective_read": {"monitors_count_signals_after": "2028-06-01"}}, "SR": {"tier": 0}}
    out = live.live_monitors(status, None, _Prices(gdays))
    assert seen == [pd.Timestamp("2028-06-01")] and out["ND"]["tier"] == 2
    seen.clear()
    before = {"ND": {"tier": 1, "why": "adopted"}, "SR": {"tier": 0}}
    out = live.live_monitors(before, None, _Prices(gdays))
    assert seen == [P.LAST_KNOWN] and out["ND"]["tier"] == 0                 # before the read it does count


def test_prospective_consequence_hands_live_the_post_sample_count(tmp_path, monkeypatch):
    from fearsignal import frozen
    from fearsignal.trading import live
    monkeypatch.setattr(P, "OUT", tmp_path)
    monkeypatch.setattr(P, "DECL", tmp_path / "decl.json")
    (tmp_path / "decl.json").write_text("{}")
    monkeypatch.setattr(frozen, "committed_spec_hash", lambda path=None: "abc")
    (tmp_path / "read.json").write_text(json.dumps({
        "declaration_sha256": "abc", "window": ["2026-11-30", "2028-07-03"], "last_countable_signal": "2028-06-01",
        "decision": {"ND_tier_after": 0, "ND_tier_after_review": 2, "why": "P2-1 claimed; Tier 2 waits"}}))
    nd = live._prospective_consequence({"tier": 1, "why": "adopted"})
    assert nd["tier"] == 0 and nd["prospective_read"]["monitors_count_signals_after"] == "2028-06-01"
    assert nd["prospective_read"]["tier_after_review"] == 2


def test_empty_trade_sets_do_not_crash(monkeypatch):
    assert P.episode_stats(pd.DataFrame(), pd.bdate_range("2027-01-01", periods=5))["n_events"] == 0
    assert P.episode_stats(pd.DataFrame([]), pd.bdate_range("2027-01-01", periods=5))["mean"] is None
    m = P.monitor(pd.DataFrame([]), pd.bdate_range("2027-01-01", periods=5), 0.02, 1)
    assert m["closed_events"] == 0 and m["tier_effective"] == 1
    from fearsignal.trading import registry
    days = P.exchange_sessions(P.LAST_KNOWN, 60)
    empty = engine.Result(pd.DataFrame(index=days), pd.DataFrame([]), pd.DataFrame([]))
    monkeypatch.setattr(registry, "run", lambda *a, **k: empty)
    out = P.closed_nd1_events(None, _Prices(days))
    assert len(out) == 0 and list(out.columns) == ["signal_date", "ticker", "excess_net"]
    from fearsignal.trading import backtest
    zero = pd.Series(0.0, index=days)
    monkeypatch.setattr(backtest, "run_window", lambda *a, **k: (empty, zero))
    r1 = P.p21(None, _Prices(days), days[0], days[-1])
    assert r1["per_event"]["n_events"] == 0 and not r1["claimed"]
    assert P.decide(r1, {"claimed": False, "switch_to_ewma": False}, 1, 0.01, False)["ND_tier_after"] == 0


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


def test_environment_policy(monkeypatch):
    """Python and the four numerical packages are checked; yfinance and the freeze are recorded."""
    env = P.environment()
    assert set(env["packages"]) == {"numpy", "pandas", "scipy", "pyarrow"} and "yfinance" in env["recorded"]
    assert env["python"].count(".") == 1
    monkeypatch.setattr(P, "prior_reads", lambda: [])
    base = {"src_tree": "aaa", "committed_inputs_sha256": {}, "environment": env}
    py = {**base, "environment": {**env, "python": "2.7"}}
    with pytest.raises(HoldoutLocked, match="python"):
        P.preflight(py, {"src_tree": "aaa"}, None, None)
    yf = {**base, "environment": {**env, "recorded": {"yfinance": "0.0.1"}, "freeze_sha256": "0" * 64}}
    rec = P.preflight(yf, {"src_tree": "aaa"}, None, None)["environment"]["differs_from_declaration"]
    assert rec["checked"] == {} and "yfinance" in rec["recorded"] and rec["freeze"]


def test_prior_reads_sees_runlog_and_marker(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "RUNLOG", P.ROOT / "outputs" / "trading" / "prospective" / "RUNLOG.md")
    monkeypatch.setattr(P, "MARKER", tmp_path / "marker.jsonl")
    base = P.prior_reads()
    (tmp_path / "marker.jsonl").write_text("{}\n")
    assert len(P.prior_reads()) == len(base) + 1


def test_prior_reads_refuses_when_git_cannot_answer(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "ROOT", tmp_path)                       # not a git repository
    monkeypatch.setattr(P, "MARKER", tmp_path / "marker.jsonl")
    monkeypatch.setattr(P, "RUNLOG", tmp_path / "RUNLOG.md")
    with pytest.raises(HoldoutLocked, match="git"):
        P.prior_reads()


def test_one_read_at_a_time(tmp_path):
    lock = tmp_path / "read.lock"
    with P._exclusive(lock):
        with pytest.raises(HoldoutLocked, match="another prospective read"):
            with P._exclusive(lock):
                pass
    with P._exclusive(lock):                                       # released afterwards
        pass


def test_a_corrupt_firings_log_refuses(tmp_path):
    log = tmp_path / "firings.jsonl"
    log.write_text(json.dumps({"rule": "guard", "at_event": 25, "as_of": "2027-02-01"}) + "\n{\"rule\": \"dem")
    with pytest.raises(HoldoutLocked, match="line 2"):
        P._persisted(log)


def _calendar(months, names=134, fetched="2028-07-10T00:00:00Z"):
    rows = []
    for m in months:
        p = pd.Period(m, "M")
        first = p.asfreq("Q").asfreq("M", "s") == p          # every name reports in a quarter's first month
        for i in range(names if first else 6):
            rows.append({"symbol": f"S{i}", "report": {"date": str(p.start_time.date() + pd.Timedelta(days=9)),
                                                       "timing": "pm", "verified": True}})
    return {"fetched_utc": fetched, "windows": [str(pd.Period(m, "M").start_time.date()) for m in months], "rows": rows}


def test_calendar_coverage(tmp_path):
    f = tmp_path / "cal.json"
    a, b = pd.Timestamp("2026-11-30"), pd.Timestamp("2028-07-03")
    months = [str(m) for m in pd.period_range("2026-08", "2028-07", freq="M")]
    f.write_text(json.dumps(_calendar(months)))
    assert P.calendar_covers(f, a, b) == (True, "covered")
    f.write_text(json.dumps(_calendar([m for m in months if m != "2027-06"])))
    ok, why = P.calendar_covers(f, a, b)
    assert not ok and "2027-06" in why
    assert not P.calendar_covers(tmp_path / "missing.json", a, b)[0]
    doc = _calendar(months)
    doc["rows"] = []                                               # windows claimed, no data: not covered
    f.write_text(json.dumps(doc))
    ok, why = P.calendar_covers(f, a, b)
    assert not ok and "report rows" in why
    f.write_text(json.dumps(_calendar(months, names=100)))         # most names missing in every quarter
    ok, why = P.calendar_covers(f, a, b)
    assert not ok and "of the universe reports" in why
    f.write_text(json.dumps(_calendar(months, fetched="2028-06-30T00:00:00Z")))
    ok, why = P.calendar_covers(f, a, b)
    assert not ok and "not after the sample" in why
    for broken in ('{"windows": [', '["2027-01-01"]', '{"windows": [{"m": 1}]}', '{"windows": ["not a date"]}'):
        f.write_text(broken)
        ok, why = P.calendar_covers(f, a, b)
        assert not ok and "unreadable" in why


def test_cli_passes_every_declared_flag(monkeypatch):
    from fearsignal import cli
    got = {}
    monkeypatch.setattr(P, "read", lambda *a: got.setdefault("args", a) and {})
    cli.main(["trading-prospective", "--rerun-reason", "r", "--env-reason", "e",
              "--closed", "2027-06-08", "--closed", "2027-06-09", "--calendar-reason", "c"])
    assert got["args"] == ("r", "e", ["2027-06-08", "2027-06-09"], "c")


def test_declaration_record_refuses_an_uncommitted_input(monkeypatch):
    head = P._git("rev-parse", "HEAD").strip()
    monkeypatch.setattr(protocol, "code_state", lambda: {"head": head,
                                                         "src_tree": P._git("rev-parse", "HEAD:src/fearsignal").strip()})
    real = P._sha
    monkeypatch.setattr(P, "_sha", lambda f: "0" * 64 if f == P.LOCKED_FILES[0] else real(f))
    with pytest.raises(HoldoutLocked, match="differs from its content"):
        P.declaration_record(head)


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
def test_p22_itself_on_the_frozen_v3_branch():
    """The read's own call: no refits, frozen v3's A and s, p = Phi(t), the declared switch rule."""
    from scipy import stats
    from fearsignal import frozen
    from fearsignal.trading import backtest, sizing
    rf, _ = backtest.load_inputs()
    cal = json.loads((protocol.OUT / "sizing_calibration.json").read_text())
    spec3 = json.loads(frozen.FROZEN_V3_PATH.read_text())["stock_model"]
    r = P.p22(rf, spec3["A"], spec3["s"], cal["kappa"], cal["kappa_ewma"], *gates.HOLDOUT)
    q = sizing.qlike_oos(rf, [], cal["kappa"], cal["kappa_ewma"], *gates.HOLDOUT, hs=(20,),
                         A_live=spec3["A"], s_live=spec3["s"])[20]
    pit = sizing.qlike_oos(rf, sizing.load_refits(), cal["kappa"], cal["kappa_ewma"], *gates.HOLDOUT, hs=(20,))[20]
    assert r["diff_mean"] == q["diff_mean"] and r["n_days"] == q["n_days"] == pit["n_days"]
    assert r["diff_mean"] != pit["diff_mean"]                     # frozen v3, not the PIT refits
    assert r["p_one_sided"] == pytest.approx(stats.norm.cdf(r["t"])) and np.isfinite(r["t"])
    assert r["switch_to_ewma"] == (r["diff_mean"] > r["diff_se"])


@pytest.mark.skipif(not HAVE_INPUTS, reason="needs the locked inputs")
def test_ir_post_all_refuses_overlapping_windows():
    from fearsignal.trading import backtest
    rf, prices = backtest.load_inputs()
    with pytest.raises(RuntimeError, match="two windows"):
        P.ir_post_all(rf, prices, pd.Timestamp("2026-06-01"), gates.HOLDOUT[1])


@pytest.mark.skipif(not HAVE_INPUTS, reason="needs the locked inputs")
def test_ir_post_all_with_no_prospective_events(monkeypatch):
    from fearsignal.trading import backtest
    rf, prices = backtest.load_inputs()
    real = backtest.run_window
    empty = engine.Result(pd.DataFrame(), pd.DataFrame([]), pd.DataFrame([]))

    def fake(v, rf_, prices_, a, b, *k, **kw):
        return (empty, pd.Series(dtype=float)) if rf_ is None else real(v, rf_, prices_, a, b, *k, **kw)
    monkeypatch.setattr(backtest, "run_window", fake)
    out = P.ir_post_all(None, None, pd.Timestamp("2026-11-30"), pd.Timestamp("2028-07-03"))
    assert out["n_events"]["prospective"] == 0 and out["n_events"]["dev"] > 0 and np.isfinite(out["IR_post"])
