import numpy as np
import pandas as pd
import pytest

from fearsignal import config, data, universe


def _long(dates, ticker="NVDA", close=100.0, factor=1.0):
    n = len(dates)
    return pd.DataFrame({
        "date": pd.DatetimeIndex(dates),
        "ticker": ticker,
        "adj_close": np.full(n, close * factor),
        "close": np.full(n, close),
        "low": np.full(n, close * 0.99),
    })


def test_era_rule_drops_pre_2023_rows():
    long = _long(pd.bdate_range("2022-12-20", "2023-01-10"))
    panel = data.to_panel(long)
    assert panel.dates.min() >= config.ERA_START


def test_assert_era_and_dev_raise():
    with pytest.raises(config.EraViolation):
        config.assert_era(pd.DatetimeIndex(["2022-12-30"]))
    with pytest.raises(config.HoldoutLocked):
        config.assert_dev(pd.DatetimeIndex(["2026-01-02"]))


def test_adjusted_low_uses_adjustment_factor():
    long = _long(pd.bdate_range("2023-01-03", "2023-01-06"), factor=0.9)
    panel = data.to_panel(long)
    np.testing.assert_allclose(panel.adj_low, 100 * 0.99 * 0.9)


def test_bad_bars_are_masked_on_all_fields():
    long = _long(pd.bdate_range("2023-01-03", "2023-01-06"))
    long.loc[1, "low"] = 0.0
    panel = data.to_panel(long)
    assert np.isnan(panel.close[1, 0]) and np.isnan(panel.adj_close[1, 0])


def test_trade_start_override_applied():
    assert universe.table().loc["OKLO", "trade_start"] == "2024-05-10"
    long = _long(pd.bdate_range("2024-05-01", "2024-05-20"), ticker="OKLO")
    panel = data.to_panel(long)
    assert panel.dates.min() == pd.Timestamp("2024-05-10")


def test_universe_has_no_duplicates_and_known_size():
    t = universe.tickers()
    assert len(t) == len(set(t)) == 134


def test_align_adjustment_basis_rescales_by_overlap_constant():
    d = _long(pd.bdate_range("2025-12-01", "2025-12-31"), factor=0.98)
    h = _long(pd.bdate_range("2025-12-01", "2026-01-09"), factor=0.97)
    out = data.align_adjustment_basis(d, h)
    np.testing.assert_allclose(out["adj_close"], 100 * 0.97)
    np.testing.assert_allclose(out["close"], d["close"])


def test_align_adjustment_basis_rejects_inconsistent_pulls():
    d = _long(pd.bdate_range("2025-12-01", "2025-12-31"))
    h = _long(pd.bdate_range("2025-12-01", "2026-01-09"))
    h.loc[3, "close"] *= 1.01
    with pytest.raises(ValueError):
        data.align_adjustment_basis(d, h)


def test_holdout_file_is_unreadable_without_a_committed_frozen_spec(tmp_path, monkeypatch):
    from fearsignal import frozen
    monkeypatch.setattr(frozen, "FROZEN_PATH", tmp_path / "frozen.json")
    monkeypatch.setattr(frozen.committed_spec_hash, "__defaults__", (tmp_path / "frozen.json",))
    with pytest.raises(config.HoldoutLocked):
        data.verify("holdout")
    with pytest.raises(config.HoldoutLocked):
        data.load_holdout_panel("0" * 64)


def test_align_adjustment_basis_handles_a_split_between_pulls():
    d = _long(pd.bdate_range("2025-12-01", "2025-12-31"), close=200.0, factor=0.98)
    h = _long(pd.bdate_range("2025-12-01", "2026-01-09"), close=100.0, factor=0.97)  # 2:1 split + dividend
    out = data.align_adjustment_basis(d, h)
    np.testing.assert_allclose(out["close"], 100.0)
    np.testing.assert_allclose(out["low"], 99.0)
    np.testing.assert_allclose(out["adj_close"], 100.0 * 0.97)
    # the adjusted low and returns are continuous across the pulls
    both = pd.concat([out, h[h["date"] > "2025-12-31"]])
    panel = data.to_panel(both)
    r = np.diff(np.log(panel.adj_close[:, 0]))
    np.testing.assert_allclose(r, 0.0, atol=1e-12)


def test_align_adjustment_basis_refuses_a_ticker_missing_from_the_overlap():
    d = pd.concat([_long(pd.bdate_range("2025-11-03", "2025-12-31")),
                   _long(pd.bdate_range("2025-11-03", "2025-11-28"), ticker="AMD")])
    h = pd.concat([_long(pd.bdate_range("2025-12-01", "2026-01-09")),
                   _long(pd.bdate_range("2026-01-05", "2026-01-09"), ticker="AMD")])
    with pytest.raises(ValueError, match="AMD"):
        data.align_adjustment_basis(d, h)


def test_align_adjustment_basis_is_identity_when_nothing_changed():
    d = _long(pd.bdate_range("2025-12-01", "2025-12-31"))
    h = _long(pd.bdate_range("2025-12-01", "2026-01-09"))
    out = data.align_adjustment_basis(d, h)
    pd.testing.assert_frame_equal(out.reset_index(drop=True), d.reset_index(drop=True))


def test_align_adjustment_basis_refuses_an_all_nan_overlap():
    d = _long(pd.bdate_range("2025-11-03", "2025-12-31"))
    h = _long(pd.bdate_range("2025-12-01", "2026-01-09"))
    h.loc[pd.to_datetime(h["date"]) <= "2025-12-31", "close"] = np.nan
    with pytest.raises(ValueError, match="NVDA"):
        data.align_adjustment_basis(d, h)


def _rh_json(tmp_path, bars_by_symbol, name="b.json"):
    import json as _json
    doc = {"data": {"results": [
        {"symbol": s, "interval": "day", "bars": [
            {"begins_at": f"{d}T00:00:00Z", "close_price": f"{c:.6f}", "low_price": f"{lo:.6f}",
             **({"interpolated": True} if interp else {})}
            for d, c, lo, interp in bars]}
        for s, bars in bars_by_symbol.items()]}}
    p = tmp_path / name
    p.write_text(_json.dumps(doc))
    return p


def test_robinhood_long_parses_and_skips_interpolated(tmp_path):
    p = _rh_json(tmp_path, {"NVDA": [("2026-11-23", 100.0, 98.0, False), ("2026-11-24", 99.0, 97.0, True)]})
    df = data.robinhood_long([p])
    assert len(df) == 1 and df.loc[0, "close"] == df.loc[0, "adj_close"] == 100.0 and df.loc[0, "low"] == 98.0


def test_stitch_robinhood_scales_fills_and_refuses(tmp_path):
    past = _long(pd.bdate_range("2026-11-16", "2026-11-23"), close=100.0, factor=0.98)
    past.loc[past["date"] == pd.Timestamp("2026-11-20"), "close"] = np.nan        # broken vendor bar
    # the other vendor's basis is half (e.g. after a 2:1 split), fully adjusted
    rh = pd.DataFrame({"date": pd.bdate_range("2026-11-16", "2026-11-24"), "ticker": "NVDA",
                       "adj_close": 49.0, "close": 49.0, "low": 48.0})
    merged, rep = data.stitch_robinhood(past, rh, start_after="2026-11-23")
    m = merged.set_index("date")
    np.testing.assert_allclose(m.loc["2026-11-24", "adj_close"], 98.0)        # scaled to past basis
    np.testing.assert_allclose(m.loc["2026-11-24", "low"], 96.0)
    np.testing.assert_allclose(m.loc["2026-11-20", "close"], 98.0)            # broken bar filled
    assert rep["history_bars_filled"] == ["NVDA 2026-11-20"] and rep["new_sessions"] == ["2026-11-24"]
    bad = rh.copy()
    bad.loc[2, "adj_close"] = bad.loc[2, "close"] = 55.0                      # non-constant ratio
    with pytest.raises(ValueError, match="more than a constant"):
        data.stitch_robinhood(past, bad, start_after="2026-11-23")
    other = rh.assign(ticker="AMD")                                           # no overlap at all
    with pytest.raises(ValueError, match="AMD"):
        data.stitch_robinhood(past, pd.concat([rh, other]), start_after="2026-11-23")
