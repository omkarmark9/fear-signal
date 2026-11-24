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
