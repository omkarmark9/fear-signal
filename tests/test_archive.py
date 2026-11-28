import numpy as np
import pandas as pd
import pytest

from fearsignal import archive, universe


@pytest.fixture
def arch(tmp_path, monkeypatch):
    monkeypatch.setattr(archive, "ARCHIVE", tmp_path / "archive")
    monkeypatch.setattr(universe, "tickers", lambda: ["AAA", "BBB", "CCC"])
    return tmp_path / "archive"


def _pull(end, tickers=("AAA", "BBB", "CCC"), start="2026-10-01", seed=0, scale=None):
    days = pd.bdate_range(start, end)
    rows = []
    for t in tickers:
        rng = np.random.default_rng(seed + {"AAA": 0, "BBB": 1, "CCC": 2}[t])
        px = 100 * np.exp(np.cumsum(0.01 * rng.standard_normal(len(days))))
        px = px * (scale or {}).get(t, 1.0)
        rows.append(pd.DataFrame({"date": days, "ticker": t, "adj_close": px, "close": px, "low": px * 0.99}))
    return pd.concat(rows, ignore_index=True)


def test_a_purged_ticker_is_restored_from_the_archive(arch):
    archive.update(_pull("2027-01-29"), "live")
    later = _pull("2027-03-31", tickers=("AAA", "CCC"))          # BBB is no longer served
    rep = archive.update(later, "live")
    assert rep["absent"] == ["BBB"] and (arch / "live" / "BBB.parquet").exists()
    merged, r = archive.merge(later, "live")
    assert r["from_archive"] == {"BBB": "absent from the pull"}
    b = merged[merged.ticker == "BBB"]
    assert pd.Timestamp(b.date.max()) == pd.Timestamp("2027-01-29") and len(b) > 60
    assert set(merged.ticker) == {"AAA", "BBB", "CCC"}


def test_a_truncated_or_replaced_history_keeps_the_archive(arch):
    archive.update(_pull("2027-01-29"), "live")
    short = _pull("2027-01-29")
    short = short[~((short.ticker == "AAA") & (short.date > "2026-12-15"))]      # truncated
    other = _pull("2027-01-29", seed=99)                                          # another company's history
    mixed = pd.concat([short[short.ticker != "CCC"], other[other.ticker == "CCC"]], ignore_index=True)
    rep = archive.update(mixed, "live")
    assert sorted(rep["kept"]) == ["AAA", "CCC"]
    merged, r = archive.merge(mixed, "live")
    assert set(r["from_archive"]) == {"AAA", "CCC"} and "fewer bars" in r["from_archive"]["AAA"]
    assert "disagree" not in r["from_archive"]["AAA"] and "returns agree" in r["from_archive"]["CCC"]


def test_a_rescaled_but_consistent_pull_replaces_the_archive(arch):
    archive.update(_pull("2027-01-29"), "live")
    later = _pull("2027-02-26", scale={"AAA": 0.5})              # a 2:1 split rescales the whole history
    rep = archive.update(later, "live")
    assert rep["kept"] == [] and rep["absent"] == []
    merged, r = archive.merge(later, "live")
    assert r["from_archive"] == {}
    pd.testing.assert_frame_equal(merged, later)


def test_merge_without_an_archive_is_the_identity(arch):
    p = _pull("2027-01-29")
    out, r = archive.merge(p, "trade")
    assert out is p and r == {"from_archive": {}}
