"""v2 exploration runner: rule equivalences and the append-only registry."""

import json

import numpy as np
import pandas as pd

from fearsignal.trading import explore as X, families as F


def test_generalised_staircase_reproduces_rf1():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2024-01-02", periods=300)
    mp = pd.Series(rng.normal(0.6, 0.9, 300), index=idx)
    hd = pd.Series(False, index=idx)
    hd.iloc[[50, 120]] = True
    sh = pd.Series(0.02, index=idx)
    np.testing.assert_array_equal(X.staircase(mp, hd, on=1.5, cut=0.5).to_numpy(),
                                  F.rf_weights("RF1", mp, sh, hd).to_numpy())


def test_registry_is_append_only_and_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(X, "REG", tmp_path)
    monkeypatch.setattr(X, "TRIALS", tmp_path / "trials.jsonl")
    monkeypatch.setattr(X, "SERIES", tmp_path / "series.parquet")
    idx = pd.bdate_range("2024-01-02", periods=5)
    calls = []

    def fake(tr, ctx):
        calls.append(tr.id)
        return {"t": 1.0, "mean": 0.1}, pd.Series(np.arange(5.0), index=idx)

    monkeypatch.setattr(X, "run_event", fake)
    trials = [X.Trial("E1", "event", {"a": 1}), X.Trial("E1", "event", {"a": 2})]
    X.run(trials, ctx=None, verbose=False)
    X.run(trials + [X.Trial("E1", "event", {"a": 3})], ctx=None, verbose=False)
    assert calls == [trials[0].id, trials[1].id, 'E1|{"a": 3}']        # never re-run
    lines = (tmp_path / "trials.jsonl").read_text().splitlines()
    assert len(lines) == 3 and json.loads(lines[0])["id"] == trials[0].id
    s = pd.read_parquet(tmp_path / "series.parquet")
    assert list(s.columns) == [t.id for t in trials] + ['E1|{"a": 3}']


def test_round1_space_matches_the_plan():
    t = X.round1_trials(with_earnings=True)
    from collections import Counter
    c = Counter(x.family for x in t)
    assert c == {"P": 9, "T": 7, "E1": 32, "E2": 16, "E3": 24, "E4": 12, "E5": 8}
    assert len(t) == 108 and len({x.id for x in t}) == 108
    assert len(X.v1_trials()) == 10
