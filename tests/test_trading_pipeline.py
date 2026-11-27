"""The trading model's full pipeline on a synthetic world.

These tests exercise the dev selection, signatures, RF adoption path and the
holdout read end-to-end without touching real outcomes. The real data must
not be backtested before the declaration is committed and audited.
"""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from fearsignal.trading import backtest, engine, families as F, gates, signals


def _world(seed=0, N=30):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-03", "2026-11-27")
    T = len(idx)
    cols = [f"T{j:02d}" for j in range(N)]
    lr = rng.normal(0.0003, 0.02, (T, N)) + rng.normal(0, 0.01, (T, 1))
    close = 50 * np.exp(np.cumsum(lr, axis=0))
    open_ = close * np.exp(rng.normal(0, 0.008, (T, N)))
    f = lambda a: pd.DataFrame(a, index=idx, columns=cols)
    vol = f(rng.uniform(2e5, 2e6, (T, N)))
    prices = engine.Prices(f(open_), f(close), f(close) * vol, f(close), vol)
    el = f(np.ones((T, N), bool))
    stress = rng.uniform(size=T) < 0.3                         # ~30% systemic days, like the real dev
    scale = np.where(stress, 1.8, 0.9)[:, None]
    x = f(np.abs(rng.normal(0, 1.0, (T, N))) * scale)
    v = f(rng.normal(0, 1.5, (T, N)))
    g = f(rng.normal(0, 1.2, (T, N)))
    # idiosyncratic shocks on calm days: about 3% of calm name-days, half with an overnight news gap
    calm = ~stress[:, None] & (rng.uniform(size=(T, N)) < 0.03)
    news = rng.uniform(size=(T, N)) < 0.5
    x = x.mask(calm, rng.uniform(3, 6, (T, N)))
    v = v.mask(calm, rng.uniform(-5, -2.1, (T, N)))
    g = g.mask(calm & news, rng.uniform(-5, -2.1, (T, N))).mask(calm & ~news, rng.uniform(-0.5, 0.5, (T, N)))
    share2 = (x >= 2).mean(axis=1)
    systemic = share2 >= 0.10
    mp = pd.Series(rng.normal(0.5, 1.0, T), index=idx)
    mp[idx < "2024-01-02"] = np.nan
    rf = signals.RuleFeatures(
        eligible=el, x=x, Z=f(rng.normal(1, 0.6, (T, N))), q_next=f(rng.uniform(0, 0.1, (T, N))),
        P_panic=f(rng.uniform(0, 0.3, (T, N))), sigma=f(np.full((T, N), 0.02)), r=f(lr),
        rB=pd.Series(lr.mean(axis=1), index=idx), beta=f(rng.uniform(0.7, 1.3, (T, N))),
        u=v + f(rng.normal(0, 0.3, (T, N))), v=v, gamma=g, g_seg=g, g_all=g,
        relvol=f(rng.uniform(0.5, 3, (T, N))), price_ok=el, dv_med20=f(np.full((T, N), 5e7)),
        liquid20=el, systemic=systemic, share_hit2=share2,
        half_day=pd.Series(idx.isin(signals.half_days()), index=idx), M_pct=mp)
    return rf, prices


@pytest.fixture(scope="module")
def world():
    return _world()


def test_dev_selection_runs_end_to_end_and_respects_the_dev_cutoff(world):
    rf, prices = world
    dev = backtest.dev_selection(F.HOLM_FAMILIES, F.SIGNATURES, rf, prices, B=49, R=9, seed=1)
    assert set(dev["families"]) == {"SR", "ND"}
    for k, f in dev["families"].items():
        assert set(f["gates"]) == {"a_t_ge_1", "b_spa_holm", "c_signature", "d_robustness"}
        assert f["adopted"] == all(f["gates"].values())
        D = f["daily"]
        assert D.index.min() >= gates.DEV[0] and D.index.max() <= gates.DEV[1]
        assert "required" in f["signature"]
    # Holm with m = 2: both thresholds 0.05
    assert all(abs(f["holm"]["threshold"] - 0.05) < 1e-12 for f in dev["families"].values())


def test_no_dev_cohort_uses_a_2026_price(world):
    rf, prices = world
    for fam in F.HOLM_FAMILIES:
        for v in fam.variants:
            res, _ = backtest.run_window(v, rf, prices, *gates.DEV)
            if len(res.cohorts):
                assert res.cohorts["exit"].max() <= gates.DEV[1]
            res2, _ = backtest.run_window(v, rf, prices, *gates.DEV, delay=2)
            if len(res2.cohorts):
                assert res2.cohorts["exit"].max() <= gates.DEV[1]


def test_rf_adoption_path_and_holdout_read_run(world):
    rf, prices = world
    r = F.rf_evaluate(rf, prices, *gates.DEV)
    assert r["chosen"] in F.RF_VARIANTS
    assert set(r["criteria"]) == {"A1_var_ratio_le_0_85", "A2_non_inferior_10_25_delay", "A3_cvar_and_maxdd_no_worse"}
    for n in F.RF_VARIANTS:
        E = r["variants"][n]["E"]
        assert E.between(0.25, 1.0).all()
    chosen = [F.SR.variants[0], F.ND.variants[0]]
    h = backtest.holdout_read(chosen, rf, prices)
    for v in chosen:
        d = h[v.id]["daily"]
        assert d.index.min() >= gates.HOLDOUT[0] and d.index.max() <= gates.HOLDOUT[1]


def test_signature_contrasts_change_only_the_declared_condition(world):
    rf, _ = world
    sr1 = F.SR.variants[0]
    m = F.sr_select(rf, sr1)
    mc = F.sr_select(rf, dataclasses.replace(sr1, params={"day": "not_systemic"}))
    sysd = rf.systemic & ~rf.half_day
    assert not (m.any(axis=1) & ~sysd).any()                  # SR only on systemic full days
    assert not (mc.any(axis=1) & (rf.systemic | rf.half_day)).any()
    nd1 = F.ND.variants[0]
    n = F.nd_select(rf, nd1)
    nc = F.nd_select(rf, dataclasses.replace(nd1, params={"news": "no_gap"}))
    assert not (n & nc).any().any()                           # g_seg <= -2 vs g_seg > -1 are disjoint
    g = rf.g_seg.to_numpy()
    assert n.to_numpy().any() and nc.to_numpy().any()
    assert (g[n.to_numpy()] <= -2).all() and (g[nc.to_numpy()] > -1).all()


def test_rf_state_machines():
    idx = pd.bdate_range("2024-01-02", periods=12)
    mp = pd.Series([0.0, 1.6, 1.2, 0.5, 0.5, 0.5, 2.0, 0.9, 1.1, 0.8, 0.8, 0.8], index=idx)
    sh = pd.Series(0.02, index=idx)
    sh.iloc[6] = 0.04                                         # vol doubles on the second spike
    hd = pd.Series(False, index=idx)
    e1 = F.rf_weights("RF1", mp, sh, hd)
    assert e1.tolist() == [1, 0.5, 0.5, 0.75, 1.0, 1.0, 0.5, 0.75, 0.75, 1.0, 1.0, 1.0]
    e2 = F.rf_weights("RF2", mp, sh, hd)
    # state on at day 1 (target 1.0 since sigma_hat = median), off at day 3; on at day 6 with
    # target clip(median/0.04) where the expanding median is still 0.02 -> 0.5
    assert e2.iloc[1] == 1.0 and e2.iloc[6] == pytest.approx(0.5) and e2.iloc[7] == pytest.approx(0.75)
    hd.iloc[3] = True
    assert F.rf_weights("RF1", mp, sh, hd).iloc[3] == 0.5     # half-day keeps the previous exposure
