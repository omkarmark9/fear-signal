import json

import numpy as np

from fearsignal import hmm, model as mdl, pipeline, validate


def test_forward_sum_and_back_mean():
    x = np.arange(10, dtype=float)
    f = validate._fwd_sum(x, 3)
    assert f[0] == 1 + 2 + 3 and f[6] == 7 + 8 + 9 and np.isnan(f[7])
    b = validate._back_mean(x, 4)
    assert np.isnan(b[2]) and b[3] == 1.5


def test_har_design_is_causal_in_regressors():
    rng = np.random.default_rng(0)
    R = rng.normal(0, 0.01, 300)
    y, X = validate.har_design(R)
    R2 = R.copy()
    R2[200:] *= 5
    _, X2 = validate.har_design(R2)
    np.testing.assert_array_equal(X[:200], X2[:200])       # regressors use data <= t
    assert not np.allclose(y[195:200], validate.har_design(R2)[0][195:200])  # target is forward


def test_model_roundtrip_through_frozen_dict():
    m = hmm.HMM(hmm.HalfNormalEmission(mdl.V1_S.copy(), np.array([.2, .2, .01]), "state", True), mdl.V1_A)
    d = json.loads(json.dumps(pipeline.model_to_dict(m)))
    m2 = pipeline.model_from_dict(d)
    np.testing.assert_allclose(m2.emission.s, m.emission.s)
    np.testing.assert_allclose(m2.A, m.A, atol=1e-10)
    assert m2.emission.zero_mode == "state" and m2.emission.censor
