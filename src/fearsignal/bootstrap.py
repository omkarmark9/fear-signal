r"""Parameter uncertainty by a synchronized moving-block bootstrap in time.

What limits precision here is not the 79,000 stock-days but the handful of
stress episodes in 2023-2025, and stocks share those episodes. So resampling
must be over *time*, keeping the cross-section intact:

1. Draw ceil(T_eff / b) block start dates uniformly from the eligible span,
   with block length b = 63 sessions (one quarter, several times the expected
   regime durations 1/(1 - A_kk)).
2. Each (block, stock) pair becomes its own sequence; all stocks in a block
   share the same dates, so cross-sectional dependence is preserved. The chain
   restarts at each block start (one transition per block per stock is lost).
3. Refit by EM, warm-started at the point estimate, and put states in
   canonical order.

The replicate parameter sets give confidence intervals for (s, pi, A) and,
pushed through the filter on the actual data, uncertainty bands for every
derived score.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from . import hmm
from .features import MISSING, Obs

BLOCK = 63


def block_resample(obs: Obs, rng, block: int = BLOCK) -> Obs:
    T, N = obs.x.shape
    any_el = obs.eligible.any(axis=1)
    t0, t1 = int(np.argmax(any_el)), int(T - np.argmax(any_el[::-1]))
    span = t1 - t0
    n_blocks = int(np.ceil(span / block))
    starts = rng.integers(t0, t1 - block + 1, size=n_blocks)
    xs = [obs.x[s:s + block] for s in starts]
    ks = [obs.kind[s:s + block] for s in starts]
    x = np.concatenate(xs, axis=1)
    kind = np.concatenate(ks, axis=1)
    reg = None if obs.regime is None else np.concatenate([obs.regime[s:s + block] for s in starts], axis=1)
    return Obs(x, kind, obs.dates[:block], [f"b{j}" for j in range(x.shape[1])], reg)


def replicate_fits(point: hmm.HMM, obs: Obs, n_boot: int = 200, seed: int = 0,
                   block: int = BLOCK, max_iter: int = 400, tol: float = 1e-8) -> list[hmm.HMM]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        bo = block_resample(obs, rng, block)
        start = replace(point, init=None)
        out.append(hmm.em(start, bo, max_iter=max_iter, tol=tol))
    return out


def parameter_table(point: hmm.HMM, reps: list[hmm.HMM], level: float = 0.90) -> dict:
    lo_q, hi_q = (1 - level) / 2, (1 + level) / 2

    def ci(get):
        vals = np.array([get(m) for m in reps])
        return {"point": np.asarray(get(point)).tolist(),
                "lo": np.quantile(vals, lo_q, axis=0).tolist(),
                "hi": np.quantile(vals, hi_q, axis=0).tolist()}

    tab = {"s": ci(lambda m: m.emission.s), "A": ci(lambda m: m.A),
           "stationary": ci(lambda m: m.stationary),
           "expected_duration": ci(lambda m: 1.0 / (1.0 - np.diagonal(m.A, axis1=-2, axis2=-1)))}
    if point.emission.pi is not None:
        tab["pi"] = ci(lambda m: m.emission.pi)
    tab["level"] = level
    tab["n_boot"] = len(reps)
    return tab
