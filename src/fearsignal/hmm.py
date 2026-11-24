r"""Hidden Markov machinery: scaled forward filter, backward smoother, exact EM.

Core recursion (all stocks vectorised, one shared parameter set)::

    pred_t  = alpha_{t-1} A               (pred_t = init on a sequence's first day)
    alpha_t = pred_t * e_t / c_t,         c_t = sum_k pred_t[k] e_t[k] = p(x_t | x_{<t})

``alpha_t = P(S_t | x_1..x_t)`` uses no information after t (causal), and
``sum_t log c_t`` is the exact log-likelihood. Missing days have e_t = 1, so
the filter simply propagates the chain. The backward pass is the matching
scaled recursion ``beta_t = A (e_{t+1} beta_{t+1}) / c_{t+1}`` so that
``gamma_t = alpha_t * beta_t`` and
``xi_t(i,j) = alpha_t(i) A_ij e_{t+1}(j) beta_{t+1}(j) / c_{t+1}``.

Each stock is its own sequence, running from its first to its last eligible
day. Emission models plug in through ``log_emissions`` and ``m_step``.

Emission model (``HalfNormalEmission``)
---------------------------------------
Given state k, a day is a *no-dip* day (the low never goes below the previous
close) with probability ``pi_k``; otherwise x is half-normal with scale
``s_k``: f_k(x) = (2/s_k) phi(x/s_k). Observations are interval data:

=========  =======================================  ==============================
kind       event                                     likelihood given state k
=========  =======================================  ==============================
ZERO       x in [0, eps]                             pi_k + (1-pi_k)(2 Phi(eps/s_k) - 1)
EXACT      x                                         (1-pi_k) f_k(x)
CENSORED   x >= c                                    (1-pi_k) 2 (1 - Phi(c/s_k))
=========  =======================================  ==============================

``zero_mode`` selects how no-dip days are treated:

* ``"none"``  (v1): no-dip days carry no evidence (e = 1) and pi is absent.
* ``"shared"``: one pi for all states. This is the likelihood-based version
  of "no evidence": a state-independent atom cannot move the filter.
* ``"state"``: pi_k free per state. "shared" is nested in it (pi_k all
  equal); the likelihood-ratio statistic tests whether no-dip days are
  informative.

``censor=False`` (v1) evaluates the density at the clip value instead of
the survival function.

EM for the emission is exact: the complete data are the state, the
atom-versus-half-normal indicator on ZERO days, and the latent exact value on
interval-censored days. With weights w = gamma, the M-step is closed form::

    pi_k  = sum_ZERO w r / sum_all w,       r = P(atom | ZERO, k)
    s_k^2 = sum w E[X^2 | obs, k] (1 - r on ZERO days) / sum w (1 - r on ZERO days)

with E[X^2 | X >= c] = s^2 (1 + a lambda(a)), a = c/s, lambda the inverse
Mills ratio, and E[X^2 | X <= eps] = s^2 (1 - 2 a phi(a) / (2 Phi(a) - 1)).
Each EM iteration cannot decrease the likelihood (tested).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field, replace

import numpy as np
from scipy.special import erf, log_ndtr, ndtr

from .config import X_CLIP, ZERO_EPS
from .features import CENSORED, EXACT, MISSING, ZERO, Obs

LOG2 = np.log(2.0)
LOG_SQRT_2PI = 0.5 * np.log(2.0 * np.pi)


# --------------------------------------------------------------------------- utilities

def stationary(A: np.ndarray) -> np.ndarray:
    """Stationary distribution of an irreducible row-stochastic matrix."""
    K = A.shape[0]
    M = np.vstack([A.T - np.eye(K), np.ones(K)])
    b = np.zeros(K + 1)
    b[-1] = 1.0
    p = np.linalg.lstsq(M, b, rcond=None)[0]
    p = np.clip(p, 0, None)
    return p / p.sum()


def sequence_bounds(eligible: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """First and last eligible index per column (T if a column is never eligible)."""
    T = eligible.shape[0]
    any_ = eligible.any(axis=0)
    start = np.where(any_, eligible.argmax(axis=0), T)
    end = np.where(any_, T - 1 - eligible[::-1].argmax(axis=0), -1)
    return start, end


@dataclass
class FilterResult:
    alpha: np.ndarray    # (T,N,K) filtered P(S_t | x_<=t)
    pred: np.ndarray     # (T,N,K) one-step predictive P(S_t | x_<t)
    logc: np.ndarray     # (T,N) log p(x_t | x_<t); 0 outside a sequence
    start: np.ndarray
    end: np.ndarray

    @property
    def loglik(self) -> float:
        return float(self.logc.sum())


def forward(logE: np.ndarray, A: np.ndarray, init: np.ndarray, start: np.ndarray,
            end: np.ndarray) -> FilterResult:
    T, N, K = logE.shape
    alpha = np.empty((T, N, K))
    pred = np.empty((T, N, K))
    logc = np.zeros((T, N))
    a_prev = np.broadcast_to(init, (N, K)).copy()
    for t in range(T):
        p = a_prev @ A
        first = t <= start            # sequence not started yet, or starts today
        p[first] = init
        m = logE[t].max(axis=1, keepdims=True)
        u = p * np.exp(logE[t] - m)
        s = u.sum(axis=1, keepdims=True)
        a = u / s
        pred[t] = p
        alpha[t] = a
        active = (t >= start) & (t <= end)
        logc[t] = np.where(active, np.log(s[:, 0]) + m[:, 0], 0.0)
        a_prev = a
    return FilterResult(alpha, pred, logc, start, end)


def backward(logE: np.ndarray, A: np.ndarray, fr: FilterResult) -> np.ndarray:
    """Scaled backward variables matched to ``forward``'s normalisers."""
    T, N, K = logE.shape
    beta = np.ones((T, N, K))
    for t in range(T - 2, -1, -1):
        inside = (t >= fr.start) & (t + 1 <= fr.end)
        if not inside.any():
            continue
        w = np.exp(logE[t + 1] - fr.logc[t + 1][:, None]) * beta[t + 1]
        b = w @ A.T
        beta[t] = np.where(inside[:, None], b, 1.0)
    return beta


def smooth(logE: np.ndarray, A: np.ndarray, init: np.ndarray, start, end):
    """Return (FilterResult, gamma, xi_sum, gamma_at_start)."""
    fr = forward(logE, A, init, start, end)
    beta = backward(logE, A, fr)
    gamma = fr.alpha * beta
    gamma /= gamma.sum(axis=2, keepdims=True)
    T, N, K = logE.shape
    tt = np.arange(T - 1)[:, None]
    pair = (tt >= start[None, :]) & (tt + 1 <= end[None, :])           # (T-1, N)
    w = np.exp(logE[1:] - fr.logc[1:][:, :, None]) * beta[1:]           # (T-1, N, K)
    xi = np.einsum("tni,tnj,tn->ij", fr.alpha[:-1], w, pair.astype(float)) * A
    has = start < T
    g0 = gamma[start[has], np.flatnonzero(has)]                          # (N', K)
    return fr, gamma, xi, g0


# --------------------------------------------------------------------------- emissions

def _std_trunc_below_sq(a: np.ndarray) -> np.ndarray:
    """E[Z^2 | |Z| <= a] for Z ~ N(0,1); series a^2 (1/3 - 2a^2/45) avoids cancellation."""
    a = np.atleast_1d(np.asarray(a, float))
    out = a**2 * (1.0 / 3.0 - 2.0 * a**2 / 45.0)
    big = a > 0.05
    ab = a[big]
    phi = np.exp(-0.5 * ab**2 - LOG_SQRT_2PI)
    out[big] = 1.0 - 2.0 * ab * phi / erf(ab / np.sqrt(2.0))
    return out


def _std_trunc_above_sq(a: np.ndarray) -> np.ndarray:
    """E[Z^2 | |Z| >= a] = 1 + a * lambda(a), lambda = phi / (1 - Phi)."""
    a = np.asarray(a, float)
    mills = np.exp(-0.5 * a**2 - LOG_SQRT_2PI - log_ndtr(-a))
    return 1.0 + a * mills


@dataclass
class HalfNormalEmission:
    s: np.ndarray                     # (K,) scales
    pi: np.ndarray | None = None      # (K,) no-dip probabilities; None iff zero_mode == "none"
    zero_mode: str = "state"          # "none" | "shared" | "state"
    censor: bool = True
    eps: float = ZERO_EPS
    clip: float = X_CLIP

    @property
    def K(self) -> int:
        return len(self.s)

    def log_emissions(self, obs: Obs) -> np.ndarray:
        s = self.s[None, None, :]
        x = np.where(obs.kind == EXACT, obs.x, 0.0)[:, :, None]
        kind = obs.kind[:, :, None]
        logf = LOG2 - np.log(s) - 0.5 * (x / s) ** 2 - LOG_SQRT_2PI
        if self.censor:
            logcens = LOG2 + log_ndtr(-self.clip / s)
        else:
            logcens = LOG2 - np.log(s) - 0.5 * (self.clip / s) ** 2 - LOG_SQRT_2PI
        out = np.zeros(obs.kind.shape + (self.K,))
        if self.zero_mode == "none":
            out = np.where(kind == EXACT, logf, out)
            out = np.where(kind == CENSORED, logcens, out)
            return out
        pi = self.pi[None, None, :]
        hn_zero = erf(self.eps / (s * np.sqrt(2.0)))       # P(HalfNormal <= eps)
        logzero = np.log(pi + (1.0 - pi) * hn_zero)
        l1p = np.log1p(-pi)
        out = np.where(kind == ZERO, logzero, out)
        out = np.where(kind == EXACT, l1p + logf, out)
        out = np.where(kind == CENSORED, l1p + logcens, out)
        return out

    def m_step(self, obs: Obs, gamma: np.ndarray) -> "HalfNormalEmission":
        kind = obs.kind
        x = np.where(kind == EXACT, obs.x, 0.0)
        s = self.s
        w_exact = np.where((kind == EXACT)[:, :, None], gamma, 0.0)
        w_cens = np.where((kind == CENSORED)[:, :, None], gamma, 0.0)
        num = np.einsum("tnk,tn->k", w_exact, x**2)
        den = w_exact.sum(axis=(0, 1))
        if self.censor:
            num += w_cens.sum(axis=(0, 1)) * s**2 * _std_trunc_above_sq(self.clip / s)
        else:
            num += w_cens.sum(axis=(0, 1)) * self.clip**2
        den += w_cens.sum(axis=(0, 1))
        new_pi = None
        if self.zero_mode != "none":
            pi = self.pi
            hn_zero = erf(self.eps / (s * np.sqrt(2.0)))
            r = pi / (pi + (1 - pi) * hn_zero)                       # P(atom | ZERO, k)
            w_zero = np.where((kind == ZERO)[:, :, None], gamma, 0.0).sum(axis=(0, 1))
            w_all = np.where((kind != MISSING)[:, :, None], gamma, 0.0).sum(axis=(0, 1))
            num += w_zero * (1 - r) * s**2 * _std_trunc_below_sq(self.eps / s)
            den += w_zero * (1 - r)
            if self.zero_mode == "state":
                new_pi = w_zero * r / w_all
            else:
                new_pi = np.full(self.K, (w_zero * r).sum() / w_all.sum())
        new_s = np.sqrt(num / den)
        return replace(self, s=new_s, pi=new_pi)

    def permute(self, order: np.ndarray) -> "HalfNormalEmission":
        return replace(self, s=self.s[order], pi=None if self.pi is None else self.pi[order])

    def sort_key(self) -> np.ndarray:
        return self.s

    def n_params(self) -> int:
        k = self.K
        return k + {"none": 0, "shared": 1, "state": k}[self.zero_mode]

    # Predictive tail probability of the observation given each state.
    def tail_prob(self, threshold: float) -> np.ndarray:
        p = 2.0 * ndtr(-threshold / self.s)
        return p if self.pi is None else (1.0 - self.pi) * p


@dataclass
class GaussianEmission:
    """Gaussian emission for scalar series (the market layer)."""

    mu: np.ndarray
    sd: np.ndarray
    sd_floor: float = 1e-3

    @property
    def K(self) -> int:
        return len(self.mu)

    def log_emissions(self, obs: Obs) -> np.ndarray:
        y = np.where(obs.kind != MISSING, obs.x, 0.0)[:, :, None]
        z = (y - self.mu) / self.sd
        out = -0.5 * z**2 - np.log(self.sd) - LOG_SQRT_2PI
        return np.where((obs.kind != MISSING)[:, :, None], out, 0.0)

    def m_step(self, obs: Obs, gamma: np.ndarray) -> "GaussianEmission":
        ok = (obs.kind != MISSING)[:, :, None]
        w = np.where(ok, gamma, 0.0)
        y = np.where(obs.kind != MISSING, obs.x, 0.0)[:, :, None]
        W = w.sum(axis=(0, 1))
        mu = (w * y).sum(axis=(0, 1)) / W
        var = (w * (y - mu) ** 2).sum(axis=(0, 1)) / W
        return replace(self, mu=mu, sd=np.maximum(np.sqrt(var), self.sd_floor))

    def permute(self, order):
        return replace(self, mu=self.mu[order], sd=self.sd[order])

    def sort_key(self):
        return self.mu

    def n_params(self):
        return 2 * self.K

    def tail_prob(self, threshold: float) -> np.ndarray:
        return ndtr(-(threshold - self.mu) / self.sd)


# --------------------------------------------------------------------------- model + EM

@dataclass
class HMM:
    emission: HalfNormalEmission | GaussianEmission
    A: np.ndarray
    init: np.ndarray | None = None        # EM-fitted initial law; filtering uses stationary(A)
    loglik: float = float("nan")
    n_obs: int = 0
    history: list = field(default_factory=list)
    converged: bool = False

    @property
    def K(self) -> int:
        return self.A.shape[0]

    @property
    def stationary(self) -> np.ndarray:
        return stationary(self.A)

    def n_params(self) -> int:
        K = self.K
        return self.emission.n_params() + K * (K - 1) + (K - 1)

    def filter(self, obs: Obs, init: np.ndarray | None = None) -> FilterResult:
        """Causal filter. New sequences start from the stationary law unless given."""
        start, end = sequence_bounds(obs.eligible)
        # Filtering continues past a stock's last eligible day only by propagation;
        # for live use every column runs to the end of the panel.
        end = np.where(start < obs.kind.shape[0], obs.kind.shape[0] - 1, end)
        logE = self.emission.log_emissions(obs)
        return forward(logE, self.A, self.stationary if init is None else init, start, end)

    def canonical(self) -> "HMM":
        order = np.argsort(self.emission.sort_key())
        return replace(
            self,
            emission=self.emission.permute(order),
            A=self.A[np.ix_(order, order)],
            init=None if self.init is None else self.init[order],
        )


def em(model: HMM, obs: Obs, max_iter: int = 500, tol: float = 1e-8,
       fit_init: bool = True) -> HMM:
    """Baum-Welch from the given starting model until relative change < tol.

    The returned model's ``loglik`` is the likelihood *of the returned
    parameters*; ``history`` holds the likelihood at every iterate.
    """
    start, end = sequence_bounds(obs.eligible)
    init = model.init if model.init is not None else model.stationary
    em_ = model.emission
    A = model.A.copy()
    hist: list[float] = []
    converged = False
    for _ in range(max_iter):
        logE = em_.log_emissions(obs)
        fr, gamma, xi, g0 = smooth(logE, A, init, start, end)
        hist.append(fr.loglik)
        if len(hist) > 1 and abs(hist[-1] - hist[-2]) <= tol * abs(hist[-2]):
            converged = True
            break
        A = xi / xi.sum(axis=1, keepdims=True)
        if fit_init and len(g0):
            init = g0.mean(axis=0)
        em_ = em_.m_step(obs, gamma)
    else:  # iteration budget exhausted: score the final update
        hist.append(forward(em_.log_emissions(obs), A, init, start, end).loglik)
    if not converged:
        warnings.warn(f"EM stopped at max_iter={max_iter} before reaching tol={tol:g}",
                      RuntimeWarning, stacklevel=2)
    n_obs = int(obs.eligible.sum())
    return HMM(em_, A, init, hist[-1], n_obs, hist, converged).canonical()


def fit(starts: list[HMM], obs: Obs, **kw) -> HMM:
    """Run EM from several starting points and keep the best likelihood.

    The winner's ``converged`` flag says whether *it* met the tolerance; a
    non-converged winner is a lower bound on the attainable likelihood.
    """
    fits = [em(m, obs, **kw) for m in starts]
    return max(fits, key=lambda m: m.loglik)
