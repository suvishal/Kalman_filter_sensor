"""Accuracy and consistency metrics.

Accuracy asks "how far is the estimate from the truth?" (RMSE).
Consistency asks "is the filter's own covariance an honest description of
that error?" (NEES, NIS). A filter can be accurate but dishonest, and the
experiments show exactly that.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def rmse(err: np.ndarray, axis: int | tuple[int, ...] | None = None) -> np.ndarray:
    """Root-mean-square of the Euclidean error norm.

    ``err`` has shape ``(..., d)``; the last axis is the vector component.
    ``axis`` selects which of the remaining axes to average over (default all).
    """
    sq = np.sum(err**2, axis=-1)
    return np.sqrt(np.mean(sq, axis=axis))


def nees(err: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Normalised estimation error squared, ``e^T P^-1 e``.

    ``err`` is ``(T, ..., n)`` and ``P`` is ``(T, n, n)``. For a consistent
    filter each value is chi-square distributed with n degrees of freedom, so
    its mean is n.
    """
    Pinv = np.linalg.inv(P)
    return np.einsum("t...i,tij,t...j->t...", err, Pinv, err)


def nis(nu: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Normalised innovation squared, ``nu^T S^-1 nu``.

    Same idea as NEES but computed from innovations, so it needs no ground
    truth and can be monitored on a real system. Steps without an update are
    NaN in, NaN out.
    """
    out = np.full(nu.shape[:-1], np.nan)
    ok = ~np.isnan(S).any(axis=(1, 2))
    out[ok] = np.einsum("t...i,tij,t...j->t...", nu[ok], np.linalg.inv(S[ok]), nu[ok])
    return out


def chi2_mean_bounds(dof: int, n_samples: int, confidence: float = 0.95) -> tuple[float, float]:
    """Two-sided acceptance interval for the mean of ``n_samples`` chi-square(dof) values.

    The sum of N independent chi-square(dof) variables is chi-square(N*dof).
    """
    tail = (1.0 - confidence) / 2.0
    total = dof * n_samples
    return (
        float(stats.chi2.ppf(tail, total) / n_samples),
        float(stats.chi2.ppf(1.0 - tail, total) / n_samples),
    )


def autocorrelation(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Sample autocorrelation of a scalar sequence for lags 0..max_lag.

    The innovations of an optimal filter are white, so every lag >= 1 should
    be near zero (within about +-2/sqrt(T)).
    """
    x = x - np.mean(x)
    denom = np.dot(x, x)
    return np.array([np.dot(x[: len(x) - lag], x[lag:]) / denom for lag in range(max_lag + 1)])
