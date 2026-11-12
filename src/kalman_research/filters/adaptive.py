"""Kalman filter that estimates its measurement noise covariance R online.

Idea (residual-based covariance matching): after an update, the post-fit
residual ``eps = z - H x_post`` satisfies, for a correctly tuned filter,

    E[eps eps^T] = R - H P_post H^T      =>      R = E[eps eps^T] + H P_post H^T

(derivation in docs/01-theory.md). Replacing the expectation with an
exponentially weighted average gives a running estimate of R. Both terms are
positive semi-definite, so the estimate is always a valid covariance, unlike
the innovation-based form ``E[nu nu^T] - H P_pred H^T``.

Only R is adapted. Q is assumed known: adapting both from the same innovation
sequence is poorly conditioned.
"""

from __future__ import annotations

import numpy as np

from ..models import LinearGaussianModel
from .kalman import Innovation, KalmanFilter


class AdaptiveKalmanFilter(KalmanFilter):
    """Kalman filter with residual-based online estimation of R.

    ``forgetting`` in (0, 1) is the weight kept on the previous estimate each
    step; the effective memory is about ``1 / (1 - forgetting)`` updates.
    Because the covariance now depends on the data, this filter does not
    support batched states.
    """

    def __init__(self, model: LinearGaussianModel, x0: np.ndarray, P0: np.ndarray, forgetting: float = 0.98) -> None:
        super().__init__(model, x0, P0)
        if not 0.0 < forgetting < 1.0:
            raise ValueError("forgetting must be in (0, 1)")
        if self.x.ndim != 1:
            raise ValueError("AdaptiveKalmanFilter runs one trajectory at a time")
        self.forgetting = forgetting
        self.R_hat = np.array(model.R, dtype=float)

    def update(self, z: np.ndarray, R: np.ndarray | None = None) -> Innovation:
        if R is not None:
            raise ValueError("AdaptiveKalmanFilter estimates R itself")
        innov = super().update(z, self.R_hat)
        H = self.model.H
        eps = z - H @ self.x
        sample = np.outer(eps, eps) + H @ self.P @ H.T
        self.R_hat = self.forgetting * self.R_hat + (1.0 - self.forgetting) * sample
        return innov

    def diagnostics(self) -> dict[str, np.ndarray]:
        return {"R_hat": self.R_hat.copy()}
