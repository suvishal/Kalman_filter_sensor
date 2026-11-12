"""The linear Kalman filter, written from scratch.

Shapes: the state mean ``x`` may carry leading batch dimensions, ``(..., n)``,
while the covariance ``P`` is always a single ``(n, n)`` matrix. This works
because for a linear filter the covariance and gain never depend on the data,
so many Monte Carlo runs can share one covariance recursion. Filters whose
covariance *does* depend on the data (see ``adaptive.py``) must be run one
trajectory at a time.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..models import LinearGaussianModel


@dataclass(frozen=True)
class Innovation:
    """By-products of one measurement update."""

    nu: np.ndarray  # innovation z - H x_pred, (..., m)
    S: np.ndarray  # innovation covariance the filter believes, (m, m)
    K: np.ndarray  # Kalman gain, (n, m)


class KalmanFilter:
    """Predict/update recursion for a :class:`LinearGaussianModel`."""

    def __init__(self, model: LinearGaussianModel, x0: np.ndarray, P0: np.ndarray) -> None:
        self.model = model
        self.x = np.array(x0, dtype=float)
        self.P = np.array(P0, dtype=float)

    def predict(self, u: np.ndarray | None = None) -> None:
        """Time update: push the estimate through the motion model.

        x <- F x + B u
        P <- F P F^T + Q
        """
        m = self.model
        self.x = self.x @ m.F.T
        if u is not None:
            if m.B is None:
                raise ValueError("model has no control matrix B")
            self.x = self.x + u @ m.B.T
        self.P = m.F @ self.P @ m.F.T + m.Q

    def update(self, z: np.ndarray, R: np.ndarray | None = None) -> Innovation:
        """Measurement update: blend the prediction with a measurement.

        ``R`` overrides the model's measurement covariance for this step
        (time-varying sensor quality).
        """
        H = self.model.H
        R = self.model.R if R is None else R
        nu = z - self.x @ H.T
        S = H @ self.P @ H.T + R
        # K = P H^T S^-1, computed by solving a linear system instead of
        # forming an explicit inverse (S is symmetric).
        K = np.linalg.solve(S, H @ self.P).T
        self.x = self.x + nu @ K.T
        # Joseph form: algebraically equal to (I - K H) P for the optimal gain,
        # but stays symmetric positive semi-definite under round-off and is
        # the correct covariance for ANY gain.
        I_KH = np.eye(self.model.n) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        return Innovation(nu=nu, S=S, K=K)

    def diagnostics(self) -> dict[str, np.ndarray]:
        """Extra per-step quantities to log. Subclasses override."""
        return {}


@dataclass
class FilterResult:
    """History of a filter run over T steps."""

    x: np.ndarray  # posterior means, (T, ..., n)
    P: np.ndarray  # posterior covariances, (T, n, n)
    nu: np.ndarray  # innovations, (T, ..., m); NaN where no update happened
    S: np.ndarray  # innovation covariances, (T, m, m); NaN where no update happened
    extras: dict[str, np.ndarray] = field(default_factory=dict)


def run_filter(
    kf: KalmanFilter,
    zs: np.ndarray,
    us: np.ndarray | None = None,
    Rs: np.ndarray | None = None,
) -> FilterResult:
    """Run ``kf`` over a measurement sequence.

    At step k the filter predicts (using ``us[k]`` if given) and then updates
    with ``zs[k]``. A step whose measurement is all-NaN is predict-only, which
    is how slower sensors and outages are represented.

    ``Rs`` optionally gives a per-step measurement covariance, ``(T, m, m)``.
    """
    T = zs.shape[0]
    m = kf.model.m
    xs, Ps, nus = [], [], []
    Ss = np.full((T, m, m), np.nan)
    extras: dict[str, list[np.ndarray]] = {}
    for k in range(T):
        kf.predict(None if us is None else us[k])
        if np.isnan(zs[k]).all():
            nus.append(np.full(zs[k].shape, np.nan))
        else:
            innov = kf.update(zs[k], None if Rs is None else Rs[k])
            nus.append(innov.nu)
            Ss[k] = innov.S
        xs.append(kf.x)
        Ps.append(kf.P)
        for name, value in kf.diagnostics().items():
            extras.setdefault(name, []).append(value)
    return FilterResult(
        x=np.array(xs),
        P=np.array(Ps),
        nu=np.array(nus),
        S=Ss,
        extras={name: np.array(values) for name, values in extras.items()},
    )
