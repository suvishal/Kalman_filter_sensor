"""Linear-Gaussian state-space models.

Every model in this project has the form

    x_k = F x_{k-1} + B u_{k-1} + w_{k-1},    w ~ N(0, Q)
    z_k = H x_k + v_k,                        v ~ N(0, R)

State ordering is always ``[positions..., velocities..., (biases...)]`` so that
"the first ``dims`` entries are position" holds for every model.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class LinearGaussianModel:
    """Matrices of a time-invariant linear-Gaussian system."""

    F: np.ndarray  # state transition, (n, n)
    H: np.ndarray  # measurement, (m, n)
    Q: np.ndarray  # process noise covariance, (n, n)
    R: np.ndarray  # measurement noise covariance, (m, m)
    B: np.ndarray | None = None  # control input, (n, p)

    def __post_init__(self) -> None:
        n, m = self.F.shape[0], self.H.shape[0]
        if self.F.shape != (n, n) or self.Q.shape != (n, n):
            raise ValueError("F and Q must be (n, n)")
        if self.H.shape != (m, n) or self.R.shape != (m, m):
            raise ValueError("H must be (m, n) and R must be (m, m)")
        if self.B is not None and self.B.shape[0] != n:
            raise ValueError("B must have n rows")

    @property
    def n(self) -> int:
        """State dimension."""
        return self.F.shape[0]

    @property
    def m(self) -> int:
        """Measurement dimension."""
        return self.H.shape[0]

    def with_noise(self, Q: np.ndarray | None = None, R: np.ndarray | None = None) -> "LinearGaussianModel":
        """Copy of this model with different noise covariances.

        Used to build a *filter* model that deliberately disagrees with the
        *true* model (mis-tuning experiments).
        """
        return replace(self, Q=self.Q if Q is None else Q, R=self.R if R is None else R)


def _kinematic_blocks(dt: float, dims: int) -> tuple[np.ndarray, np.ndarray]:
    """F and G for position/velocity kinematics driven by acceleration.

    Over one step with constant acceleration ``a``:
        p += v*dt + 0.5*dt^2*a
        v += dt*a
    so ``x_k = F x_{k-1} + G a``.
    """
    eye = np.eye(dims)
    F = np.block([[eye, dt * eye], [np.zeros((dims, dims)), eye]])
    G = np.vstack([0.5 * dt**2 * eye, dt * eye])
    return F, G


def constant_velocity(dt: float, accel_std: float, meas_std: float, dims: int = 2) -> LinearGaussianModel:
    """Nearly-constant-velocity model with a position sensor.

    State ``[p, v]`` (2*dims). The object is assumed to move at constant
    velocity, perturbed by an unknown acceleration that is constant over each
    step and drawn from N(0, accel_std^2) (the "discrete white noise
    acceleration" model). That gives ``Q = accel_std^2 * G G^T``.
    """
    F, G = _kinematic_blocks(dt, dims)
    H = np.hstack([np.eye(dims), np.zeros((dims, dims))])
    return LinearGaussianModel(F=F, H=H, Q=accel_std**2 * G @ G.T, R=meas_std**2 * np.eye(dims))


def imu_driven(dt: float, accel_noise_std: float, gps_std: float, dims: int = 2) -> LinearGaussianModel:
    """Kinematics driven by a measured acceleration, corrected by GPS position.

    State ``[p, v]``. The accelerometer reading is the control input ``u``.
    Its white noise enters through the same matrix as the input, so
    ``Q = accel_noise_std^2 * G G^T``. Accelerometer bias is NOT modelled.
    """
    F, G = _kinematic_blocks(dt, dims)
    H = np.hstack([np.eye(dims), np.zeros((dims, dims))])
    return LinearGaussianModel(F=F, H=H, Q=accel_noise_std**2 * G @ G.T, R=gps_std**2 * np.eye(dims), B=G)


def imu_driven_with_bias(
    dt: float, accel_noise_std: float, bias_walk_std: float, gps_std: float, dims: int = 2
) -> LinearGaussianModel:
    """Like :func:`imu_driven` but the accelerometer bias is part of the state.

    State ``[p, v, b]`` (3*dims). The sensor reads ``u = a + b + noise`` so the
    true acceleration is ``a = u - b - noise``; substituting into the
    kinematics puts ``-G`` in the bias column of F. The bias is a random walk
    with per-step variance ``bias_walk_std^2 * dt``.
    """
    F_pv, G = _kinematic_blocks(dt, dims)
    n = 3 * dims
    F = np.eye(n)
    F[: 2 * dims, : 2 * dims] = F_pv
    F[: 2 * dims, 2 * dims :] = -G
    B = np.vstack([G, np.zeros((dims, dims))])
    Q = np.zeros((n, n))
    Q[: 2 * dims, : 2 * dims] = accel_noise_std**2 * G @ G.T
    Q[2 * dims :, 2 * dims :] = bias_walk_std**2 * dt * np.eye(dims)
    H = np.hstack([np.eye(dims), np.zeros((dims, 2 * dims))])
    return LinearGaussianModel(F=F, H=H, Q=Q, R=gps_std**2 * np.eye(dims), B=B)
