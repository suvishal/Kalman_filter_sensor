"""Ground truth and sensor simulators. Everything takes an explicit ``rng``."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import LinearGaussianModel, _kinematic_blocks


def _sample(rng: np.random.Generator, cov: np.ndarray, size: tuple[int, ...]) -> np.ndarray:
    """Zero-mean Gaussian samples; SVD factorisation tolerates singular covariances."""
    return rng.multivariate_normal(np.zeros(cov.shape[0]), cov, size=size, method="svd")


def simulate_linear(
    model: LinearGaussianModel,
    x0: np.ndarray,
    n_steps: int,
    rng: np.random.Generator,
    us: np.ndarray | None = None,
    Rs: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate ``model`` for ``n_steps`` from initial state(s) ``x0``.

    ``x0`` may be ``(n,)`` or batched ``(N, n)``. Returns ``(xs, zs)`` with
    shapes ``(T, ..., n)`` and ``(T, ..., m)``; index k holds the state after
    k+1 transitions and its measurement. ``Rs`` (T, m, m) makes the
    measurement noise time-varying.
    """
    x = np.array(x0, dtype=float)
    batch = x.shape[:-1]
    ws = _sample(rng, model.Q, (n_steps, *batch))
    if Rs is None:
        vs = _sample(rng, model.R, (n_steps, *batch))
    else:
        vs = np.stack([_sample(rng, Rs[k], batch) for k in range(n_steps)])
    xs = np.empty((n_steps, *x.shape))
    for k in range(n_steps):
        x = x @ model.F.T + ws[k]
        if us is not None:
            x = x + us[k] @ model.B.T
        xs[k] = x
    zs = xs @ model.H.T + vs
    return xs, zs


@dataclass(frozen=True)
class FusionScenario:
    """Parameters of the simulated GPS + IMU experiment (2D)."""

    dt: float = 0.01  # IMU period, s (100 Hz)
    duration: float = 60.0  # s
    gps_every: int = 100  # IMU steps per GPS fix (1 Hz)
    gps_std: float = 3.0  # m, per axis
    accel_noise_std: float = 0.2  # m/s^2, white, per axis
    bias_init_std: float = 0.15  # m/s^2, per axis
    bias_walk_std: float = 0.005  # m/s^2 per sqrt(s)
    amplitude: float = 50.0  # m, half-width of the figure-eight
    init_pos_std: float = 3.0  # m, uncertainty of the starting position
    init_vel_std: float = 0.5  # m/s, uncertainty of the starting velocity
    outage: tuple[float, float] | None = None  # (start, end) in s with no GPS

    @property
    def n_steps(self) -> int:
        return int(round(self.duration / self.dt))


@dataclass(frozen=True)
class FusionData:
    t: np.ndarray  # time of each step, (T,)
    x: np.ndarray  # true [p, v], (T, N, 4)
    bias: np.ndarray  # true accelerometer bias state, (T, N, 2)
    accel_meas: np.ndarray  # accelerometer readings, (T, N, 2); row k drives step k
    gps: np.ndarray  # GPS fixes, (T, N, 2); NaN where there is no fix
    x0_nominal: np.ndarray  # the starting state the estimators are told, (4,)


def figure_eight_accel(t: np.ndarray, amplitude: float, period: float) -> tuple[np.ndarray, np.ndarray]:
    """Acceleration and initial state of ``p(t) = [A sin(wt), A/2 sin(2wt)]``."""
    w = 2.0 * np.pi / period
    accel = np.stack([-amplitude * w**2 * np.sin(w * t), -2.0 * amplitude * w**2 * np.sin(2.0 * w * t)], axis=-1)
    x0 = np.array([0.0, 0.0, amplitude * w, amplitude * w])
    return accel, x0


def simulate_fusion(sc: FusionScenario, n_runs: int, rng: np.random.Generator) -> FusionData:
    """Simulate ``n_runs`` realisations of the GPS + IMU scenario.

    The truth is produced by the same discrete kinematics the filters use
    (acceleration held constant over each IMU step), driven by the sampled
    figure-eight acceleration. This keeps the filter model exact, so any
    inconsistency seen in the results comes from the noise model, not from
    discretisation error.

    Simplification: the accelerometer measures acceleration directly in the
    world frame. A real IMU measures in the body frame and needs orientation,
    which makes the problem nonlinear (EKF territory).
    """
    T, dims = sc.n_steps, 2
    F, G = _kinematic_blocks(sc.dt, dims)
    t = sc.dt * np.arange(1, T + 1)
    accel, x0_nominal = figure_eight_accel(t - sc.dt, sc.amplitude, sc.duration)

    init_std = np.array([sc.init_pos_std] * dims + [sc.init_vel_std] * dims)
    x = x0_nominal + init_std * rng.standard_normal((n_runs, 2 * dims))
    xs = np.empty((T, n_runs, 2 * dims))
    for k in range(T):
        x = x @ F.T + accel[k] @ G.T
        xs[k] = x

    bias0 = sc.bias_init_std * rng.standard_normal((n_runs, dims))
    walk = sc.bias_walk_std * np.sqrt(sc.dt) * rng.standard_normal((T, n_runs, dims))
    # bias[k] is the bias state after step k, aligned with xs[k]; the reading
    # that drives step k was taken with the bias before that step's increment.
    bias = bias0 + np.cumsum(walk, axis=0)
    accel_meas = accel[:, None, :] + (bias - walk) + sc.accel_noise_std * rng.standard_normal((T, n_runs, dims))

    gps = np.full((T, n_runs, dims), np.nan)
    fix = np.arange(sc.gps_every - 1, T, sc.gps_every)
    if sc.outage is not None:
        fix = fix[(t[fix] < sc.outage[0]) | (t[fix] >= sc.outage[1])]
    gps[fix] = xs[fix][..., :dims] + sc.gps_std * rng.standard_normal((len(fix), n_runs, dims))

    return FusionData(t=t, x=xs, bias=bias, accel_meas=accel_meas, gps=gps, x0_nominal=x0_nominal)


def dead_reckon(x0: np.ndarray, accel_meas: np.ndarray, dt: float) -> np.ndarray:
    """Integrate accelerometer readings from ``x0`` with no corrections."""
    dims = accel_meas.shape[-1]
    F, G = _kinematic_blocks(dt, dims)
    x = np.broadcast_to(x0, (*accel_meas.shape[1:-1], 2 * dims)).astype(float)
    xs = np.empty((accel_meas.shape[0], *x.shape))
    for k in range(accel_meas.shape[0]):
        x = x @ F.T + accel_meas[k] @ G.T
        xs[k] = x
    return xs
