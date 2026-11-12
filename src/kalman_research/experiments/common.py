"""The tracking scenario shared by the single-sensor experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..models import LinearGaussianModel, constant_velocity
from ..simulation import simulate_linear


@dataclass(frozen=True)
class TrackingScenario:
    """A 2D target moving at nearly constant velocity, seen by a position sensor."""

    dt: float = 0.1  # s
    accel_std: float = 0.5  # m/s^2, true manoeuvre intensity
    meas_std: float = 2.0  # m, true sensor noise per axis
    n_steps: int = 300
    burn_in: int = 100  # steps discarded before computing steady-state metrics
    mean_velocity: tuple[float, float] = (5.0, 2.0)  # m/s
    init_vel_std: float = 2.0  # m/s

    def model(self) -> LinearGaussianModel:
        return constant_velocity(self.dt, self.accel_std, self.meas_std)

    @property
    def x0_mean(self) -> np.ndarray:
        return np.array([0.0, 0.0, *self.mean_velocity])

    @property
    def P0(self) -> np.ndarray:
        return np.diag([self.meas_std**2] * 2 + [self.init_vel_std**2] * 2)

    def simulate(self, n_runs: int, rng: np.random.Generator, Rs: np.ndarray | None = None):
        """Truth and measurements; the initial state is drawn from N(x0_mean, P0).

        Drawing the truth from the same prior the filter is given is what
        makes the filter's initial covariance honest.
        """
        x0 = rng.multivariate_normal(self.x0_mean, self.P0, size=n_runs)
        return simulate_linear(self.model(), x0, self.n_steps, rng, Rs=Rs)

    def x0_batch(self, n_runs: int) -> np.ndarray:
        return np.tile(self.x0_mean, (n_runs, 1))


def r3(value) -> float:
    """Round to 4 significant figures for the JSON summaries."""
    return float(f"{float(value):.4g}")
