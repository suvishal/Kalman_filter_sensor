"""From-scratch Kalman filter research code.

Public API re-exported here so experiments and notebooks can write
``from kalman_research import KalmanFilter, constant_velocity``.
"""

from .filters import AdaptiveKalmanFilter, FilterResult, KalmanFilter, run_filter
from .models import LinearGaussianModel, constant_velocity, imu_driven, imu_driven_with_bias

__all__ = [
    "AdaptiveKalmanFilter",
    "FilterResult",
    "KalmanFilter",
    "LinearGaussianModel",
    "constant_velocity",
    "imu_driven",
    "imu_driven_with_bias",
    "run_filter",
]
