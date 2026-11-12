"""Model-free smoothers used as baselines.

They operate directly on the measurement sequence ``zs`` of shape
``(T, ..., m)`` and return an array of the same shape. Both are causal: the
output at step k uses only measurements up to k, like the Kalman filter.
"""

from __future__ import annotations

import numpy as np


def moving_average(zs: np.ndarray, window: int) -> np.ndarray:
    """Mean of the last ``window`` measurements (fewer at the start)."""
    if window < 1:
        raise ValueError("window must be >= 1")
    csum = np.cumsum(zs, axis=0)
    out = np.empty_like(zs, dtype=float)
    for k in range(zs.shape[0]):
        lo = k - window
        total = csum[k] - (csum[lo] if lo >= 0 else 0.0)
        out[k] = total / min(k + 1, window)
    return out


def exponential_smoothing(zs: np.ndarray, alpha: float) -> np.ndarray:
    """``s_k = alpha * z_k + (1 - alpha) * s_{k-1}``, with ``s_0 = z_0``."""
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    out = np.empty_like(zs, dtype=float)
    out[0] = zs[0]
    for k in range(1, zs.shape[0]):
        out[k] = alpha * zs[k] + (1.0 - alpha) * out[k - 1]
    return out
