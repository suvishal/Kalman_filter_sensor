"""E1. Does the Kalman filter beat model-free smoothing, and by how much?

Position RMSE of raw measurements, a moving average, exponential smoothing and
the Kalman filter, across sensor noise levels. The two smoothers get their
best window / alpha per noise level, chosen with knowledge of the truth, so
the comparison is against the strongest version of each baseline.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .. import plotting as pl
from ..analysis import steady_state
from ..filters import KalmanFilter, exponential_smoothing, moving_average, run_filter
from ..metrics import rmse
from .common import TrackingScenario, r3

NOISE_LEVELS = [0.5, 1.0, 2.0, 5.0, 10.0]
WINDOWS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 30]
ALPHAS = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]


def run(out_dir: Path, n_runs: int | None = None, seed: int = 2026) -> dict:
    n_runs = n_runs or 500
    rng = np.random.default_rng(seed)
    base = TrackingScenario(n_steps=800, burn_in=400)
    rows, example = [], None
    for sigma in NOISE_LEVELS:
        sc = replace(base, meas_std=sigma)
        xs, zs = sc.simulate(n_runs, rng)
        truth = xs[sc.burn_in :, :, :2]

        def score(est: np.ndarray) -> float:
            return float(rmse(est[sc.burn_in :] - truth))

        kf = run_filter(KalmanFilter(sc.model(), sc.x0_batch(n_runs), sc.P0), zs)
        ma = {w: score(moving_average(zs, w)) for w in WINDOWS}
        es = {a: score(exponential_smoothing(zs, a)) for a in ALPHAS}
        best_w, best_a = min(ma, key=ma.get), min(es, key=es.get)
        P_pos = steady_state(sc.model()).P_post[:2, :2]
        rows.append(
            {
                "meas_std": sigma,
                "raw": r3(score(zs)),
                "moving_average": r3(ma[best_w]),
                "moving_average_window": best_w,
                "exp_smoothing": r3(es[best_a]),
                "exp_smoothing_alpha": best_a,
                "kalman": r3(score(kf.x[..., :2])),
                "kalman_theory": r3(np.sqrt(np.trace(P_pos))),
                "kalman_velocity_rmse": r3(rmse(kf.x[sc.burn_in :, :, 2:] - xs[sc.burn_in :, :, 2:])),
            }
        )
        if sigma == base.meas_std:
            example = (xs[:300, 0], zs[:300, 0], kf.x[:300, 0])

    pl.apply_style()
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    series = [
        ("raw", "Raw measurements", pl.NEUTRAL, "o"),
        ("moving_average", "Moving average (best window)", pl.ORANGE, "s"),
        ("exp_smoothing", "Exponential smoothing (best alpha)", pl.AQUA, "^"),
        ("kalman", "Kalman filter", pl.BLUE, "D"),
    ]
    for key, label, color, marker in series:
        ax.plot(NOISE_LEVELS, [r[key] for r in rows], color=color, marker=marker, label=label)
    ax.plot(NOISE_LEVELS, [r["kalman_theory"] for r in rows], color=pl.TRUTH, lw=1, ls=":", label="Kalman, Riccati prediction")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Sensor noise standard deviation per axis (m)")
    ax.set_ylabel("Position RMSE (m)")
    ax.set_title("Position error against sensor noise")
    ax.legend()
    figures = [pl.save(fig, out_dir, "e1_rmse_vs_noise")]

    xs, zs, est = example
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.scatter(zs[:, 0], zs[:, 1], s=8, color=pl.NEUTRAL, label="Measurements", zorder=1)
    ax.plot(xs[:, 0], xs[:, 1], color=pl.TRUTH, lw=1.5, label="Truth", zorder=2)
    ax.plot(est[:, 0], est[:, 1], color=pl.BLUE, label="Kalman filter", zorder=3)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"One run, sensor noise {base.meas_std:g} m")
    ax.legend()
    figures.append(pl.save(fig, out_dir, "e1_example_track"))

    return {"n_runs": n_runs, "seed": seed, "scenario": base.__dict__, "rows": rows, "figures": figures}
