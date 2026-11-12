"""E4. GPS + IMU fusion: what each sensor contributes and why bias must be modelled.

Five position estimators on a figure-eight trajectory (IMU 100 Hz, GPS 1 Hz):
  - last GPS fix held until the next one
  - IMU dead reckoning (no GPS)
  - GPS-only Kalman filter (constant-velocity model, best process noise)
  - fused filter that ignores accelerometer bias (4 states)
  - fused filter that estimates accelerometer bias (6 states)
The scenario is run again with a 15 s GPS outage.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .. import plotting as pl
from ..filters import KalmanFilter, run_filter
from ..metrics import nees, rmse
from ..models import constant_velocity, imu_driven, imu_driven_with_bias
from ..simulation import FusionData, FusionScenario, dead_reckon, simulate_fusion
from .common import r3

# White acceleration per 10 ms step; large values are needed to stand in for the
# smooth (strongly time-correlated) acceleration of the real trajectory.
CV_ACCEL_STDS = [2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
BURN_IN_S = 10.0
OUTAGE = (30.0, 45.0)
LABELS = {
    "gps_hold": ("GPS, last fix held", pl.NEUTRAL),
    "dead_reckoning": ("IMU dead reckoning", pl.YELLOW),
    "kf_gps_only": ("Kalman filter, GPS only", pl.AQUA),
    "fused_no_bias": ("Fused, bias ignored", pl.ORANGE),
    "fused_bias": ("Fused, bias estimated", pl.BLUE),
}


def hold_last_fix(data: FusionData) -> np.ndarray:
    """Zero-order hold of GPS fixes (the nominal start until the first fix)."""
    out = np.empty_like(data.gps)
    last = np.broadcast_to(data.x0_nominal[:2], data.gps.shape[1:])
    for k in range(data.gps.shape[0]):
        if not np.isnan(data.gps[k]).all():
            last = data.gps[k]
        out[k] = last
    return out


def estimate_all(sc: FusionScenario, data: FusionData, cv_accel_std: float | None = None) -> tuple[dict, dict, float]:
    """Run every estimator. Returns (states, FilterResults, CV accel std used).

    The GPS-only filter's process noise is picked from ``CV_ACCEL_STDS`` by
    lowest position RMSE unless ``cv_accel_std`` fixes it.
    """
    n_runs = data.x.shape[1]
    P0_pv = np.diag([sc.init_pos_std**2] * 2 + [sc.init_vel_std**2] * 2)
    x0_pv = np.tile(data.x0_nominal, (n_runs, 1))
    burn = int(BURN_IN_S / sc.dt)
    est, results = {"gps_hold": hold_last_fix(data), "dead_reckoning": dead_reckon(data.x0_nominal, data.accel_meas, sc.dt)}, {}

    best = None
    for accel_std in CV_ACCEL_STDS if cv_accel_std is None else [cv_accel_std]:
        res = run_filter(KalmanFilter(constant_velocity(sc.dt, accel_std, sc.gps_std), x0_pv, P0_pv), data.gps)
        score = rmse((data.x - res.x)[burn:, :, :2])
        if best is None or score < best[0]:
            best = (score, accel_std, res)
    est["kf_gps_only"], results["kf_gps_only"] = best[2].x, best[2]

    res = run_filter(KalmanFilter(imu_driven(sc.dt, sc.accel_noise_std, sc.gps_std), x0_pv, P0_pv), data.gps, us=data.accel_meas)
    est["fused_no_bias"], results["fused_no_bias"] = res.x, res

    P0 = np.diag([*np.diag(P0_pv), *[sc.bias_init_std**2] * 2])
    x0 = np.concatenate([x0_pv, np.zeros((n_runs, 2))], axis=1)
    model = imu_driven_with_bias(sc.dt, sc.accel_noise_std, sc.bias_walk_std, sc.gps_std)
    res = run_filter(KalmanFilter(model, x0, P0), data.gps, us=data.accel_meas)
    est["fused_bias"], results["fused_bias"] = res.x, res
    return est, results, best[1]


def pos_error_over_time(est: dict, data: FusionData) -> dict:
    """RMS position error across runs at each time step."""
    return {k: rmse(v[..., :2] - data.x[..., :2], axis=1) for k, v in est.items()}


def run(out_dir: Path, n_runs: int | None = None, seed: int = 2026) -> dict:
    n_runs = n_runs or 200
    rng = np.random.default_rng(seed)
    sc = FusionScenario()
    burn = int(BURN_IN_S / sc.dt)

    data = simulate_fusion(sc, n_runs, rng)
    est, results, cv_accel = estimate_all(sc, data)
    err_t = pos_error_over_time(est, data)
    truth6 = np.concatenate([data.x, data.bias], axis=-1)

    rows = []
    for key, (label, _) in LABELS.items():
        row = {"estimator": label, "position_rmse": r3(rmse((est[key][..., :2] - data.x[..., :2])[burn:]))}
        if est[key].shape[-1] >= 4:
            row["velocity_rmse"] = r3(rmse((est[key][..., 2:4] - data.x[..., 2:4])[burn:]))
        if key in results:
            # NEES of the [p, v] block so that all filters are compared on 4 degrees of freedom
            e = (est[key][..., :4] - data.x)[burn:]
            row["mean_nees_pv"] = r3(nees(e, results[key].P[burn:, :4, :4]).mean())
        rows.append(row)
    bias_err = (est["fused_bias"] - truth6)[..., 4:]
    bias_rmse_t = rmse(bias_err, axis=1)

    sc_out = replace(sc, outage=OUTAGE)
    data_out = simulate_fusion(sc_out, n_runs, rng)
    est_out, _, _ = estimate_all(sc_out, data_out, cv_accel)  # same tuning as the nominal run
    err_out_t = pos_error_over_time(est_out, data_out)
    in_outage = (data_out.t >= OUTAGE[0]) & (data_out.t < OUTAGE[1])
    outage_rows = [
        {
            "estimator": LABELS[k][0],
            "position_rmse_during_outage": r3(np.sqrt(np.mean(err_out_t[k][in_outage] ** 2))),
            "position_error_at_end_of_outage": r3(err_out_t[k][in_outage][-1]),
        }
        for k in LABELS
    ]

    pl.apply_style()
    figures = []

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    fix = ~np.isnan(data.gps[:, 0, 0])
    ax.scatter(data.gps[fix, 0, 0], data.gps[fix, 0, 1], s=14, color=pl.NEUTRAL, label="GPS fixes", zorder=1)
    ax.plot(data.x[:, 0, 0], data.x[:, 0, 1], color=pl.TRUTH, lw=1.5, label="Truth", zorder=2)
    ax.plot(est["dead_reckoning"][:, 0, 0], est["dead_reckoning"][:, 0, 1], color=pl.YELLOW, label=LABELS["dead_reckoning"][0] + " (leaves the frame)", zorder=3)
    ax.plot(est["fused_bias"][:, 0, 0], est["fused_bias"][:, 0, 1], color=pl.BLUE, label=LABELS["fused_bias"][0], zorder=4)
    lo, hi = data.x[:, 0, :2].min(axis=0) - 15, data.x[:, 0, :2].max(axis=0) + 15
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("One run of the GPS + IMU scenario")
    ax.legend(ncols=2, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    figures.append(pl.save(fig, out_dir, "e4_trajectory"))

    for errs, t, name, title, span in [
        (err_t, data.t, "e4_error_over_time", "Position error over time", None),
        (err_out_t, data_out.t, "e4_error_with_outage", "Position error with a 15 s GPS outage", OUTAGE),
    ]:
        fig, ax = plt.subplots(figsize=(6.8, 4.4))
        if span:
            ax.axvspan(*span, color=pl.NEUTRAL, alpha=0.25, lw=0, label="GPS outage")
        for key, (label, color) in LABELS.items():
            ax.plot(t, errs[key], color=color, lw=1.6, label=label)
        ax.set_yscale("log")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(f"RMS position error over {n_runs} runs (m)")
        ax.set_title(title)
        ax.legend(ncols=2, loc="upper center", bbox_to_anchor=(0.5, -0.16))
        figures.append(pl.save(fig, out_dir, name))

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    sigma = np.sqrt(results["fused_bias"].P[:, 4, 4])
    b_hat = est["fused_bias"][:, 0, 4]
    ax.fill_between(data.t, b_hat - 2 * sigma, b_hat + 2 * sigma, color=pl.BLUE, alpha=0.2, lw=0, label="Estimate +/- 2 sigma")
    ax.plot(data.t, b_hat, color=pl.BLUE, label="Estimated bias")
    ax.plot(data.t, data.bias[:, 0, 0], color=pl.TRUTH, lw=1.5, label="True bias")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Accelerometer bias, x axis (m/s$^2$)")
    ax.set_title("Bias estimation in one run")
    ax.legend()
    figures.append(pl.save(fig, out_dir, "e4_bias_estimate"))

    return {
        "n_runs": n_runs,
        "seed": seed,
        "scenario": {k: v for k, v in sc.__dict__.items()},
        "burn_in_s": BURN_IN_S,
        "gps_only_best_accel_std": cv_accel,
        "rows": rows,
        "bias_rmse_initial": r3(bias_rmse_t[0]),
        "bias_rmse_final": r3(bias_rmse_t[-1]),
        "outage_s": list(OUTAGE),
        "outage_rows": outage_rows,
        "figures": figures,
    }
