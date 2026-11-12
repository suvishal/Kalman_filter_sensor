"""E5. Can the filter learn R by itself?

Part A, wrong but constant R: the filter starts with R off by a factor of
0.01 to 100. Part B, changing R: the sensor degrades from 2 m to 8 m noise in
the middle third of the run. In both parts the adaptive filter is compared
with a fixed filter given the same wrong R and with an oracle that knows the
true R at every step.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .. import plotting as pl
from ..filters import AdaptiveKalmanFilter, KalmanFilter, run_filter
from ..metrics import nees, rmse
from .common import TrackingScenario, r3

R_SCALES = [0.01, 0.1, 1.0, 10.0, 100.0]
FORGETTING = 0.98
DEGRADED_STD = 8.0


def run_adaptive(model, sc, zs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the adaptive filter on each trajectory. Returns (x, P, R_hat[..., 0, 0])."""
    xs, Ps, Rs = [], [], []
    for i in range(zs.shape[1]):
        res = run_filter(AdaptiveKalmanFilter(model, sc.x0_mean, sc.P0, FORGETTING), zs[:, i])
        xs.append(res.x)
        Ps.append(res.P)
        Rs.append(res.extras["R_hat"][:, 0, 0])
    return np.stack(xs, axis=1), np.stack(Ps, axis=1), np.stack(Rs, axis=1)


def nees_per_run(err: np.ndarray, P: np.ndarray) -> np.ndarray:
    """NEES when every run has its own covariance history, P of shape (T, N, n, n)."""
    return np.einsum("tni,tnij,tnj->tn", err, np.linalg.inv(P), err)


def run(out_dir: Path, n_runs: int | None = None, seed: int = 2026) -> dict:
    n_runs = n_runs or 100
    rng = np.random.default_rng(seed)
    sc = TrackingScenario(n_steps=900, burn_in=300)
    model = sc.model()
    s = slice(sc.burn_in, None)

    # Part A: constant true R, wrong initial guess
    xs, zs = sc.simulate(n_runs, rng)
    oracle = run_filter(KalmanFilter(model, sc.x0_batch(n_runs), sc.P0), zs)
    oracle_rmse = rmse((xs - oracle.x)[s][..., :2])
    static_rows = []
    for scale in R_SCALES:
        wrong = model.with_noise(R=scale * model.R)
        fixed = run_filter(KalmanFilter(wrong, sc.x0_batch(n_runs), sc.P0), zs)
        ax_, aP, aR = run_adaptive(wrong, sc, zs)
        static_rows.append(
            {
                "r_scale": scale,
                "fixed_rmse": r3(rmse((xs - fixed.x)[s][..., :2])),
                "adaptive_rmse": r3(rmse((xs - ax_)[s][..., :2])),
                "oracle_rmse": r3(oracle_rmse),
                "fixed_nees": r3(nees((xs - fixed.x)[s], fixed.P[s]).mean()),
                "adaptive_nees": r3(nees_per_run((xs - ax_)[s], aP[s]).mean()),
                "adaptive_final_meas_std": r3(np.sqrt(aR[-200:].mean())),
                "true_meas_std": sc.meas_std,
            }
        )

    # Part B: the sensor degrades in the middle third
    T = sc.n_steps
    t = sc.dt * np.arange(1, T + 1)
    true_std = np.full(T, sc.meas_std)
    true_std[T // 3 : 2 * T // 3] = DEGRADED_STD
    Rs = true_std[:, None, None] ** 2 * np.eye(2)
    xs, zs = sc.simulate(n_runs, rng, Rs=Rs)
    fixed = run_filter(KalmanFilter(model, sc.x0_batch(n_runs), sc.P0), zs)
    oracle = run_filter(KalmanFilter(model, sc.x0_batch(n_runs), sc.P0), zs, Rs=Rs)
    ax_, aP, aR = run_adaptive(model, sc, zs)
    errs = {"Fixed nominal R": xs - fixed.x, "Adaptive R": xs - ax_, "Oracle R": xs - oracle.x}
    neess = {
        "Fixed nominal R": nees(errs["Fixed nominal R"], fixed.P).mean(axis=1),
        "Adaptive R": nees_per_run(errs["Adaptive R"], aP).mean(axis=1),
        "Oracle R": nees(errs["Oracle R"], oracle.P).mean(axis=1),
    }
    # skip 100 steps after each change so segments describe settled behaviour
    segments = {"before": slice(100, T // 3), "degraded": slice(T // 3 + 100, 2 * T // 3), "after": slice(2 * T // 3 + 100, T)}
    switch_rows = [
        {
            "filter": name,
            **{f"rmse_{seg}": r3(rmse(err[sl][..., :2])) for seg, sl in segments.items()},
            **{f"nees_{seg}": r3(neess[name][sl].mean()) for seg, sl in segments.items()},
        }
        for name, err in errs.items()
    ]

    pl.apply_style()
    colors = {"Fixed nominal R": pl.ORANGE, "Adaptive R": pl.BLUE, "Oracle R": pl.AQUA}
    figures = []

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(R_SCALES, [r["fixed_rmse"] for r in static_rows], color=pl.ORANGE, marker="s", label="Fixed wrong R")
    ax.plot(R_SCALES, [r["adaptive_rmse"] for r in static_rows], color=pl.BLUE, marker="D", label="Adaptive R")
    ax.axhline(oracle_rmse, color=pl.AQUA, lw=1.5, ls="--", label="Oracle (true R)")
    ax.set_xscale("log")
    ax.set_xlabel("Initial R / true R")
    ax.set_ylabel("Position RMSE (m)")
    ax.set_title("Recovering from a wrong measurement noise guess")
    ax.legend()
    figures.append(pl.save(fig, out_dir, "e5_static_misspecification"))

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    est_std = np.sqrt(aR)
    ax.fill_between(t, np.percentile(est_std, 10, axis=1), np.percentile(est_std, 90, axis=1), color=pl.BLUE, alpha=0.2, lw=0, label="10th to 90th percentile of runs")
    ax.plot(t, np.sqrt(aR.mean(axis=1)), color=pl.BLUE, label="Estimated (mean over runs)")
    ax.plot(t, true_std, color=pl.TRUTH, lw=1.5, label="True")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Measurement noise standard deviation (m)")
    ax.set_title("Tracking a change in sensor noise")
    ax.legend(ncols=2, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    figures.append(pl.save(fig, out_dir, "e5_noise_tracking"))

    for values, ylabel, title, name, log in [
        ({k: rmse(v[..., :2], axis=1) for k, v in errs.items()}, f"RMS position error over {n_runs} runs (m)", "Position error while the sensor degrades", "e5_error_over_time", False),
        (neess, f"NEES averaged over {n_runs} runs (ideal: {model.n})", "Consistency while the sensor degrades", "e5_nees_over_time", True),
    ]:
        fig, ax = plt.subplots(figsize=(6.8, 4.2))
        ax.axvspan(t[T // 3], t[2 * T // 3 - 1], color=pl.NEUTRAL, alpha=0.25, lw=0, label="Sensor degraded")
        for name_, series in values.items():
            ax.plot(t, series, color=colors[name_], lw=1.5, label=name_)
        if log:
            ax.set_yscale("log")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(ncols=2, loc="upper center", bbox_to_anchor=(0.5, -0.16))
        figures.append(pl.save(fig, out_dir, name))

    return {
        "n_runs": n_runs,
        "seed": seed,
        "forgetting": FORGETTING,
        "static_rows": static_rows,
        "degraded_meas_std": DEGRADED_STD,
        "switch_rows": switch_rows,
        "figures": figures,
    }
