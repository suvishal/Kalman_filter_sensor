"""E2. What does mis-tuning Q and R actually cost?

The filter is given ``Q_f = a*Q`` and ``R_f = b*R`` while the world uses Q and
R. For every (a, b) the true steady-state error covariance is computed in
closed form (Lyapunov equation) and compared with the covariance the filter
believes. A Monte Carlo run on a few grid points checks the closed form.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .. import plotting as pl
from ..analysis import actual_error_covariance, steady_state
from ..filters import KalmanFilter, run_filter
from ..metrics import nees, rmse
from .common import TrackingScenario, r3

SCALES = np.logspace(-3, 3, 13)
MC_POINTS = [(1e-3, 1.0), (1e-2, 1.0), (1.0, 1.0), (1e2, 1.0), (1e3, 1.0), (1e-2, 1e-2), (1e2, 1e2)]


def analytic(true_model, a: float, b: float) -> tuple[float, float]:
    """(position RMSE, expected NEES) at steady state for scales (a, b)."""
    belief = steady_state(true_model.with_noise(Q=a * true_model.Q, R=b * true_model.R))
    actual = actual_error_covariance(true_model, belief.K)
    # E[e^T P^-1 e] = trace(P^-1 Sigma) when e has covariance Sigma
    return float(np.sqrt(np.trace(actual[:2, :2]))), float(np.trace(np.linalg.solve(belief.P_post, actual)))


def run(out_dir: Path, n_runs: int | None = None, seed: int = 2026) -> dict:
    n_runs = n_runs or 400
    rng = np.random.default_rng(seed)
    sc = TrackingScenario(n_steps=1500, burn_in=1000)
    model = sc.model()
    rmse_opt, _ = analytic(model, 1.0, 1.0)

    rmse_grid = np.empty((len(SCALES), len(SCALES)))
    nees_grid = np.empty_like(rmse_grid)
    for i, b in enumerate(SCALES):  # rows: R scale
        for j, a in enumerate(SCALES):  # columns: Q scale
            rmse_grid[i, j], nees_grid[i, j] = analytic(model, a, b)

    xs, zs = sc.simulate(n_runs, rng)
    mc = []
    for a, b in MC_POINTS:
        f_model = model.with_noise(Q=a * model.Q, R=b * model.R)
        res = run_filter(KalmanFilter(f_model, sc.x0_batch(n_runs), sc.P0), zs)
        err = (xs - res.x)[sc.burn_in :]
        th_rmse, th_nees = analytic(model, a, b)
        mc.append(
            {
                "q_scale": a,
                "r_scale": b,
                "rmse_theory": r3(th_rmse),
                "rmse_monte_carlo": r3(rmse(err[..., :2])),
                "nees_theory": r3(th_nees),
                "nees_monte_carlo": r3(nees(err, res.P[sc.burn_in :]).mean()),
            }
        )

    ratio_curve = [analytic(model, a, 1.0) for a in SCALES]

    pl.apply_style()
    ticks = np.arange(0, len(SCALES), 2)
    tick_labels = [f"$10^{{{int(np.log10(s))}}}$" for s in SCALES[ticks]]
    figures = []
    panels = [
        (np.log10(rmse_grid / rmse_opt), pl.SEQUENTIAL, "log10(RMSE / optimal RMSE)", "Accuracy: position RMSE relative to the optimal filter", "e2_rmse_heatmap", None),
        (np.log10(nees_grid / model.n), pl.DIVERGING, "log10(NEES / n)", "Honesty: NEES relative to its ideal value n", "e2_nees_heatmap", 3.0),
    ]
    for grid, cmap, cbar_label, title, name, vlim in panels:
        fig, ax = plt.subplots(figsize=(5.6, 4.6))
        kwargs = {"vmin": -vlim, "vmax": vlim} if vlim else {"vmin": 0.0}
        im = ax.imshow(grid, origin="lower", cmap=cmap, **kwargs)
        ax.grid(False)
        ax.set_xticks(ticks, tick_labels)
        ax.set_yticks(ticks, tick_labels)
        ax.set_xlabel("Assumed Q / true Q")
        ax.set_ylabel("Assumed R / true R")
        ax.set_title(title)
        ax.plot(len(SCALES) // 2, len(SCALES) // 2, marker="+", color=pl.TEXT, ms=12, mew=2)
        fig.colorbar(im, ax=ax, label=cbar_label)
        figures.append(pl.save(fig, out_dir, name))

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(SCALES, [v[0] / rmse_opt for v in ratio_curve], color=pl.BLUE, label="Closed form (Lyapunov)")
    slice_pts = [p for p in mc if p["r_scale"] == 1.0]
    ax.scatter([p["q_scale"] for p in slice_pts], [p["rmse_monte_carlo"] / rmse_opt for p in slice_pts],
               color=pl.ORANGE, zorder=3, s=45, edgecolor=pl.SURFACE, linewidth=1.5, label="Monte Carlo")
    ax.set_xscale("log")
    ax.set_xlabel("Assumed Q/R ratio relative to the true ratio")
    ax.set_ylabel("RMSE / optimal RMSE")
    ax.set_title("Cost of a wrong Q/R ratio")
    ax.legend()
    figures.append(pl.save(fig, out_dir, "e2_ratio_slice"))

    diag = [r3(rmse_grid[i, i] / rmse_opt) for i in range(len(SCALES))]
    return {
        "n_runs": n_runs,
        "seed": seed,
        "optimal_rmse": r3(rmse_opt),
        "scales": [float(s) for s in SCALES],
        "rmse_inflation_vs_q_scale_at_true_r": [r3(v[0] / rmse_opt) for v in ratio_curve],
        "rmse_inflation_on_diagonal": diag,
        "nees_over_n_on_diagonal": [r3(nees_grid[i, i] / model.n) for i in range(len(SCALES))],
        "monte_carlo_check": mc,
        "figures": figures,
    }
