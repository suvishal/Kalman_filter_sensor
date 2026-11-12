"""E3. Can a filter tell that it is mis-tuned without knowing the truth?

Three filters see the same data: one matched, one overconfident (Q too
small) and one conservative (Q too large). NEES needs ground truth; NIS and
innovation whiteness do not, so they are the tests available on a real system.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .. import plotting as pl
from ..filters import KalmanFilter, run_filter
from ..metrics import autocorrelation, chi2_mean_bounds, nees, nis, rmse
from .common import TrackingScenario, r3

FILTERS = [("Matched", 1.0, pl.BLUE), ("Overconfident (Q / 100)", 0.01, pl.ORANGE), ("Conservative (Q x 100)", 100.0, pl.AQUA)]
MAX_LAG = 20


def run(out_dir: Path, n_runs: int | None = None, seed: int = 2026) -> dict:
    n_runs = n_runs or 200
    rng = np.random.default_rng(seed)
    sc = TrackingScenario(n_steps=600)
    model = sc.model()
    xs, zs = sc.simulate(n_runs, rng)
    t = sc.dt * np.arange(1, sc.n_steps + 1)
    nees_lo, nees_hi = chi2_mean_bounds(model.n, n_runs)
    nis_lo, nis_hi = chi2_mean_bounds(model.m, n_runs)

    rows, nees_t, acfs = [], {}, {}
    for label, q_scale, _ in FILTERS:
        res = run_filter(KalmanFilter(model.with_noise(Q=q_scale * model.Q), sc.x0_batch(n_runs), sc.P0), zs)
        err = xs - res.x
        e_t = nees(err, res.P).mean(axis=1)  # average over runs at each time
        i_t = nis(res.nu, res.S).mean(axis=1)
        acf = np.mean([autocorrelation(res.nu[sc.burn_in :, i, 0], MAX_LAG) for i in range(n_runs)], axis=0)
        nees_t[label], acfs[label] = e_t, acf
        s = slice(sc.burn_in, None)
        rows.append(
            {
                "filter": label,
                "q_scale": q_scale,
                "position_rmse": r3(rmse(err[s][..., :2])),
                "mean_nees": r3(e_t[s].mean()),
                "nees_steps_inside_95pct": r3(np.mean((e_t[s] > nees_lo) & (e_t[s] < nees_hi))),
                "mean_nis": r3(i_t[s].mean()),
                "nis_steps_inside_95pct": r3(np.mean((i_t[s] > nis_lo) & (i_t[s] < nis_hi))),
                "innovation_autocorr_lag1": r3(acf[1]),
            }
        )

    pl.apply_style()
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.axhspan(nees_lo, nees_hi, color=pl.NEUTRAL, alpha=0.35, lw=0, label="95% acceptance region")
    for label, _, color in FILTERS:
        ax.plot(t, nees_t[label], color=color, lw=1.4, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"NEES averaged over {n_runs} runs (ideal: {model.n})")
    ax.set_title("Filter consistency over time")
    ax.legend(ncols=2, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    figures = [pl.save(fig, out_dir, "e3_nees_over_time")]

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    band = 2.0 / np.sqrt(sc.n_steps - sc.burn_in)
    ax.axhspan(-band, band, color=pl.NEUTRAL, alpha=0.35, lw=0, label="Single-run 95% band for white noise")
    lags = np.arange(1, MAX_LAG + 1)
    for (label, _, color), marker in zip(FILTERS, "oD^"):
        ax.plot(lags * sc.dt, acfs[label][1:], color=color, marker=marker, ms=4, lw=1.4, label=label)
    ax.set_xlabel("Lag (s)")
    ax.set_ylabel("Innovation autocorrelation (x axis)")
    ax.set_title("Innovation whiteness")
    ax.legend(ncols=2, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    figures.append(pl.save(fig, out_dir, "e3_innovation_autocorrelation"))

    return {
        "n_runs": n_runs,
        "seed": seed,
        "nees_95pct_bounds": [r3(nees_lo), r3(nees_hi)],
        "nis_95pct_bounds": [r3(nis_lo), r3(nis_hi)],
        "rows": rows,
        "figures": figures,
    }
