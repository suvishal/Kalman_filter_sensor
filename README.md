# Kalman Filter Research

A from-scratch study of Kalman filter state estimation and GPS + IMU sensor fusion, built as a small reproducible research project.

**Working title:** *Accuracy Is Not Honesty: Tuning Sensitivity, Consistency and Adaptive Noise Estimation in Kalman Filter Sensor Fusion*

Everything is implemented with NumPy only (no filtering library). SciPy is used for two closed-form solvers and chi-square quantiles, Matplotlib for figures.

## What the study found

All numbers are simulated, seed 2026, and regenerate with one command. Full tables and figures: [`docs/03-results.md`](docs/03-results.md).

| # | Question | Result |
|---|---|---|
| E1 | Does the filter beat model-free smoothing? | Position RMSE is 2.1 to 2.3 times lower than the best-tuned moving average or exponential smoother at every noise level tested, and it also estimates velocity. |
| E2 | What does mis-tuning Q and R cost? | Accuracy depends only on the ratio Q/R. Scaling both by the same factor (10⁻³ to 10³) leaves RMSE unchanged but makes the reported covariance wrong by that factor. Under-estimating Q is far worse than over-estimating it: 6.8x RMSE at Q/1000 against 2.0x at Q×1000. |
| E3 | Can a filter detect its own mis-tuning? | An overconfident filter is caught by every test. A conservative one passes the NIS test 86% of the time; the innovation autocorrelation exposes it. |
| E4 | What does fusion add? | Fusing a 100 Hz accelerometer with 1 Hz GPS gives 2.3 m RMSE against 4.2 m for the best GPS-only filter, but only if the accelerometer bias is a state. Ignoring a 0.15 m/s² bias gives 21 m, worse than GPS alone. |
| E5 | Can the filter learn R online? | A residual-based adaptive filter starting with R wrong by 100x in either direction ends within 0.3% of the oracle RMSE, and follows a 4x jump in sensor noise with a time constant of about 5 s. |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

pytest                                    # 21 tests, about 10 s
python -m kalman_research.experiments     # all experiments, about 35 s, writes results/
python -m kalman_research.experiments gps_imu_fusion --runs 50 --seed 1
```

Minimal use of the library:

```python
import numpy as np
from kalman_research import KalmanFilter, constant_velocity, run_filter
from kalman_research.simulation import simulate_linear

model = constant_velocity(dt=0.1, accel_std=0.5, meas_std=2.0)   # 2D, state [px, py, vx, vy]
rng = np.random.default_rng(0)
truth, measurements = simulate_linear(model, x0=np.array([0, 0, 5, 2]), n_steps=300, rng=rng)

kf = KalmanFilter(model, x0=np.zeros(4), P0=np.diag([4, 4, 25, 25]))
result = run_filter(kf, measurements)    # result.x, result.P, result.nu, result.S
```

## Layout

| Path | Purpose |
|---|---|
| `src/kalman_research/models.py` | State-space models (constant velocity, IMU-driven, IMU-driven with bias) |
| `src/kalman_research/filters/` | Kalman filter, adaptive-R filter, moving-average and exponential baselines |
| `src/kalman_research/simulation.py` | Truth and sensor simulators |
| `src/kalman_research/metrics.py` | RMSE, NEES, NIS, chi-square bounds, autocorrelation |
| `src/kalman_research/analysis.py` | Closed-form steady state (Riccati) and mis-tuned error covariance (Lyapunov) |
| `src/kalman_research/experiments/` | The five experiments and the CLI |
| `tests/` | Correctness tests for the filter, the closed forms and the experiments |
| `results/` | JSON summaries and figures produced by the experiments (committed) |
| `docs/` | Theory, architecture, results |
| `report/paper_draft.md` | Paper draft |
| `notes/` | Learning notebook (curriculum, assignment 0) |

## Documentation

1. [`docs/01-theory.md`](docs/01-theory.md): the maths, derived step by step, each equation linked to the code that implements it.
2. [`docs/02-architecture.md`](docs/02-architecture.md): how the code is organised and how to extend it (new filter, model, sensor, experiment, EKF).
3. [`docs/03-results.md`](docs/03-results.md): every experiment with setup, numbers, figures and interpretation.
4. [`report/paper_draft.md`](report/paper_draft.md): the paper draft and what is still needed before submitting it.

## Scope and limits

Results are from simulation with linear models and Gaussian noise. The IMU is simplified to a world-frame accelerometer, so there is no orientation and no nonlinearity. See the limitations section of the paper draft.
