# Architecture

## Design rules

1. **Models are data, filters are algorithms.** A `LinearGaussianModel` holds only matrices. A filter takes a model. The *true* model used to simulate and the *filter's* model are separate objects, which is what makes mis-tuning experiments one line: `model.with_noise(Q=100 * model.Q)`.
2. **No global state, explicit randomness.** Every simulator takes a `numpy.random.Generator`. The same seed gives the same numbers.
3. **One state ordering.** `[positions..., velocities..., biases...]` in every model, so "the first two entries are position" is always true.
4. **Missing measurements are NaN.** A slower sensor or an outage is a row of NaN, and the runner turns it into a predict-only step. No special multi-rate code.
5. **Experiments only orchestrate.** They call the library, write a JSON summary and figures, and contain no estimation maths.

## Data flow

```
models.py ──────────────┐
  LinearGaussianModel    │
                         ▼
simulation.py ──► truth xs, measurements zs ──► filters/ ──► FilterResult(x, P, nu, S)
                         │                                          │
                         └──────────────► metrics.py ◄──────────────┘
                                           RMSE, NEES, NIS
analysis.py: closed-form steady state, used to cross-check Monte Carlo
experiments/: one module per research question, registered in experiments/__init__.py
```

## Modules

| Module | Contents |
|---|---|
| `models.py` | `LinearGaussianModel`; builders `constant_velocity`, `imu_driven`, `imu_driven_with_bias` |
| `filters/kalman.py` | `KalmanFilter` (`predict`, `update`), `run_filter`, `FilterResult` |
| `filters/adaptive.py` | `AdaptiveKalmanFilter`, overrides `update` to estimate R |
| `filters/baselines.py` | `moving_average`, `exponential_smoothing` |
| `simulation.py` | `simulate_linear`, `FusionScenario`, `simulate_fusion`, `dead_reckon` |
| `metrics.py` | `rmse`, `nees`, `nis`, `chi2_mean_bounds`, `autocorrelation` |
| `analysis.py` | `steady_state` (Riccati), `actual_error_covariance` (Lyapunov) |
| `plotting.py` | one figure style and colour assignment for all plots |
| `experiments/` | `baselines`, `tuning_sensitivity`, `consistency`, `gps_imu_fusion`, `adaptive_noise`, CLI |

## Array shapes

| Quantity | Shape |
|---|---|
| state history | `(T, n)` or batched `(T, N, n)` |
| covariance history | `(T, n, n)` |
| measurements | `(T, m)` or `(T, N, m)` |

**Batching.** For a linear Kalman filter the covariance and gain do not depend on the data. `KalmanFilter` therefore accepts a batch of state means `(N, n)` sharing one covariance, and N Monte Carlo runs cost one covariance recursion. This is why 200 runs of 6000 steps take about a second. The adaptive filter's covariance does depend on the data, so it runs one trajectory at a time and rejects batched input.

## How to extend

**A new experiment.** Add `experiments/my_question.py` with `run(out_dir, n_runs, seed) -> dict` returning at least `n_runs` and `figures`, and add it to `REGISTRY` in `experiments/__init__.py`. The CLI and the smoke test in `tests/test_experiments.py` pick it up.

**A new linear model** (constant acceleration, 3D, another sensor). Add a builder function in `models.py` that returns a `LinearGaussianModel`. Nothing else changes.

**A second sensor with a different H** (for example a velocity sensor). `KalmanFilter.update` currently uses the model's single `H`. Add optional `H` and `R` arguments to `update` and call it once per sensor at each step: sequential updates with independent sensors are equivalent to one stacked update.

**An extended Kalman filter** (body-frame IMU with heading, range/bearing sensors). Subclass `KalmanFilter`, and in `predict`/`update` replace `F x` and `H x` with nonlinear functions `f(x, u)` and `h(x)` while using their Jacobians in the covariance equations. `run_filter`, `FilterResult` and all of `metrics.py` work unchanged. Note the batching shortcut no longer applies, because the Jacobians depend on the state.

**Real data.** Write a loader that returns the same arrays `simulate_fusion` returns (`t`, `accel_meas`, `gps` with NaN between fixes, and reference positions as truth). The filters and metrics need no changes. NIS and innovation whiteness still work without ground truth.

**Adaptive Q, smoothers, other estimators.** Follow `filters/adaptive.py`: subclass, override one method, expose extra per-step quantities through `diagnostics()` so `run_filter` logs them into `FilterResult.extras`.

## Testing strategy

- **Against known answers**: a static scalar reduces to the running mean; one update matches the textbook formula.
- **Against closed forms**: the recursion converges to the Riccati solution; Monte Carlo error of a mis-tuned filter matches the Lyapunov solution.
- **Statistical**: a matched filter passes NEES and NIS chi-square tests; the six-state fusion filter is consistent.
- **Smoke**: every registered experiment runs end to end with three Monte Carlo runs.
