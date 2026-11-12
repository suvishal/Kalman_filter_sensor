import numpy as np
import pytest

from kalman_research import AdaptiveKalmanFilter, KalmanFilter, constant_velocity, imu_driven_with_bias, run_filter
from kalman_research.filters import exponential_smoothing, moving_average
from kalman_research.metrics import autocorrelation, chi2_mean_bounds, nees, rmse
from kalman_research.simulation import FusionScenario, dead_reckon, simulate_fusion, simulate_linear


def test_moving_average_and_exponential_smoothing():
    zs = np.arange(6, dtype=float).reshape(6, 1)
    np.testing.assert_allclose(moving_average(zs, 3)[:, 0], [0, 0.5, 1, 2, 3, 4])
    np.testing.assert_allclose(moving_average(zs, 1), zs)
    np.testing.assert_allclose(exponential_smoothing(zs, 1.0), zs)
    np.testing.assert_allclose(exponential_smoothing(zs, 0.5)[:3, 0], [0, 0.5, 1.25])


def test_rmse_and_autocorrelation():
    assert rmse(np.array([[3.0, 4.0], [0.0, 0.0]])) == pytest.approx(np.sqrt(12.5))
    white = np.random.default_rng(0).standard_normal(5000)
    acf = autocorrelation(white, 5)
    assert acf[0] == pytest.approx(1.0)
    assert np.all(np.abs(acf[1:]) < 0.05)


def test_adaptive_filter_recovers_true_R():
    true_model = constant_velocity(dt=0.1, accel_std=0.5, meas_std=3.0)
    rng = np.random.default_rng(3)
    P0 = np.diag([9.0, 9.0, 25.0, 25.0])
    _, zs = simulate_linear(true_model, rng.multivariate_normal(np.zeros(4), P0), 5000, rng)
    for wrong in (0.01, 100.0):
        kf = AdaptiveKalmanFilter(true_model.with_noise(R=wrong * true_model.R), np.zeros(4), P0, forgetting=0.995)
        res = run_filter(kf, zs)
        R_avg = res.extras["R_hat"][2000:].mean(axis=0)
        np.testing.assert_allclose(np.diag(R_avg), [9.0, 9.0], rtol=0.15)


def test_adaptive_filter_rejects_batches_and_fixed_R():
    model = constant_velocity(dt=0.1, accel_std=0.5, meas_std=3.0)
    with pytest.raises(ValueError):
        AdaptiveKalmanFilter(model, np.zeros((2, 4)), np.eye(4))
    kf = AdaptiveKalmanFilter(model, np.zeros(4), np.eye(4))
    kf.predict()
    with pytest.raises(ValueError):
        kf.update(np.zeros(2), R=np.eye(2))


def fusion_run(sc, n_runs=200, seed=4):
    data = simulate_fusion(sc, n_runs, np.random.default_rng(seed))
    model = imu_driven_with_bias(sc.dt, sc.accel_noise_std, sc.bias_walk_std, sc.gps_std)
    x0 = np.concatenate([data.x0_nominal, np.zeros(2)])
    P0 = np.diag([sc.init_pos_std**2] * 2 + [sc.init_vel_std**2] * 2 + [sc.bias_init_std**2] * 2)
    res = run_filter(KalmanFilter(model, np.tile(x0, (n_runs, 1)), P0), data.gps, us=data.accel_meas)
    truth = np.concatenate([data.x, data.bias], axis=-1)
    return data, res, truth


def test_fusion_gps_schedule_and_outage():
    sc = FusionScenario(duration=10.0, outage=(4.0, 7.0))
    data = simulate_fusion(sc, 2, np.random.default_rng(0))
    has_fix = ~np.isnan(data.gps[:, 0, 0])
    np.testing.assert_allclose(data.t[has_fix], [1, 2, 3, 7, 8, 9, 10])


def test_bias_filter_is_consistent_and_beats_dead_reckoning():
    sc = FusionScenario(duration=30.0)
    data, res, truth = fusion_run(sc)
    err = truth - res.x
    lo, hi = chi2_mean_bounds(6, err.shape[1], 0.999)
    # NEES at a single time step is independent across runs
    for k in (500, 1500, 2999):
        assert lo < nees(err, res.P)[k].mean() < hi
    dr = dead_reckon(data.x0_nominal, data.accel_meas, sc.dt)
    assert rmse(err[-500:, :, :2]) < 0.25 * rmse((data.x - dr)[-500:, :, :2])
