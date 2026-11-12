import numpy as np
import pytest

from kalman_research import KalmanFilter, constant_velocity, run_filter
from kalman_research.analysis import actual_error_covariance, steady_state
from kalman_research.metrics import chi2_mean_bounds, nees, nis
from kalman_research.models import LinearGaussianModel
from kalman_research.simulation import simulate_linear


def scalar_model(q=0.0, r=4.0):
    one = np.eye(1)
    return LinearGaussianModel(F=one, H=one, Q=q * one, R=r * one)


def test_static_scalar_matches_running_mean():
    # Estimating a constant with a diffuse prior is just averaging: after k
    # measurements the estimate is their mean and the variance is r / k.
    rng = np.random.default_rng(0)
    zs = 10.0 + 2.0 * rng.standard_normal((50, 1))
    kf = KalmanFilter(scalar_model(), x0=np.zeros(1), P0=1e12 * np.eye(1))
    res = run_filter(kf, zs)
    assert res.x[-1, 0] == pytest.approx(zs.mean(), rel=1e-6)
    assert res.P[-1, 0, 0] == pytest.approx(4.0 / 50, rel=1e-6)


def test_update_matches_textbook_form():
    model = constant_velocity(dt=0.1, accel_std=0.5, meas_std=2.0)
    P0 = np.diag([4.0, 4.0, 25.0, 25.0])
    kf = KalmanFilter(model, x0=np.zeros(4), P0=P0)
    kf.predict()
    P_pred = kf.P.copy()
    z = np.array([1.0, -2.0])
    innov = kf.update(z)
    S = model.H @ P_pred @ model.H.T + model.R
    K = P_pred @ model.H.T @ np.linalg.inv(S)
    np.testing.assert_allclose(innov.K, K, atol=1e-12)
    np.testing.assert_allclose(kf.P, (np.eye(4) - K @ model.H) @ P_pred, atol=1e-12)
    np.testing.assert_allclose(kf.x, K @ z, atol=1e-12)


def test_covariance_stays_symmetric_positive_definite():
    model = constant_velocity(dt=0.1, accel_std=0.5, meas_std=2.0)
    res = run_filter(KalmanFilter(model, np.zeros(4), 1e6 * np.eye(4)), np.zeros((500, 2)))
    np.testing.assert_allclose(res.P, np.swapaxes(res.P, 1, 2), atol=1e-9)
    assert np.all(np.linalg.eigvalsh(res.P) > 0)


def test_missing_measurement_is_predict_only():
    model = constant_velocity(dt=0.1, accel_std=0.5, meas_std=2.0)
    kf = KalmanFilter(model, np.array([0.0, 0.0, 1.0, 2.0]), np.eye(4))
    res = run_filter(kf, np.full((3, 2), np.nan))
    np.testing.assert_allclose(res.x[-1], [0.3, 0.6, 1.0, 2.0])
    assert np.isnan(res.nu).all() and np.isnan(res.S).all()
    assert np.trace(res.P[-1]) > np.trace(res.P[0])


def test_converges_to_riccati_steady_state():
    model = constant_velocity(dt=0.1, accel_std=0.5, meas_std=2.0)
    res = run_filter(KalmanFilter(model, np.zeros(4), 100 * np.eye(4)), np.zeros((2000, 2)))
    np.testing.assert_allclose(res.P[-1], steady_state(model).P_post, rtol=1e-6, atol=1e-9)


def test_batched_run_equals_single_runs():
    model = constant_velocity(dt=0.1, accel_std=0.5, meas_std=2.0)
    rng = np.random.default_rng(1)
    zs = rng.standard_normal((40, 3, 2))
    batched = run_filter(KalmanFilter(model, np.zeros((3, 4)), np.eye(4)), zs)
    for i in range(3):
        single = run_filter(KalmanFilter(model, np.zeros(4), np.eye(4)), zs[:, i])
        np.testing.assert_allclose(batched.x[:, i], single.x, atol=1e-12)
        np.testing.assert_allclose(batched.P, single.P, atol=1e-12)


def monte_carlo(true_model, filter_model, n_runs=400, n_steps=400, seed=2):
    rng = np.random.default_rng(seed)
    P0 = np.diag([4.0, 4.0, 25.0, 25.0])
    x0 = rng.multivariate_normal(np.zeros(4), P0, size=n_runs)
    xs, zs = simulate_linear(true_model, x0, n_steps, rng)
    res = run_filter(KalmanFilter(filter_model, np.zeros((n_runs, 4)), P0), zs)
    return xs - res.x, res


def test_matched_filter_is_consistent():
    model = constant_velocity(dt=0.1, accel_std=0.5, meas_std=2.0)
    err, res = monte_carlo(model, model)
    # Errors are correlated in time, so NEES is only independent ACROSS runs:
    # test it at individual time steps.
    lo, hi = chi2_mean_bounds(model.n, err.shape[1], 0.999)
    for k in (50, 200, 399):
        assert lo < nees(err, res.P)[k].mean() < hi
    # Innovations are white, so NIS can be pooled over time as well.
    lo, hi = chi2_mean_bounds(model.m, err.shape[0] * err.shape[1], 0.999)
    assert lo < nis(res.nu, res.S).mean() < hi


def test_lyapunov_predicts_error_of_mistuned_filter():
    true_model = constant_velocity(dt=0.1, accel_std=0.5, meas_std=2.0)
    filter_model = true_model.with_noise(Q=100 * true_model.Q)
    err, _ = monte_carlo(true_model, filter_model, n_runs=2000)
    empirical = np.einsum("tni,tnj->ij", err[200:], err[200:]) / (err[200:].shape[0] * err.shape[1])
    predicted = actual_error_covariance(true_model, steady_state(filter_model).K)
    np.testing.assert_allclose(np.diag(empirical), np.diag(predicted), rtol=0.05)


def test_optimal_gain_minimises_actual_error():
    model = constant_velocity(dt=0.1, accel_std=0.5, meas_std=2.0)
    best = steady_state(model)
    np.testing.assert_allclose(actual_error_covariance(model, best.K), best.P_post, atol=1e-9)
    for scale in (0.01, 0.1, 10.0, 100.0):
        K = steady_state(model.with_noise(Q=scale * model.Q)).K
        assert np.trace(actual_error_covariance(model, K)) > np.trace(best.P_post)


def test_gain_depends_only_on_q_to_r_ratio():
    model = constant_velocity(dt=0.1, accel_std=0.5, meas_std=2.0)
    scaled = model.with_noise(Q=7.0 * model.Q, R=7.0 * model.R)
    np.testing.assert_allclose(steady_state(scaled).K, steady_state(model).K, atol=1e-9)
    np.testing.assert_allclose(steady_state(scaled).P_post, 7.0 * steady_state(model).P_post, rtol=1e-8)
