"""Closed-form steady-state analysis.

These give the same answers as long Monte Carlo runs, exactly and instantly,
and the tests check the two against each other.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import linalg

from .models import LinearGaussianModel


@dataclass(frozen=True)
class SteadyState:
    K: np.ndarray  # steady-state Kalman gain
    P_pred: np.ndarray  # prior covariance the filter believes
    P_post: np.ndarray  # posterior covariance the filter believes


def steady_state(model: LinearGaussianModel) -> SteadyState:
    """Steady-state gain and covariances of the Kalman filter for ``model``.

    The prior covariance is the fixed point of the Riccati recursion, i.e. the
    solution of the discrete algebraic Riccati equation (DARE).
    """
    P_pred = linalg.solve_discrete_are(model.F.T, model.H.T, model.Q, model.R)
    S = model.H @ P_pred @ model.H.T + model.R
    K = np.linalg.solve(S, model.H @ P_pred).T
    I_KH = np.eye(model.n) - K @ model.H
    P_post = I_KH @ P_pred @ I_KH.T + K @ model.R @ K.T
    return SteadyState(K=K, P_pred=P_pred, P_post=P_post)


def actual_error_covariance(true_model: LinearGaussianModel, K: np.ndarray) -> np.ndarray:
    """True steady-state posterior error covariance of a fixed-gain filter.

    For ANY gain K (optimal or not) the posterior error obeys

        e_k = (I - K H)(F e_{k-1} + w_{k-1}) - K v_k

    so its covariance is the solution of the discrete Lyapunov equation

        P = A P A^T + (I - K H) Q (I - K H)^T + K R K^T,   A = (I - K H) F

    with the TRUE Q and R. Comparing this to what the filter believes
    (``steady_state(filter_model).P_post``) quantifies the cost of mis-tuning.
    """
    I_KH = np.eye(true_model.n) - K @ true_model.H
    A = I_KH @ true_model.F
    if np.max(np.abs(np.linalg.eigvals(A))) >= 1.0:
        raise ValueError("fixed-gain error dynamics are unstable")
    W = I_KH @ true_model.Q @ I_KH.T + K @ true_model.R @ K.T
    return linalg.solve_discrete_lyapunov(A, W)
