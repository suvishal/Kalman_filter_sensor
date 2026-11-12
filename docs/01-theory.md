# Theory

Everything the code does, derived in the order the code uses it. Each section ends with a pointer to the implementation.

## 1. The problem

We want to know a quantity we cannot observe directly (the *state* $x$, here position and velocity) from measurements $z$ that are noisy and incomplete. We have two imperfect sources of information:

- a **motion model** that says how the state evolves, and
- a **sensor model** that says how measurements relate to the state.

The Kalman filter combines them, weighting each by how uncertain it is.

## 2. State-space model

$$x_k = F x_{k-1} + B u_{k-1} + w_{k-1}, \qquad w \sim \mathcal N(0, Q)$$
$$z_k = H x_k + v_k, \qquad v \sim \mathcal N(0, R)$$

| Symbol | Meaning | In the tracking scenario |
|---|---|---|
| $x$ | state | $[p_x, p_y, v_x, v_y]$ |
| $F$ | how the state moves in one step | position += velocity × dt |
| $u$, $B$ | known input and how it enters | accelerometer reading (fusion only) |
| $w$, $Q$ | what the motion model gets wrong | unknown accelerations |
| $z$, $H$ | measurement and what it sees | position only |
| $v$, $R$ | sensor noise | GPS-like position noise |

**Where Q comes from.** Assume the target has an unknown acceleration $a$ that is constant during a step and drawn from $\mathcal N(0, \sigma_a^2)$. Over one step position changes by $\tfrac12 dt^2 a$ and velocity by $dt\,a$, so $w = G a$ with $G = [\tfrac12 dt^2,\ dt]^T$ per axis, and

$$Q = \sigma_a^2\, G G^T .$$

Q is therefore not a free matrix of knobs: it has one physical parameter, the manoeuvre intensity $\sigma_a$.

Code: `models.py` (`constant_velocity`, `_kinematic_blocks`).

## 3. The filter

The filter carries a Gaussian belief: a mean $\hat x$ and a covariance $P$ ("where I think the state is, and how unsure I am").

### Predict

Push the belief through the motion model. A linear map of a Gaussian is Gaussian, and independent noise adds its covariance:

$$\hat x^- = F \hat x + B u, \qquad P^- = F P F^T + Q .$$

Uncertainty always grows in this step.

### Update

The **innovation** is the part of the measurement the prediction did not expect:

$$\nu = z - H \hat x^-, \qquad S = H P^- H^T + R .$$

$S$ is the covariance the innovation should have if the filter's model is right. We correct the prediction by a gain times the innovation, $\hat x = \hat x^- + K\nu$. For any $K$, the error after the update is $e = (I-KH)e^- - Kv$, so

$$P = (I-KH)\,P^-\,(I-KH)^T + K R K^T \quad\text{(Joseph form)}.$$

Choosing $K$ to minimise the total variance $\mathrm{tr}(P)$ (set the derivative with respect to $K$ to zero) gives the **Kalman gain**

$$K = P^- H^T S^{-1} .$$

Read it as "uncertainty of the prediction divided by total uncertainty". In one dimension with $H=1$: $K = P^-/(P^- + R)$. If the sensor is perfect ($R\to 0$), $K\to 1$ and the filter copies the measurement. If the prediction is perfect ($P^-\to 0$), $K\to 0$ and the measurement is ignored.

With the optimal gain the Joseph form simplifies to $P = (I-KH)P^-$. The code keeps the Joseph form because it stays symmetric and positive semi-definite under round-off, and because it is still correct when the gain is *not* optimal, which is the whole subject of the mis-tuning experiments.

**Why this is optimal.** For linear models with Gaussian noise the true posterior is exactly Gaussian and the filter computes it exactly, so it is the Bayes-optimal estimator. If the noise is not Gaussian, it is still the best *linear* estimator.

Code: `filters/kalman.py` (`KalmanFilter.predict`, `KalmanFilter.update`). Test: `test_update_matches_textbook_form`, `test_static_scalar_matches_running_mean`.

## 4. Two different questions: accuracy and consistency

- **Accuracy**: how far is $\hat x$ from the truth? Measured by RMSE.
- **Consistency**: does $P$ honestly describe that error? A filter that is 5 m off while reporting 0.5 m uncertainty is dangerous to anything downstream that trusts $P$ (gating, fusion with other estimators, planning).

**NEES** (normalised estimation error squared), needs ground truth:

$$\epsilon_k = e_k^T P_k^{-1} e_k \sim \chi^2_n \quad\Rightarrow\quad \mathbb E[\epsilon_k] = n .$$

NEES above $n$ means overconfident; below $n$ means conservative.

**NIS** (normalised innovation squared), needs no ground truth:

$$\epsilon^\nu_k = \nu_k^T S_k^{-1} \nu_k \sim \chi^2_m .$$

**Whiteness.** The innovations of an optimal filter are uncorrelated in time. If they are correlated, there is predictable information the filter is not using.

Averaging over $N$ independent Monte Carlo runs gives a tight acceptance interval, because the sum of $N$ independent $\chi^2_n$ variables is $\chi^2_{Nn}$. Errors are correlated *in time*, so NEES may only be pooled across runs, not across time steps.

Code: `metrics.py`. Tests: `test_matched_filter_is_consistent`.

## 5. Steady state and the cost of mis-tuning, in closed form

For a time-invariant model the covariance recursion does not depend on the data, and it converges to a fixed point: the solution of the discrete algebraic Riccati equation (DARE)

$$P^- = F\left(P^- - P^- H^T (H P^- H^T + R)^{-1} H P^-\right)F^T + Q .$$

Now suppose the filter is designed with wrong values $Q_f, R_f$. It computes a gain $K_f$ and *believes* a covariance $P_f$. The error it actually makes follows $e_k = (I-K_fH)(F e_{k-1} + w) - K_f v$ with the **true** noises, so its true covariance $\Sigma$ solves the discrete Lyapunov equation

$$\Sigma = A \Sigma A^T + (I-K_fH)\,Q\,(I-K_fH)^T + K_f R K_f^T, \qquad A = (I-K_fH)F .$$

This gives the exact RMSE ($\sqrt{\mathrm{tr}\,\Sigma_{pos}}$) and the exact expected NEES ($\mathrm{tr}(P_f^{-1}\Sigma)$) of any mis-tuned filter without running a simulation.

**The ratio property.** Multiply $Q_f$ and $R_f$ by the same constant $c$. The DARE solution scales by $c$, and the gain $K = P^-H^T(HP^-H^T+R)^{-1}$ is unchanged because $c$ cancels. So:

- the estimates, and therefore the RMSE, depend only on the *ratio* of Q to R;
- the reported covariance scales by $c$, so NEES scales by $1/c$.

Accuracy cannot tell you the absolute scale of your noise model. Only a consistency test can.

Code: `analysis.py`. Tests: `test_lyapunov_predicts_error_of_mistuned_filter`, `test_gain_depends_only_on_q_to_r_ratio`, `test_optimal_gain_minimises_actual_error`.

## 6. Learning R online

After an update, define the post-fit residual $\varepsilon = z - H\hat x$. Since $\hat x = \hat x^- + K\nu$, we have $\varepsilon = (I - HK)\nu$. With the optimal gain, $I - HK = R S^{-1}$, so

$$\mathbb E[\varepsilon\varepsilon^T] = R S^{-1} R .$$

Writing $A = HP^-H^T$ (so $S = A + R$), the posterior covariance seen by the sensor is $HPH^T = A - AS^{-1}A = AS^{-1}R$, and $R - AS^{-1}R = RS^{-1}R$. Therefore

$$R = \mathbb E[\varepsilon\varepsilon^T] + H P H^T .$$

The adaptive filter replaces the expectation with an exponentially weighted running average:

$$\hat R_k = \lambda \hat R_{k-1} + (1-\lambda)\left(\varepsilon_k\varepsilon_k^T + H P_k H^T\right).$$

Notes:

- Both terms are positive semi-definite, so $\hat R$ is always a valid covariance. The innovation-based alternative $\mathbb E[\nu\nu^T] - HP^-H^T$ can go negative.
- The identity above assumes the gain is already optimal. While $\hat R$ is wrong the update is a fixed-point iteration, not an exact estimator. It converges in every case tested here (E5), but that is an empirical result, not a proof.
- $\lambda$ trades noise in $\hat R$ against speed of reaction; memory is about $1/(1-\lambda)$ updates.
- Only R is adapted. Estimating Q and R together from one innovation sequence is poorly conditioned, and by section 5 the innovations mostly constrain their ratio.

This family of estimators goes back to Mehra (1970) and, in the residual-based form used for INS/GPS, Mohamed and Schwarz (1999).

Code: `filters/adaptive.py`. Test: `test_adaptive_filter_recovers_true_R`.

## 7. GPS + IMU fusion

The two sensors fail in opposite ways:

| | GPS | Accelerometer |
|---|---|---|
| Rate | slow (1 Hz) | fast (100 Hz) |
| Error | large, but does not accumulate | small per sample, but integrates into unbounded drift |

**Model.** Use the accelerometer reading as the control input $u$ of the kinematic model, and GPS as the measurement. The accelerometer's white noise enters through the same $G$ as the input, so $Q = \sigma_{acc}^2 GG^T$. The filter predicts at 100 Hz and updates whenever a GPS fix arrives; a step without a fix is predict-only.

**Bias.** A real accelerometer reads $u = a + b + n$ with a slowly varying bias $b$. A constant 0.15 m/s² bias integrates to 270 m of position error in 60 s. The fix is to make the bias a state:

$$x = [p, v, b], \qquad p_k = p_{k-1} + v\,dt + \tfrac12 dt^2 (u - b), \quad v_k = v_{k-1} + dt\,(u - b), \quad b_k = b_{k-1} + w_b .$$

GPS never measures $b$ directly. The filter infers it because a bias produces a position error that grows in a recognisable way, and the off-diagonal terms of $P$ carry that relationship from the position innovation back to the bias state.

**Simplification.** The accelerometer here measures in the world frame. A real IMU measures in the body frame, so orientation must be estimated too and the model becomes nonlinear (extended Kalman filter). See `docs/02-architecture.md` for where that extension fits.

Code: `models.py` (`imu_driven`, `imu_driven_with_bias`), `simulation.py` (`simulate_fusion`). Test: `test_bias_filter_is_consistent_and_beats_dead_reckoning`.

## References

Check each entry against the original before citing it in a submission.

- R. E. Kalman, "A New Approach to Linear Filtering and Prediction Problems", *Journal of Basic Engineering*, 82(1), 1960.
- R. K. Mehra, "On the Identification of Variances and Adaptive Kalman Filtering", *IEEE Transactions on Automatic Control*, 15(2), 1970.
- A. H. Mohamed and K. P. Schwarz, "Adaptive Kalman Filtering for INS/GPS", *Journal of Geodesy*, 73, 1999.
- Y. Bar-Shalom, X. R. Li, T. Kirubarajan, *Estimation with Applications to Tracking and Navigation*, Wiley, 2001. (Consistency tests, kinematic models.)
- D. Simon, *Optimal State Estimation*, Wiley, 2006.
- S. Thrun, W. Burgard, D. Fox, *Probabilistic Robotics*, MIT Press, 2005.
- M. S. Grewal and A. P. Andrews, *Kalman Filtering: Theory and Practice*, Wiley.
