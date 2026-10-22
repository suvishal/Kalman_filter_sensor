# Roadmap

Goal: understand state estimation well enough to derive the Kalman filter, defend every equation, and write it up.

## Status
- [x] Repo scaffolded
- [ ] Assignment 0 answered (`notes/01-assignment-0.md`)
- [ ] Module 1: linear algebra for estimation
- [ ] Module 2: probability
- [ ] Module 3: dynamic systems
- [ ] Module 4: state space
- [ ] Module 5: noise
- [ ] Module 6: Kalman filter
- [ ] Pick the final research question (`notes/02-research-question.md`)

## Curriculum

1. **Linear algebra for estimation**: vectors, matrices, multiplication, identity, transpose, inverse, rank, positive definite matrices (why covariances must be PSD).
2. **Probability**: random variables, expectation, variance, covariance, correlation, Gaussians, conditional probability, Bayes.
3. **Dynamic systems**: modelling a moving car, predicting tomorrow from today.
4. **State space**: what x_k means, x_k = A x_{k-1} + B u_k + w_k.
5. **Noise**: process noise vs measurement noise, sensor error models (GPS, accelerometer, gyro, encoder).
6. **Kalman filter**: predict, update, covariance propagation, derivation.

## Research sprints

1. Kalman theory from scratch
2. Simulation + implementation
3. Error analysis experiments
4. Write results like a paper
5. Identify a publication venue

## Experiment plan (draft)

Baselines: raw sensor, moving average, Kalman filter, optionally EKF.
Metrics: RMSE, estimation variance, convergence rate, robustness to noise.
Cases: A low noise, B high noise, C badly tuned Q/R, D well tuned Q/R.
Possible novelty: adaptive Q/R tuning, or a GPS+IMU drift model, or a Kalman vs particle filter comparison.

## Reading

1. Thrun, Burgard, Fox: *Probabilistic Robotics*
2. Dan Simon: *Optimal State Estimation*
3. Grewal, Andrews: *Kalman Filtering: Theory and Practice*
