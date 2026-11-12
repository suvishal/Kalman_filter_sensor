# Accuracy Is Not Honesty: Tuning Sensitivity, Consistency and Adaptive Noise Estimation in Kalman Filter Sensor Fusion

*Draft. Author, affiliation and venue formatting to be added. All numbers are from `results/*.json`, seed 2026.*

## Abstract

The Kalman filter is optimal only when its noise covariances Q and R match reality, and in practice they are tuned by hand. We study what mis-tuning costs, how it can be detected, and how far simple online adaptation repairs it, using a from-scratch implementation and reproducible Monte Carlo experiments on 2D tracking and GPS + accelerometer fusion. Using a closed-form expression for the true error covariance of a mis-tuned filter, validated against simulation to within 0.4%, we show that (i) accuracy depends only on the ratio of Q to R, so a filter can match the optimal RMSE while reporting a covariance wrong by three orders of magnitude; (ii) the cost of mis-tuning is asymmetric, with a thousandfold under-estimate of Q inflating RMSE 6.8 times against 2.0 times for the same over-estimate; (iii) the innovation-based NIS test accepts a conservative filter 86% of the time, while innovation autocorrelation exposes it; (iv) fusing an accelerometer with GPS while ignoring a 0.15 m/s² bias is five times worse than using GPS alone, whereas estimating the bias is 1.8 times better and limits drift in a 15 s GPS outage to 11 m; and (v) a residual-based adaptive estimate of R recovers oracle accuracy from a hundredfold initial error, at the cost of a lag of a few seconds when the sensor changes.

## 1. Introduction

- State estimation from noisy sensors; the Kalman filter as the standard tool (Kalman, 1960).
- The gap this paper addresses: textbooks present the filter with known Q and R; practitioners must guess them. Reports of filter performance usually give accuracy (RMSE) only.
- Thesis: accuracy and consistency are separate properties, and RMSE alone cannot validate a noise model.
- Contributions:
  1. A closed-form, simulation-validated map of accuracy and consistency over a 13 × 13 grid of Q and R mis-scaling.
  2. A demonstration that NIS alone is a weak test for conservative tuning and that innovation whiteness complements it.
  3. A controlled GPS + accelerometer study isolating the effect of modelling sensor bias, including a GPS outage.
  4. An evaluation of residual-based adaptive R estimation under static and time-varying mis-specification, with its lag quantified.
  5. An open, tested, NumPy-only implementation that reproduces every number with one command.

## 2. Background

Material: `docs/01-theory.md` sections 2 to 4. State-space model, predict and update equations, Joseph form, NEES, NIS, whiteness (Bar-Shalom et al., 2001).

## 3. Method

### 3.1 Exact error of a mis-tuned filter
Riccati steady state for the filter's belief, Lyapunov equation for the true error covariance, the ratio property. Material: `docs/01-theory.md` section 5.

### 3.2 Residual-based adaptive R
Derivation of $R = \mathbb E[\varepsilon\varepsilon^T] + HPH^T$ and the exponentially weighted estimator. Material: `docs/01-theory.md` section 6. Related work: Mehra (1970), Mohamed and Schwarz (1999).

### 3.3 Fusion models
Four-state IMU-driven model and six-state model with accelerometer bias. Material: `docs/01-theory.md` section 7.

## 4. Experimental setup

Scenario parameters, Monte Carlo sizes, burn-in and the baselines are in `docs/03-results.md` ("Scenarios" and the first paragraph of each experiment). State explicitly that baselines were tuned with access to the truth so that comparisons favour them.

## 5. Results

Tables and figures from `docs/03-results.md`, in this order:

| Section | Source | Figures |
|---|---|---|
| 5.1 Filter against model-free smoothing | E1 | `e1_rmse_vs_noise` |
| 5.2 Mis-tuning: accuracy and consistency | E2 | `e2_rmse_heatmap`, `e2_nees_heatmap`, `e2_ratio_slice` |
| 5.3 Detecting mis-tuning without ground truth | E3 | `e3_nees_over_time`, `e3_innovation_autocorrelation` |
| 5.4 GPS + accelerometer fusion and the role of bias | E4 | `e4_error_over_time`, `e4_error_with_outage`, `e4_bias_estimate` |
| 5.5 Adaptive measurement noise | E5 | `e5_static_misspecification`, `e5_noise_tracking`, `e5_nees_over_time` |

## 6. Discussion

- **Report consistency, not only RMSE.** E2 shows two filters with identical RMSE and a millionfold spread in NEES.
- **When unsure, over-estimate Q.** The error of a filter with large Q is bounded by the sensor noise; with small Q it is not.
- **Tune with whiteness as well as NIS.** E3.
- **A wrong model can make a second sensor harmful.** E4: adding the accelerometer without a bias state degrades the estimate; the filter's tiny Q makes it ignore the sensor that could have corrected it.
- **Adaptation removes the need to know R but not the need to know Q**, and it introduces a memory-length trade-off. E5.

## 7. Limitations

- All results are from simulation; noise is Gaussian and white by construction.
- Models are linear. The accelerometer measures in the world frame, so orientation, gravity compensation and gyroscope errors are absent. A real IMU needs an extended or error-state filter.
- The truth in E4 is generated with the same discretisation the filters use, so discretisation error is excluded.
- Convergence of the adaptive estimator is shown empirically for this system, not proved. Only R is adapted.
- One trajectory shape and one set of sensor parameters in E4.

## 8. Conclusion

To be written last. One paragraph restating the five findings and the practical tuning procedure they imply.

## References

See `docs/01-theory.md`. Verify every entry against the original source before submission.

---

## Before submitting: what this draft still needs

1. **Positioning.** The ratio property, NEES/NIS tests and covariance-matching adaptation are established results. The contribution here is a unified, reproducible, quantitative treatment, which suits a tutorial or student track, an educational venue, or an arXiv preprint. It is not a claim of a new algorithm, and the paper should say so.
2. **A literature review.** Read and cite work on mis-tuned and adaptive Kalman filtering beyond the starting references, and state how these results compare.
3. **Real data.** One experiment on a public GPS/IMU dataset would be the single largest improvement. NIS and whiteness need no ground truth, so E3's method transfers directly. This requires the nonlinear extension outlined in `docs/02-architecture.md`.
4. **Sensitivity of the conclusions.** Repeat E2 and E3 for other values of dt and σ_a/σ_z to show the asymmetry and the weak NIS test are not specific to one operating point. Vary the forgetting factor in E5.
5. **Uncertainty on the tables.** Add confidence intervals across Monte Carlo runs to the RMSE figures.
6. **Your own derivations.** Work through `docs/01-theory.md` by hand until each step can be reproduced without the notes; reviewers and interviewers will ask.
