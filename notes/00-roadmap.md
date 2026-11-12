# Roadmap

Goal: understand state estimation well enough to derive the Kalman filter, defend every equation, and write it up.

## Status

- [x] Repository, library, tests, five experiments, documentation, paper draft
- [ ] Assignment 0 answered (`notes/01-assignment-0.md`)
- [ ] Worked through `docs/01-theory.md` by hand, one note per section using `notes/_template.md`
- [ ] Re-derived the Kalman gain and the residual-based R estimator without looking
- [ ] Changed a parameter in each experiment and predicted the effect before running it
- [ ] Items in "Before submitting" at the end of `report/paper_draft.md`

## Suggested reading order

1. `docs/01-theory.md` sections 1 to 3, with `src/kalman_research/filters/kalman.py` open next to it (80 lines).
2. `tests/test_kalman.py`: each test is one claim about the filter.
3. `docs/03-results.md` E1, then theory sections 4 and 5, then E2 and E3.
4. Theory section 7 and E4.
5. Theory section 6 and E5.
6. `docs/02-architecture.md` before changing any code.

## Extensions, roughly in order of value

1. Parameter sweeps of E2/E3 over dt and σ_a/σ_z.
2. Confidence intervals in the result tables.
3. Extended Kalman filter with heading and a body-frame IMU.
4. A public GPS/IMU dataset.
5. Rauch-Tung-Striebel smoother as an offline upper bound.
