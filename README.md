# Kalman Filter Research

A from-scratch study of Kalman filter state estimation and sensor fusion, structured as a small research project with the aim of a short workshop/student-style paper.

**Working title:** Performance Analysis of Kalman Filter-Based Sensor Fusion under Varying Noise Conditions

**Rules of the project**
- No Kalman library (no FilterPy) for the core filter. Implement everything with NumPy.
- Every experiment is reproducible (fixed seeds, scripts in `experiments/`).
- Every formula in code traces back to a note in `notes/`.
- Every concept in the notes answers: what problem does it solve, what assumptions does it make, when does it fail, how can I show it experimentally?

## Layout

| Path | Purpose |
|---|---|
| `notes/` | Research notebook: theory, derivations, assignment answers |
| `papers/` | Reference papers and reading notes |
| `src/kalman_research/` | Library code (simulation, filters, metrics, plotting) |
| `experiments/` | Runnable, seeded experiment scripts |
| `notebooks/` | Exploratory notebooks |
| `tests/` | pytest tests (filter correctness, sanity checks) |
| `results/` | Generated outputs (gitignored except `.gitkeep`) |
| `report/` | The paper draft |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest
```

## Where we are

See [`notes/00-roadmap.md`](notes/00-roadmap.md) for the plan and current status.
