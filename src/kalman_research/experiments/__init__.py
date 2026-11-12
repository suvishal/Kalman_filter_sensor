"""Reproducible experiments. Each module exposes ``run(out_dir, n_runs, seed) -> dict``.

To add an experiment: write a module with that ``run`` function and register
it in ``REGISTRY`` below. The CLI, the JSON summaries, and the smoke test pick
it up automatically.
"""

from . import adaptive_noise, baselines, consistency, gps_imu_fusion, tuning_sensitivity

REGISTRY = {
    "baselines": baselines,
    "tuning_sensitivity": tuning_sensitivity,
    "consistency": consistency,
    "gps_imu_fusion": gps_imu_fusion,
    "adaptive_noise": adaptive_noise,
}
