import json

import pytest

from kalman_research.experiments import REGISTRY
from kalman_research.experiments.__main__ import main


@pytest.mark.parametrize("name", list(REGISTRY))
def test_experiment_runs_and_writes_outputs(name, tmp_path):
    main([name, "--out", str(tmp_path), "--runs", "3", "--seed", "0"])
    summary = json.loads((tmp_path / f"{name}.json").read_text())
    assert summary["n_runs"] == 3
    for figure in summary["figures"]:
        assert (tmp_path / figure).stat().st_size > 0
