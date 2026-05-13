from __future__ import annotations

import pytest

from neuroacoustic_resonator.core.field import FieldConfig
from neuroacoustic_resonator.core.simulation import Simulation
from neuroacoustic_resonator.io.persistence import (
    load_checkpoint_metadata,
    save_simulation_checkpoint,
)
from neuroacoustic_resonator.io.resume import resume_simulation_checkpoint


def test_resume_simulation_checkpoint_writes_resumed_checkpoint(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.npz"
    output = tmp_path / "resumed.npz"
    simulation = Simulation(FieldConfig(size=5, seed=1))
    simulation.run(3)
    save_simulation_checkpoint(checkpoint, simulation)

    paths = resume_simulation_checkpoint(
        checkpoint,
        steps=4,
        output=output,
    )
    metadata = load_checkpoint_metadata(output)

    assert paths.arrays_path == output
    assert output.exists()
    assert metadata["step_index"] == 7
    assert metadata["metadata"]["start_step"] == 3
    assert metadata["metadata"]["resumed_steps"] == 4


def test_resume_simulation_checkpoint_rejects_invalid_steps(tmp_path) -> None:
    with pytest.raises(ValueError, match="steps"):
        resume_simulation_checkpoint(tmp_path / "missing.npz", steps=0)
