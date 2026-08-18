from __future__ import annotations

import numpy as np
import pytest

from neuroacoustic_resonator.core.equilibration import (
    FieldEquilibrationConfig,
    equilibrate_simulation,
)
from neuroacoustic_resonator.core.field import FieldConfig
from neuroacoustic_resonator.core.simulation import Simulation


def test_equilibration_applies_controlled_phase_and_metabolite_baselines() -> None:
    simulation = Simulation(FieldConfig(size=5, seed=7))

    result = equilibrate_simulation(
        simulation,
        FieldEquilibrationConfig(
            neutral_steps=0,
            phase_damping=1.0,
            metabolite_baseline=0.75,
        ),
    )

    assert np.allclose(simulation.field.state.phase, 0.0)
    assert np.allclose(simulation.field.state.metabolite, 0.75)
    assert result["start_step"] == 0
    assert result["end_step"] == 0


def test_equilibration_runs_only_neutral_input() -> None:
    simulation = Simulation(FieldConfig(size=5, seed=7))

    equilibrate_simulation(
        simulation,
        FieldEquilibrationConfig(
            neutral_steps=4,
            phase_damping=0.0,
            metabolite_baseline=None,
        ),
    )

    assert simulation.step_index == 4
    assert simulation.last_input_value == 0.0


def test_equilibration_rejects_invalid_phase_damping() -> None:
    with pytest.raises(ValueError, match="phase_damping"):
        FieldEquilibrationConfig(phase_damping=1.1)
