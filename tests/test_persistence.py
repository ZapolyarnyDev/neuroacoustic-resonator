from __future__ import annotations

import numpy as np
import pytest

from neuroacoustic_resonator import FieldConfig, Simulation
from neuroacoustic_resonator.core.input_drive import SyntheticInputConfig
from neuroacoustic_resonator.io.persistence import (
    checkpoint_fingerprint,
    checkpoint_paths,
    load_checkpoint_metadata,
    load_field_state,
    load_simulation_checkpoint,
    save_field_state,
    save_simulation_checkpoint,
)


def test_checkpoint_paths_normalizes_npz_suffix() -> None:
    paths = checkpoint_paths("experiments/states/demo")

    assert paths.arrays_path.name == "demo.npz"
    assert paths.metadata_path.name == "demo.yaml"


def test_save_and_load_field_state_roundtrip(tmp_path) -> None:
    simulation = Simulation(FieldConfig(size=5, seed=1))
    simulation.step()

    paths = save_field_state(
        tmp_path / "state",
        simulation.field.state,
        metadata={"note": "demo"},
    )
    loaded = load_field_state(paths.arrays_path)
    metadata = load_checkpoint_metadata(paths.arrays_path)

    assert paths.arrays_path.exists()
    assert paths.metadata_path.exists()
    assert metadata["note"] == "demo"
    assert np.allclose(loaded.phase, simulation.field.state.phase)
    assert np.allclose(loaded.frequency, simulation.field.state.frequency)
    assert np.allclose(loaded.metabolite, simulation.field.state.metabolite)
    assert np.allclose(loaded.coupling, simulation.field.state.coupling)
    assert np.allclose(loaded.trace, simulation.field.state.trace)


def test_save_and_load_simulation_checkpoint_roundtrip(tmp_path) -> None:
    simulation = Simulation(
        FieldConfig(
            size=5,
            seed=1,
            coupling_strength=0.2,
            boundary_x="open",
            boundary_y="periodic",
        ),
        synthetic_input=SyntheticInputConfig(enabled=True, mode="pulse", seed=2),
    )
    simulation.run(3)

    paths = save_simulation_checkpoint(tmp_path / "checkpoint", simulation)
    loaded = load_simulation_checkpoint(paths.arrays_path)
    metadata = load_checkpoint_metadata(paths.arrays_path)

    assert loaded.step_index == 3
    assert loaded.field.config.coupling_strength == 0.2
    assert loaded.field.config.boundary_x == "open"
    assert loaded.field.config.boundary_y == "periodic"
    assert loaded.input_drive.config.enabled is True
    assert loaded.input_drive.config.mode == "pulse"
    assert np.allclose(loaded.field.state.phase, simulation.field.state.phase)
    assert metadata["fingerprint"] == checkpoint_fingerprint(paths.arrays_path)


def test_load_simulation_checkpoint_rejects_tampered_state(tmp_path) -> None:
    simulation = Simulation(FieldConfig(size=5, seed=1))
    paths = save_simulation_checkpoint(tmp_path / "checkpoint", simulation)
    state = load_field_state(paths.arrays_path)
    state.phase[0, 0] += 0.25
    np.savez_compressed(
        paths.arrays_path,
        phase=state.phase,
        frequency=state.frequency,
        metabolite=state.metabolite,
        coupling=state.coupling,
        trace=state.trace,
    )

    with pytest.raises(ValueError, match="fingerprint"):
        load_simulation_checkpoint(paths.arrays_path)


def test_load_simulation_checkpoint_rejects_missing_metadata(tmp_path) -> None:
    simulation = Simulation(FieldConfig(size=5, seed=1))
    save_field_state(tmp_path / "state", simulation.field.state, metadata={})

    with pytest.raises(ValueError, match="unsupported checkpoint version"):
        load_simulation_checkpoint(tmp_path / "state")


def test_first_order_checkpoint_stores_no_velocity(tmp_path) -> None:
    path = tmp_path / "first.npz"
    save_simulation_checkpoint(path, Simulation(FieldConfig(size=4, seed=1)))

    with np.load(path) as data:
        assert "velocity" not in data
    assert load_field_state(path).velocity is None


def test_second_order_checkpoint_roundtrips_velocity(tmp_path) -> None:
    path = tmp_path / "second.npz"
    simulation = Simulation(FieldConfig(size=4, seed=1, order=2, damping=0.05))
    simulation.run(3)
    save_simulation_checkpoint(path, simulation)

    loaded = load_simulation_checkpoint(path)

    assert loaded.field.config.order == 2
    np.testing.assert_array_equal(
        loaded.field.state.velocity, simulation.field.state.velocity
    )
