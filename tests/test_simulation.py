import numpy as np
import pytest

from neuroacoustic_resonator import FieldConfig, FieldState, OscillatorField, Simulation
from neuroacoustic_resonator.core.input_drive import SyntheticInputConfig
from neuroacoustic_resonator.core.regions import RegionMasks


def test_simulation_starts_at_step_zero() -> None:
    simulation = Simulation(FieldConfig(size=4, seed=1))

    frame = simulation.snapshot()

    assert simulation.step_index == 0
    assert frame.metrics.step == 0
    assert frame.state.phase.shape == (4, 4)


def test_simulation_step_advances_counter() -> None:
    simulation = Simulation(FieldConfig(size=4, seed=1))

    frame = simulation.step()

    assert simulation.step_index == 1
    assert frame.metrics.step == 1


def test_simulation_run_returns_frames() -> None:
    simulation = Simulation(FieldConfig(size=4, seed=1))

    frames = simulation.run(3)

    assert len(frames) == 3
    assert simulation.step_index == 3
    assert [frame.metrics.step for frame in frames] == [1, 2, 3]


def test_simulation_accepts_existing_field() -> None:
    field = OscillatorField(FieldConfig(size=5, seed=1))
    simulation = Simulation(field=field)

    frame = simulation.snapshot()

    assert frame.state.phase.shape == (5, 5)


def test_simulation_applies_memory_drive_region_gains() -> None:
    config = FieldConfig(
        size=6,
        memory_drive_strength=1.0,
        memory_drive_input_gain=0.0,
        memory_drive_assoc_gain=2.0,
        memory_drive_output_gain=0.5,
        base_frequency=0.0,
        coupling_strength=0.0,
    )
    regions = RegionMasks.from_size(config.size)
    shape = (config.size, config.size)
    trace = np.zeros(shape)
    trace[regions.assoc] = 1.0
    trace[regions.output] = 1.0
    field = OscillatorField.from_state(
        config,
        FieldState(
            phase=np.zeros(shape),
            frequency=np.zeros(shape),
            metabolite=np.ones(shape),
            coupling=np.zeros(shape),
            trace=trace,
        ),
    )
    simulation = Simulation(field=field)

    drive = simulation.field.memory_drive()

    assert np.allclose(drive[regions.input], 0.0)
    assert np.mean(np.abs(drive[regions.assoc])) > np.mean(
        np.abs(drive[regions.output])
    )


def test_simulation_rejects_ambiguous_inputs() -> None:
    field = OscillatorField(FieldConfig(size=4, seed=1))

    with pytest.raises(ValueError, match="either config or field"):
        Simulation(config=FieldConfig(size=4), field=field)


def test_simulation_rejects_negative_run_length() -> None:
    simulation = Simulation(FieldConfig(size=4, seed=1))

    with pytest.raises(ValueError, match="steps"):
        simulation.run(-1)


def test_simulation_applies_synthetic_input_before_step() -> None:
    baseline = Simulation(
        FieldConfig(
            size=4,
            seed=1,
            coupling_strength=0.0,
            frequency_plasticity_rate=0.0,
            frequency_homeostasis_rate=0.0,
            coupling_homeostasis_rate=0.0,
        )
    )
    driven = Simulation(
        FieldConfig(
            size=4,
            seed=1,
            coupling_strength=0.0,
            frequency_plasticity_rate=0.0,
            frequency_homeostasis_rate=0.0,
            coupling_homeostasis_rate=0.0,
        ),
        synthetic_input=SyntheticInputConfig(
            enabled=True,
            mode="pulse",
            strength=0.5,
        ),
    )

    baseline_phase = baseline.step().state.phase
    driven_phase = driven.step().state.phase

    assert not np.allclose(driven_phase, baseline_phase)


def test_simulation_records_last_input_value() -> None:
    simulation = Simulation(
        FieldConfig(size=4, seed=1),
        synthetic_input=SyntheticInputConfig(
            enabled=True,
            mode="pulse",
            strength=0.5,
        ),
    )

    simulation.step()

    assert simulation.last_input_value == 0.5
