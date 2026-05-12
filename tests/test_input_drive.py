import numpy as np
import pytest

from neuroacoustic_resonator import (
    FieldConfig,
    OscillatorField,
    RegionMasks,
    SyntheticInputConfig,
    SyntheticInputDrive,
)
from neuroacoustic_resonator.field import TAU


def test_disabled_synthetic_input_returns_zero() -> None:
    regions = RegionMasks.from_size(4)
    drive = SyntheticInputDrive(SyntheticInputConfig(enabled=False), regions)

    assert drive.value(0) == 0.0


def test_sine_synthetic_input_is_periodic() -> None:
    regions = RegionMasks.from_size(4)
    drive = SyntheticInputDrive(
        SyntheticInputConfig(enabled=True, mode="sine", strength=2.0, period_steps=4),
        regions,
    )

    assert drive.value(0) == pytest.approx(0.0)
    assert drive.value(1) == pytest.approx(2.0)
    assert drive.value(2) == pytest.approx(0.0)
    assert drive.value(3) == pytest.approx(-2.0)


def test_pulse_synthetic_input_uses_duty_cycle() -> None:
    regions = RegionMasks.from_size(4)
    drive = SyntheticInputDrive(
        SyntheticInputConfig(
            enabled=True,
            mode="pulse",
            strength=0.5,
            period_steps=4,
            duty_cycle=0.5,
        ),
        regions,
    )

    assert [drive.value(step) for step in range(5)] == [0.5, 0.5, 0.0, 0.0, 0.5]


def test_synthetic_input_applies_phase_impulse_to_input_region() -> None:
    field = OscillatorField(FieldConfig(size=4, seed=1))
    regions = RegionMasks.from_size(4)
    drive = SyntheticInputDrive(
        SyntheticInputConfig(enabled=True, mode="pulse", strength=0.5),
        regions,
    )
    before = field.state.phase

    applied = drive.apply(field, step=0)
    after = field.state.phase

    assert applied == pytest.approx(0.5)
    assert np.allclose(after[regions.input], np.mod(before[regions.input] + 0.5, TAU))
    assert np.allclose(after[~regions.input], before[~regions.input])


def test_synthetic_input_config_rejects_invalid_strength() -> None:
    with pytest.raises(ValueError, match="strength"):
        SyntheticInputConfig(strength=-1.0)
