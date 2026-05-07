import numpy as np
import pytest

from neuroacoustic_resonator import FieldConfig, OscillatorField


def test_field_initializes_with_expected_shape() -> None:
    field = OscillatorField(FieldConfig(size=8, seed=1))

    state = field.state

    assert state.phase.shape == (8, 8)
    assert state.frequency.shape == (8, 8)
    assert state.metabolite.shape == (8, 8)
    assert state.coupling.shape == (8, 8)


def test_step_keeps_state_finite_and_bounded() -> None:
    field = OscillatorField(FieldConfig(size=8, seed=1))

    state = field.step()

    assert np.all(np.isfinite(state.phase))
    assert np.all(np.isfinite(state.frequency))
    assert np.all(np.isfinite(state.metabolite))
    assert np.all((0.0 <= state.phase) & (state.phase < 2.0 * np.pi))
    assert np.all((0.0 <= state.metabolite) & (state.metabolite <= 1.0))


def test_state_returns_copy() -> None:
    field = OscillatorField(FieldConfig(size=4, seed=1))
    state = field.state

    state.phase[0, 0] = -1.0

    assert field.state.phase[0, 0] >= 0.0


def test_config_rejects_invalid_size() -> None:
    with pytest.raises(ValueError, match="size"):
        FieldConfig(size=1)
