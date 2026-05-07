import numpy as np
import pytest

from neuroacoustic_resonator import FieldConfig, FieldState, OscillatorField


def test_field_initializes_with_expected_shape() -> None:
    field = OscillatorField(FieldConfig(size=8, seed=1))

    state = field.state

    assert state.phase.shape == (8, 8)
    assert state.frequency.shape == (8, 8)
    assert state.metabolite.shape == (8, 8)
    assert state.coupling.shape == (8, 8)
    assert state.trace.shape == (8, 8)


def test_step_keeps_state_finite_and_bounded() -> None:
    field = OscillatorField(FieldConfig(size=8, seed=1))

    state = field.step()

    assert np.all(np.isfinite(state.phase))
    assert np.all(np.isfinite(state.frequency))
    assert np.all(np.isfinite(state.metabolite))
    assert np.all(np.isfinite(state.trace))
    assert np.all((0.0 <= state.phase) & (state.phase < 2.0 * np.pi))
    assert np.all((0.0 <= state.metabolite) & (state.metabolite <= 1.0))
    assert np.all(state.trace >= 0.0)


def test_state_returns_copy() -> None:
    field = OscillatorField(FieldConfig(size=4, seed=1))
    state = field.state

    state.phase[0, 0] = -1.0

    assert field.state.phase[0, 0] >= 0.0


def test_config_rejects_invalid_size() -> None:
    with pytest.raises(ValueError, match="size"):
        FieldConfig(size=1)


def test_synchrony_metrics_are_bounded() -> None:
    field = OscillatorField(FieldConfig(size=8, seed=1))

    local = field.local_synchrony()
    global_value = field.global_synchrony()

    assert local.shape == (8, 8)
    assert np.all(np.isfinite(local))
    assert np.all((0.0 <= local) & (local <= 1.0))
    assert 0.0 <= global_value <= 1.0


def test_synchrony_is_one_for_uniform_phase() -> None:
    config = FieldConfig(size=4)
    shape = (config.size, config.size)
    field = OscillatorField.from_state(
        config,
        FieldState(
            phase=np.full(shape, 0.5),
            frequency=np.ones(shape),
            metabolite=np.ones(shape),
            coupling=np.zeros(shape),
            trace=np.zeros(shape),
        ),
    )

    assert np.allclose(field.local_synchrony(), 1.0)
    assert field.global_synchrony() == pytest.approx(1.0)


def test_from_state_rejects_wrong_shape() -> None:
    config = FieldConfig(size=4)

    with pytest.raises(ValueError, match="phase shape"):
        OscillatorField.from_state(
            config,
            FieldState(
                phase=np.zeros((2, 2)),
                frequency=np.ones((4, 4)),
                metabolite=np.ones((4, 4)),
                coupling=np.zeros((4, 4)),
                trace=np.zeros((4, 4)),
            ),
        )


def test_trace_accumulates_activity() -> None:
    field = OscillatorField(
        FieldConfig(size=4, base_frequency=2.0, frequency_spread=0.0)
    )

    before = field.state.trace
    after = field.step().trace

    assert np.all(after > before)


def test_config_rejects_invalid_trace_rate() -> None:
    with pytest.raises(ValueError, match="trace_rate"):
        FieldConfig(trace_rate=-0.1)


def test_frequency_plasticity_updates_frequency() -> None:
    config = FieldConfig(
        size=4,
        frequency_spread=0.0,
        frequency_plasticity_rate=1.0,
        min_frequency=0.2,
        max_frequency=3.0,
    )
    shape = (config.size, config.size)
    phase_impulse = np.zeros(shape)
    phase_impulse[1, 1] = np.pi / 2.0
    field = OscillatorField.from_state(
        config,
        FieldState(
            phase=phase_impulse,
            frequency=np.ones(shape),
            metabolite=np.ones(shape),
            coupling=np.full(shape, config.coupling_strength),
            trace=np.full(shape, 1.0),
        ),
    )

    before = field.state.frequency
    after = field.step().frequency

    assert not np.allclose(after, before)
    assert np.all(np.isfinite(after))
    assert np.all((config.min_frequency <= after) & (after <= config.max_frequency))


def test_frequency_plasticity_can_be_disabled() -> None:
    config = FieldConfig(size=4, frequency_plasticity_rate=0.0)
    field = OscillatorField(config)

    before = field.state.frequency
    after = field.step().frequency

    assert np.allclose(after, before)


def test_config_rejects_invalid_frequency_bounds() -> None:
    with pytest.raises(ValueError, match="max_frequency"):
        FieldConfig(min_frequency=2.0, max_frequency=1.0)


def test_metrics_snapshot_is_bounded() -> None:
    field = OscillatorField(FieldConfig(size=8, seed=1))
    field.step()

    metrics = field.metrics(step=1)

    assert metrics.step == 1
    assert 0.0 <= metrics.min_metabolite <= metrics.mean_metabolite <= 1.0
    assert metrics.mean_trace >= 0.0
    assert metrics.max_trace >= metrics.mean_trace
    assert 0.0 <= metrics.global_synchrony <= 1.0
    assert 0.0 <= metrics.mean_local_synchrony <= metrics.max_local_synchrony <= 1.0


def test_metrics_report_uniform_phase_synchrony() -> None:
    config = FieldConfig(size=4)
    shape = (config.size, config.size)
    field = OscillatorField.from_state(
        config,
        FieldState(
            phase=np.full(shape, 1.25),
            frequency=np.ones(shape),
            metabolite=np.ones(shape),
            coupling=np.zeros(shape),
            trace=np.full(shape, 0.4),
        ),
    )

    metrics = field.metrics()

    assert metrics.global_synchrony == pytest.approx(1.0)
    assert metrics.mean_local_synchrony == pytest.approx(1.0)
    assert metrics.max_local_synchrony == pytest.approx(1.0)
    assert metrics.mean_trace == pytest.approx(0.4)
