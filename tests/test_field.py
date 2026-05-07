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
