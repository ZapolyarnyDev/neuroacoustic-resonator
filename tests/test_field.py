import numpy as np
import pytest

from neuroacoustic_resonator import (
    FieldConfig,
    FieldState,
    OscillatorField,
    RegionMasks,
)


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


def test_apply_region_plasticity_changes_masked_output_state() -> None:
    field = OscillatorField(FieldConfig(size=8, seed=1))
    regions = RegionMasks.from_size(8)
    before = field.state
    field._trace[regions.output] = np.linspace(0.0, 1.0, int(np.sum(regions.output)))

    field.apply_region_plasticity(
        regions.output,
        0.5,
        coupling_rate=0.1,
        frequency_rate=0.1,
    )
    after = field.state

    assert not np.allclose(
        before.coupling[regions.output],
        after.coupling[regions.output],
    )
    assert not np.allclose(
        before.frequency[regions.output],
        after.frequency[regions.output],
    )
    assert np.allclose(
        before.coupling[~regions.output],
        after.coupling[~regions.output],
    )


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


def test_base_rotation_does_not_accumulate_trace() -> None:
    field = OscillatorField(
        FieldConfig(
            size=4,
            base_frequency=2.0,
            frequency_spread=0.0,
            coupling_strength=0.0,
        )
    )

    after = field.step().trace

    assert np.allclose(after, 0.0)


def test_base_rotation_does_not_consume_metabolite() -> None:
    field = OscillatorField(
        FieldConfig(
            size=4,
            base_frequency=2.0,
            frequency_spread=0.0,
            coupling_strength=0.0,
            metabolite_recovery=0.0,
            metabolite_cost=1.0,
        )
    )

    after = field.step().metabolite

    assert np.allclose(after, 1.0)


def test_trace_accumulates_coupling_activity() -> None:
    config = FieldConfig(
        size=4,
        dt=1.0,
        base_frequency=0.0,
        frequency_spread=0.0,
        coupling_strength=0.5,
        trace_rate=0.5,
    )
    shape = (config.size, config.size)
    phase = np.zeros(shape)
    phase[:, 1::2] = np.pi / 2.0
    field = OscillatorField.from_state(
        config,
        FieldState(
            phase=phase,
            frequency=np.zeros(shape),
            metabolite=np.ones(shape),
            coupling=np.full(shape, config.coupling_strength),
            trace=np.zeros(shape),
        ),
    )

    after = field.step().trace

    assert np.max(after) > 0.0


def test_trace_decay_reduces_sustained_activity_memory() -> None:
    base = FieldConfig(
        size=4,
        dt=1.0,
        base_frequency=0.0,
        frequency_spread=0.0,
        coupling_strength=0.5,
        trace_rate=0.5,
        trace_decay=0.0,
    )
    decaying = FieldConfig(
        size=4,
        dt=1.0,
        base_frequency=0.0,
        frequency_spread=0.0,
        coupling_strength=0.5,
        trace_rate=0.5,
        trace_decay=0.25,
    )
    shape = (base.size, base.size)
    phase = np.zeros(shape)
    phase[:, 1::2] = np.pi / 2.0
    state = FieldState(
        phase=phase,
        frequency=np.zeros(shape),
        metabolite=np.ones(shape),
        coupling=np.full(shape, base.coupling_strength),
        trace=np.zeros(shape),
    )
    base_field = OscillatorField.from_state(base, state)
    decaying_field = OscillatorField.from_state(decaying, state)
    for _ in range(3):
        base_trace = base_field.step().trace
        decaying_trace = decaying_field.step().trace

    assert np.all(decaying_trace < base_trace)


def test_config_rejects_invalid_trace_rate() -> None:
    with pytest.raises(ValueError, match="trace_rate"):
        FieldConfig(trace_rate=-0.1)


def test_config_rejects_invalid_trace_decay() -> None:
    with pytest.raises(ValueError, match="trace_decay"):
        FieldConfig(trace_decay=-0.1)


def test_config_rejects_invalid_metabolite_diffusion() -> None:
    with pytest.raises(ValueError, match="metabolite_diffusion"):
        FieldConfig(metabolite_diffusion=-0.1)


def test_config_rejects_invalid_adaptive_metabolite_recovery() -> None:
    with pytest.raises(ValueError, match="metabolite_adaptive_recovery"):
        FieldConfig(metabolite_adaptive_recovery=-0.1)


def test_config_rejects_invalid_synchrony_target_window() -> None:
    with pytest.raises(ValueError, match="synchrony_target_high"):
        FieldConfig(synchrony_target_low=0.8, synchrony_target_high=0.4)


def test_config_rejects_invalid_coupling_bounds() -> None:
    with pytest.raises(ValueError, match="max_coupling"):
        FieldConfig(min_coupling=0.5, max_coupling=0.4)


def test_metabolite_diffusion_spreads_local_depletion() -> None:
    config = FieldConfig(
        size=3,
        dt=1.0,
        metabolite_recovery=0.0,
        metabolite_cost=0.0,
        metabolite_diffusion=0.1,
        frequency_spread=0.0,
        coupling_strength=0.0,
    )
    shape = (config.size, config.size)
    metabolite = np.ones(shape)
    metabolite[1, 1] = 0.0
    field = OscillatorField.from_state(
        config,
        FieldState(
            phase=np.zeros(shape),
            frequency=np.ones(shape),
            metabolite=metabolite,
            coupling=np.zeros(shape),
            trace=np.zeros(shape),
        ),
    )

    after = field.step().metabolite

    assert after[1, 1] == pytest.approx(0.4)
    assert after[0, 1] == pytest.approx(0.9)
    assert after[0, 0] == pytest.approx(1.0)


def test_adaptive_recovery_accelerates_depleted_metabolite_recovery() -> None:
    config = FieldConfig(
        size=4,
        dt=1.0,
        base_frequency=0.0,
        frequency_spread=0.0,
        coupling_strength=0.0,
        metabolite_recovery=0.1,
        metabolite_adaptive_recovery=0.2,
        metabolite_cost=0.0,
    )
    no_adaptive_config = FieldConfig(
        size=4,
        dt=1.0,
        base_frequency=0.0,
        frequency_spread=0.0,
        coupling_strength=0.0,
        metabolite_recovery=0.1,
        metabolite_adaptive_recovery=0.0,
        metabolite_cost=0.0,
    )
    shape = (config.size, config.size)
    depleted = np.full(shape, 0.5)
    adaptive = OscillatorField.from_state(
        config,
        FieldState(
            phase=np.zeros(shape),
            frequency=np.zeros(shape),
            metabolite=depleted,
            coupling=np.zeros(shape),
            trace=np.zeros(shape),
        ),
    )
    baseline = OscillatorField.from_state(
        no_adaptive_config,
        FieldState(
            phase=np.zeros(shape),
            frequency=np.zeros(shape),
            metabolite=depleted,
            coupling=np.zeros(shape),
            trace=np.zeros(shape),
        ),
    )

    assert np.mean(adaptive.step().metabolite) > np.mean(baseline.step().metabolite)


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
    config = FieldConfig(
        size=4,
        frequency_plasticity_rate=0.0,
        frequency_homeostasis_rate=0.0,
    )
    field = OscillatorField(config)

    before = field.state.frequency
    after = field.step().frequency

    assert np.allclose(after, before)


def test_config_rejects_invalid_frequency_bounds() -> None:
    with pytest.raises(ValueError, match="max_frequency"):
        FieldConfig(min_frequency=2.0, max_frequency=1.0)


def test_frequency_homeostasis_pulls_frequency_toward_base() -> None:
    config = FieldConfig(
        size=4,
        dt=1.0,
        base_frequency=1.0,
        frequency_spread=0.0,
        frequency_plasticity_rate=0.0,
        frequency_homeostasis_rate=0.25,
        coupling_homeostasis_rate=0.0,
        min_frequency=0.2,
        max_frequency=3.0,
    )
    shape = (config.size, config.size)
    field = OscillatorField.from_state(
        config,
        FieldState(
            phase=np.zeros(shape),
            frequency=np.full(shape, 2.0),
            metabolite=np.ones(shape),
            coupling=np.zeros(shape),
            trace=np.zeros(shape),
        ),
    )

    after = field.step().frequency

    assert np.all(after < 2.0)
    assert np.all(after > config.base_frequency)


def test_synchrony_homeostasis_reduces_coupling_when_synchrony_is_high() -> None:
    config = FieldConfig(
        size=4,
        dt=1.0,
        coupling_strength=0.5,
        frequency_plasticity_rate=0.0,
        frequency_homeostasis_rate=0.0,
        coupling_homeostasis_rate=0.1,
        synchrony_target_low=0.1,
        synchrony_target_high=0.5,
        min_coupling=0.0,
        max_coupling=1.0,
    )
    shape = (config.size, config.size)
    field = OscillatorField.from_state(
        config,
        FieldState(
            phase=np.zeros(shape),
            frequency=np.ones(shape),
            metabolite=np.ones(shape),
            coupling=np.full(shape, config.coupling_strength),
            trace=np.zeros(shape),
        ),
    )

    after = field.step().coupling

    assert np.all(after < config.coupling_strength)


def test_synchrony_homeostasis_increases_coupling_when_synchrony_is_low() -> None:
    config = FieldConfig(
        size=4,
        dt=1.0,
        coupling_strength=0.5,
        base_frequency=0.0,
        frequency_plasticity_rate=0.0,
        frequency_homeostasis_rate=0.0,
        coupling_homeostasis_rate=0.1,
        synchrony_target_low=0.2,
        synchrony_target_high=0.8,
        min_coupling=0.0,
        max_coupling=1.0,
    )
    shape = (config.size, config.size)
    phase = np.zeros(shape)
    phase[:, 1::2] = np.pi
    field = OscillatorField.from_state(
        config,
        FieldState(
            phase=phase,
            frequency=np.zeros(shape),
            metabolite=np.ones(shape),
            coupling=np.full(shape, config.coupling_strength),
            trace=np.zeros(shape),
        ),
    )

    after = field.step().coupling

    assert np.all(after > config.coupling_strength)


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
