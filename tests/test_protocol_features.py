import numpy as np
import pytest

from neuroacoustic_resonator.analysis.protocol_features import (
    pattern_snapshot,
    region_snapshot,
    weighted_phase_lock,
)
from neuroacoustic_resonator.core.field import FieldMetrics, FieldState
from neuroacoustic_resonator.core.simulation import SimulationFrame


def simulation_frame(phase: np.ndarray, *, step: int = 1) -> SimulationFrame:
    shape = phase.shape
    state = FieldState(
        phase=phase.copy(),
        frequency=np.ones(shape),
        metabolite=np.full(shape, 0.8),
        coupling=np.full(shape, 0.2),
        trace=np.full(shape, 0.1),
    )
    return SimulationFrame(
        state=state,
        metrics=FieldMetrics(
            step=step,
            mean_metabolite=0.8,
            min_metabolite=0.8,
            mean_trace=0.1,
            max_trace=0.1,
            global_synchrony=0.4,
            mean_local_synchrony=0.5,
            max_local_synchrony=0.6,
        ),
        local_synchrony=np.full(shape, 0.5),
    )


def test_pattern_snapshot_recognizes_uniform_phase_order() -> None:
    frame = simulation_frame(np.zeros((4, 4)))
    snapshot = pattern_snapshot(
        frame.state,
        np.ones((4, 4), dtype=np.bool_),
    )

    assert snapshot.phase_order_1 == pytest.approx(1.0)
    assert snapshot.phase_order_2 == pytest.approx(1.0)
    assert snapshot.phase_order_3 == pytest.approx(1.0)


def test_pattern_snapshot_exposes_split_and_triadic_orders() -> None:
    split = np.tile([0.0, np.pi], 8).reshape(4, 4)
    triadic = np.resize(np.array([0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0]), 18)
    split_snapshot = pattern_snapshot(
        simulation_frame(split).state,
        np.ones(split.shape, dtype=np.bool_),
    )
    triadic_snapshot = pattern_snapshot(
        simulation_frame(triadic.reshape(3, 6)).state,
        np.ones((3, 6), dtype=np.bool_),
    )

    assert split_snapshot.phase_order_1 == pytest.approx(0.0, abs=1e-12)
    assert split_snapshot.phase_order_2 == pytest.approx(1.0)
    assert triadic_snapshot.phase_order_1 == pytest.approx(0.0, abs=1e-12)
    assert triadic_snapshot.phase_order_3 == pytest.approx(1.0)


def test_region_snapshot_uses_only_selected_cells() -> None:
    frame = simulation_frame(np.zeros((2, 2)))
    frame.state.metabolite[0, :] = 0.4
    mask = np.zeros((2, 2), dtype=np.bool_)
    mask[0, :] = True

    snapshot = region_snapshot(frame, mask, "input")

    assert snapshot.mean_metabolite == pytest.approx(0.4)
    assert snapshot.min_metabolite == pytest.approx(0.4)
    assert snapshot.name == "input"


def test_weighted_phase_lock_handles_zero_weights() -> None:
    phase = np.array([0.0, np.pi])

    assert weighted_phase_lock(phase, np.zeros(2)) == 0.0


def test_protocol_feature_extraction_does_not_mutate_state() -> None:
    frame = simulation_frame(np.arange(16, dtype=np.float64).reshape(4, 4))
    original_phase = frame.state.phase.copy()

    pattern_snapshot(frame.state, np.ones((4, 4), dtype=np.bool_))

    np.testing.assert_array_equal(frame.state.phase, original_phase)
