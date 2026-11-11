from __future__ import annotations

import numpy as np
import pytest

from neuroacoustic_resonator.core.topology import BoundaryMode, GridTopology


def test_periodic_topology_gives_every_cell_four_neighbors() -> None:
    topology = GridTopology((3, 4))

    assert np.all(topology.neighbor_count() == 4.0)


def test_open_topology_counts_edges_and_corners() -> None:
    topology = GridTopology((3, 4), boundary_x="open", boundary_y="open")
    count = topology.neighbor_count()

    assert count[0, 0] == 2.0
    assert count[0, 1] == 3.0
    assert count[1, 1] == 4.0


def test_cylindrical_topology_wraps_y_but_not_x() -> None:
    values = np.zeros((3, 4), dtype=np.float64)
    values[0, 0] = 1.0
    topology = GridTopology((3, 4), boundary_x="open", boundary_y="periodic")
    neighbors = topology.neighbor_sum(values)

    assert neighbors[2, 0] == 1.0
    assert neighbors[0, 3] == 0.0


@pytest.mark.parametrize(
    ("boundary_x", "boundary_y"),
    [("open", "open"), ("open", "periodic"), ("periodic", "periodic")],
)
def test_constant_field_has_zero_laplacian(
    boundary_x: BoundaryMode,
    boundary_y: BoundaryMode,
) -> None:
    topology = GridTopology(
        (4, 5),
        boundary_x=boundary_x,
        boundary_y=boundary_y,
    )
    values = np.full((4, 5), 0.75, dtype=np.float64)

    assert np.allclose(topology.laplacian(values), 0.0)


def test_topology_phase_operations_are_finite() -> None:
    topology = GridTopology((3, 4), boundary_x="open", boundary_y="periodic")
    phase = np.linspace(0.0, 2.0 * np.pi, 12).reshape(3, 4)

    assert np.all(np.isfinite(topology.phase_coupling(phase)))
    assert np.all(np.isfinite(topology.local_phase_order(phase)))


def test_topology_rejects_mismatched_shape() -> None:
    topology = GridTopology((3, 4))

    with pytest.raises(ValueError, match="shape"):
        topology.neighbor_sum(np.zeros((4, 3), dtype=np.float64))


def _reference_coupling(
    phase: np.ndarray, phase_lag: float, second_harmonic: float, forward_bias: float
) -> np.ndarray:
    rows, cols = phase.shape
    result = np.zeros_like(phase)
    for y in range(rows):
        for x in range(cols):
            total = 0.0
            norm = 0.0
            neighbors = [
                ((y - 1) % rows, x, 1.0),
                ((y + 1) % rows, x, 1.0),
                (y, x - 1, 1.0 + forward_bias),
                (y, x + 1, 1.0 - forward_bias),
            ]
            for ny, nx, weight in neighbors:
                if not 0 <= nx < cols:
                    continue
                difference = phase[ny, nx] - phase[y, x]
                total += weight * (
                    np.sin(difference - phase_lag)
                    + second_harmonic * np.sin(2.0 * difference)
                )
                norm += weight
            result[y, x] = total / norm
    return result


def test_phase_coupling_options_default_to_plain_kuramoto() -> None:
    topology = GridTopology((5, 6), boundary_x="open", boundary_y="periodic")
    phase = np.random.default_rng(3).uniform(0.0, 2.0 * np.pi, (5, 6))

    plain = topology.phase_coupling(phase)

    np.testing.assert_allclose(plain, _reference_coupling(phase, 0.0, 0.0, 0.0))
    np.testing.assert_allclose(
        topology.phase_coupling(phase, phase_lag=1e-300), plain, atol=1e-12
    )


@pytest.mark.parametrize(
    ("phase_lag", "second_harmonic", "forward_bias"),
    [(0.3, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, 0.5), (0.15, 0.5, 0.4)],
)
def test_phase_coupling_matches_reference_with_options(
    phase_lag: float, second_harmonic: float, forward_bias: float
) -> None:
    topology = GridTopology((5, 6), boundary_x="open", boundary_y="periodic")
    phase = np.random.default_rng(4).uniform(0.0, 2.0 * np.pi, (5, 6))

    actual = topology.phase_coupling(
        phase,
        phase_lag=phase_lag,
        second_harmonic=second_harmonic,
        forward_bias=forward_bias,
    )

    np.testing.assert_allclose(
        actual,
        _reference_coupling(phase, phase_lag, second_harmonic, forward_bias),
        atol=1e-12,
    )


def test_phase_lag_pulls_uniform_phases_by_minus_sine() -> None:
    topology = GridTopology((4, 4))

    coupling = topology.phase_coupling(np.full((4, 4), 1.2), phase_lag=0.3)

    np.testing.assert_allclose(coupling, -np.sin(0.3))
