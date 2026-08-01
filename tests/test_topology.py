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
