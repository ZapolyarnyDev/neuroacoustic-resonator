from __future__ import annotations

import numpy as np
import pytest

from neuroacoustic_resonator.core.spatial_transport import (
    SpatialTransport,
    SpatialTransportConfig,
)


def test_transport_moves_pulse_right_and_continues_after_input() -> None:
    transport = SpatialTransport(
        SpatialTransportConfig(
            size=8,
            bands=1,
            source_positions=(0.5,),
            dt=0.02,
            flow_rate=50.0,
            diffusion_rate=0.0,
            decay_rate=0.0,
        )
    )
    first = transport.step(np.array([[1.0]]))
    second = transport.step()
    assert np.sum(first[:, :, 0]) == pytest.approx(1.0)
    assert np.sum(second[:, :, 1]) == pytest.approx(1.0)
    assert np.sum(second[:, :, 0]) == pytest.approx(0.0)


def test_transport_preserves_channel_position_and_mass_before_exit() -> None:
    transport = SpatialTransport(
        SpatialTransportConfig(
            size=12,
            bands=2,
            source_positions=(0.2, 0.8),
            dt=0.02,
            flow_rate=10.0,
            diffusion_rate=4.0,
            decay_rate=0.0,
        )
    )
    state = transport.step(np.array([[1.0, 0.0], [0.0, 0.5]]))
    assert np.sum(state) == pytest.approx(1.5)
    assert np.argmax(state[0, :, 0]) < np.argmax(state[1, :, 0])
    for _ in range(10):
        state = transport.step()
        assert np.all(state >= 0.0)
    assert np.sum(state) == pytest.approx(1.5)


def test_transport_rejects_unstable_parameters_and_invalid_energy() -> None:
    with pytest.raises(ValueError, match="non-negative update"):
        SpatialTransportConfig(size=8, bands=1, source_positions=(0.5,), flow_rate=60.0)
    transport = SpatialTransport(
        SpatialTransportConfig(size=8, bands=1, source_positions=(0.5,))
    )
    with pytest.raises(ValueError, match="non-negative"):
        transport.step(np.array([[-1.0]]))
