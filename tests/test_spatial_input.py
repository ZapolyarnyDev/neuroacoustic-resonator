from __future__ import annotations

import numpy as np
import pytest

from neuroacoustic_resonator.audio.input import ChannelAudioFeatures
from neuroacoustic_resonator.audio.spatial_input import (
    SpatialAudioDrive,
    SpatialInputConfig,
)
from neuroacoustic_resonator.core.field import FieldConfig
from neuroacoustic_resonator.core.regions import RegionMasks
from neuroacoustic_resonator.core.simulation import Simulation


def test_spatial_drive_uses_model_time_and_does_not_force_output() -> None:
    features = ChannelAudioFeatures(
        sample_rate=100,
        hop_size=1,
        sample_count=2,
        band_energy=np.array([[[1.0]], [[0.0]]]),
    )
    config = FieldConfig(size=8, dt=0.01, coupling_strength=0.0, seed=9)
    active = Simulation(config)
    silent = Simulation(config)
    drive = SpatialAudioDrive(
        features,
        active,
        SpatialInputConfig(source_positions=(0.5,), flow_rate=100.0),
    )
    assert drive.input_steps == 2

    driven = drive.step(active, 0.0)
    baseline = silent.step_with_input(0.0)
    regions = RegionMasks.from_size(8)
    assert np.any(
        driven.state.phase[regions.input] != baseline.state.phase[regions.input]
    )
    assert np.allclose(
        driven.state.phase[regions.output], baseline.state.phase[regions.output]
    )
    after_end = drive.step(active, 0.02)
    assert active.last_input_value == 0.0
    assert np.any(drive.transport.state[:, :, 1] > 0.0)
    assert after_end.metrics.step == 2


def test_spatial_drive_validates_channel_geometry() -> None:
    features = ChannelAudioFeatures(
        sample_rate=100,
        hop_size=1,
        sample_count=1,
        band_energy=np.zeros((1, 2, 1)),
    )
    with pytest.raises(ValueError, match="source position"):
        SpatialAudioDrive(
            features,
            Simulation(FieldConfig(size=8, seed=1)),
            SpatialInputConfig(source_positions=(0.5,)),
        )
