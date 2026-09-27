from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import numpy as np

from neuroacoustic_resonator.audio.input import ChannelAudioFeatures
from neuroacoustic_resonator.core.regions import RegionMasks
from neuroacoustic_resonator.core.simulation import Simulation, SimulationFrame
from neuroacoustic_resonator.core.spatial_transport import (
    SpatialTransport,
    SpatialTransportConfig,
)


@dataclass(frozen=True)
class SpatialInputConfig:
    source_positions: tuple[float, ...]
    drive_strength: float = 0.45
    flow_rate: float = 40.0
    diffusion_rate: float = 5.0
    decay_rate: float = 0.5
    aperture_sigma: float = 1.0

    @classmethod
    def canonical(cls, channels: int) -> SpatialInputConfig:
        if channels < 1:
            msg = "channels must be positive"
            raise ValueError(msg)
        if channels == 1:
            return cls(source_positions=(0.5,))
        return cls(
            source_positions=tuple(
                0.25 + 0.5 * index / (channels - 1) for index in range(channels)
            )
        )


class SpatialAudioDrive:
    def __init__(
        self,
        features: ChannelAudioFeatures,
        simulation: Simulation,
        config: SpatialInputConfig,
    ) -> None:
        if features.channel_count != len(config.source_positions):
            msg = "every audio channel requires one source position"
            raise ValueError(msg)
        if not np.isfinite(config.drive_strength) or config.drive_strength < 0.0:
            msg = "drive_strength must be finite and non-negative"
            raise ValueError(msg)
        size = simulation.field.config.size
        self.features = features
        self.config = config
        self.transport = SpatialTransport(
            SpatialTransportConfig(
                size=size,
                bands=features.band_energy.shape[2],
                source_positions=config.source_positions,
                dt=simulation.field.config.dt,
                flow_rate=config.flow_rate,
                diffusion_rate=config.diffusion_rate,
                decay_rate=config.decay_rate,
                aperture_sigma=config.aperture_sigma,
            )
        )
        self._output = RegionMasks.from_size(size).output

    @property
    def input_steps(self) -> int:
        return ceil(self.features.duration_seconds / self.transport.config.dt)

    def step(self, simulation: Simulation, time_seconds: float) -> SimulationFrame:
        if (
            simulation.field.config.size != self.transport.config.size
            or simulation.field.config.dt != self.transport.config.dt
        ):
            msg = "drive and simulation grid and dt must match"
            raise ValueError(msg)
        source = self.features.energy_at_time(time_seconds)
        plume = self.transport.step(source)
        weights = np.linspace(0.5, 1.5, plume.shape[0], dtype=np.float64)
        phase_pattern = np.einsum("b,byx->yx", weights, plume)
        phase_pattern[self._output] = 0.0
        simulation.field.apply_phase_pattern(
            np.tanh(phase_pattern) * self.config.drive_strength
        )
        return simulation.step_with_input(float(np.sum(source)))
