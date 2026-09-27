from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SpatialTransportConfig:
    size: int
    bands: int
    source_positions: tuple[float, ...]
    dt: float = 0.02
    flow_rate: float = 40.0
    diffusion_rate: float = 5.0
    decay_rate: float = 0.5
    aperture_sigma: float = 1.0

    def __post_init__(self) -> None:
        if self.size < 3 or self.bands < 1 or not self.source_positions:
            msg = "transport requires size >= 3, bands >= 1, and sources"
            raise ValueError(msg)
        values = (
            self.dt,
            self.flow_rate,
            self.diffusion_rate,
            self.decay_rate,
            self.aperture_sigma,
            *self.source_positions,
        )
        if not all(isfinite(value) for value in values):
            msg = "transport parameters must be finite"
            raise ValueError(msg)
        if self.dt <= 0.0 or self.aperture_sigma <= 0.0:
            msg = "transport dt and aperture_sigma must be positive"
            raise ValueError(msg)
        if min(self.flow_rate, self.diffusion_rate, self.decay_rate) < 0.0:
            msg = "transport rates must be non-negative"
            raise ValueError(msg)
        if self.flow_rate * self.dt > 1.0 or self.diffusion_rate * self.dt > 0.5:
            msg = "transport step exceeds the non-negative update limit"
            raise ValueError(msg)
        if any(not 0.0 <= position <= 1.0 for position in self.source_positions):
            msg = "source positions must be in [0, 1]"
            raise ValueError(msg)


class SpatialTransport:
    def __init__(self, config: SpatialTransportConfig) -> None:
        self.config = config
        self._state = np.zeros(
            (config.bands, config.size, config.size), dtype=np.float64
        )
        rows = np.arange(config.size, dtype=np.float64)
        apertures = []
        for position in config.source_positions:
            center = position * (config.size - 1)
            aperture = np.exp(-0.5 * ((rows - center) / config.aperture_sigma) ** 2)
            apertures.append(aperture / np.sum(aperture))
        self._apertures = np.stack(apertures)

    @property
    def state(self) -> FloatArray:
        return self._state.copy()

    def step(self, source: FloatArray | None = None) -> FloatArray:
        config = self.config
        if source is None:
            source = np.zeros((len(config.source_positions), config.bands))
        if source.shape != (len(config.source_positions), config.bands):
            msg = "source shape must be channel x band"
            raise ValueError(msg)
        if not np.all(np.isfinite(source)) or np.any(source < 0.0):
            msg = "source energy must be finite and non-negative"
            raise ValueError(msg)

        flow = config.flow_rate * config.dt
        moved = (1.0 - flow) * self._state
        moved[:, :, 1:] += flow * self._state[:, :, :-1]

        diffusion = config.diffusion_rate * config.dt
        mixed = moved.copy()
        vertical_exchange = diffusion * (moved[:, 1:, :] - moved[:, :-1, :])
        mixed[:, :-1, :] += vertical_exchange
        mixed[:, 1:, :] -= vertical_exchange

        self._state = exp(-config.decay_rate * config.dt) * mixed
        self._state[:, :, 0] += np.einsum("cb,cy->by", source, self._apertures)
        return self.state
