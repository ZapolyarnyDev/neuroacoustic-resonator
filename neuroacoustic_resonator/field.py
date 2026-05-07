from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

TAU = 2.0 * np.pi


@dataclass(frozen=True)
class FieldConfig:
    size: int = 64
    dt: float = 0.02
    base_frequency: float = 1.0
    frequency_spread: float = 0.05
    coupling_strength: float = 0.16
    metabolite_recovery: float = 0.03
    metabolite_cost: float = 0.02
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.size <= 1:
            msg = "size must be greater than 1"
            raise ValueError(msg)
        if self.dt <= 0.0:
            msg = "dt must be positive"
            raise ValueError(msg)
        if self.frequency_spread < 0.0:
            msg = "frequency_spread must be non-negative"
            raise ValueError(msg)
        if self.coupling_strength < 0.0:
            msg = "coupling_strength must be non-negative"
            raise ValueError(msg)
        if self.metabolite_recovery < 0.0:
            msg = "metabolite_recovery must be non-negative"
            raise ValueError(msg)
        if self.metabolite_cost < 0.0:
            msg = "metabolite_cost must be non-negative"
            raise ValueError(msg)


@dataclass(frozen=True)
class FieldState:
    phase: FloatArray
    frequency: FloatArray
    metabolite: FloatArray
    coupling: FloatArray


class OscillatorField:
    def __init__(self, config: FieldConfig) -> None:
        self.config = config
        rng = np.random.default_rng(config.seed)
        shape = (config.size, config.size)

        self._phase = rng.uniform(0.0, TAU, size=shape).astype(np.float64)
        self._frequency = rng.normal(
            loc=config.base_frequency,
            scale=config.frequency_spread,
            size=shape,
        ).astype(np.float64)
        self._metabolite = np.ones(shape, dtype=np.float64)
        self._coupling = np.full(shape, config.coupling_strength, dtype=np.float64)

    @property
    def state(self) -> FieldState:
        return FieldState(
            phase=self._phase.copy(),
            frequency=self._frequency.copy(),
            metabolite=self._metabolite.copy(),
            coupling=self._coupling.copy(),
        )

    def step(self) -> FieldState:
        phase_drive = self._frequency + self._coupling_drive()
        activity = np.abs(phase_drive)

        self._phase = np.mod(self._phase + self.config.dt * phase_drive, TAU)
        self._metabolite = np.clip(
            self._metabolite
            + self.config.dt
            * (
                self.config.metabolite_recovery * (1.0 - self._metabolite)
                - self.config.metabolite_cost * activity
            ),
            0.0,
            1.0,
        )

        return self.state

    def _coupling_drive(self) -> FloatArray:
        neighbors = (
            np.sin(np.roll(self._phase, 1, axis=0) - self._phase)
            + np.sin(np.roll(self._phase, -1, axis=0) - self._phase)
            + np.sin(np.roll(self._phase, 1, axis=1) - self._phase)
            + np.sin(np.roll(self._phase, -1, axis=1) - self._phase)
        )
        return self._metabolite * self._coupling * neighbors / 4.0
