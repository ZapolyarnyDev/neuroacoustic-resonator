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
    metabolite_diffusion: float = 0.0
    trace_rate: float = 0.08
    frequency_plasticity_rate: float = 0.01
    min_frequency: float = 0.2
    max_frequency: float = 3.0
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
        if self.metabolite_diffusion < 0.0:
            msg = "metabolite_diffusion must be non-negative"
            raise ValueError(msg)
        if self.trace_rate < 0.0:
            msg = "trace_rate must be non-negative"
            raise ValueError(msg)
        if self.frequency_plasticity_rate < 0.0:
            msg = "frequency_plasticity_rate must be non-negative"
            raise ValueError(msg)
        if self.min_frequency <= 0.0:
            msg = "min_frequency must be positive"
            raise ValueError(msg)
        if self.max_frequency <= self.min_frequency:
            msg = "max_frequency must be greater than min_frequency"
            raise ValueError(msg)


@dataclass(frozen=True)
class FieldState:
    phase: FloatArray
    frequency: FloatArray
    metabolite: FloatArray
    coupling: FloatArray
    trace: FloatArray


@dataclass(frozen=True)
class FieldMetrics:
    step: int
    mean_metabolite: float
    min_metabolite: float
    mean_trace: float
    max_trace: float
    global_synchrony: float
    mean_local_synchrony: float
    max_local_synchrony: float


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
        self._trace = np.zeros(shape, dtype=np.float64)

    @classmethod
    def from_state(cls, config: FieldConfig, state: FieldState) -> OscillatorField:
        expected_shape = (config.size, config.size)
        for name, value in (
            ("phase", state.phase),
            ("frequency", state.frequency),
            ("metabolite", state.metabolite),
            ("coupling", state.coupling),
            ("trace", state.trace),
        ):
            if value.shape != expected_shape:
                msg = f"{name} shape must be {expected_shape}"
                raise ValueError(msg)

        field = cls.__new__(cls)
        field.config = config
        field._phase = np.mod(state.phase, TAU).astype(np.float64, copy=True)
        field._frequency = state.frequency.astype(np.float64, copy=True)
        field._metabolite = np.clip(state.metabolite, 0.0, 1.0).astype(
            np.float64, copy=True
        )
        field._coupling = state.coupling.astype(np.float64, copy=True)
        field._trace = np.clip(state.trace, 0.0, None).astype(np.float64, copy=True)
        return field

    @property
    def state(self) -> FieldState:
        return FieldState(
            phase=self._phase.copy(),
            frequency=self._frequency.copy(),
            metabolite=self._metabolite.copy(),
            coupling=self._coupling.copy(),
            trace=self._trace.copy(),
        )

    def step(self) -> FieldState:
        coupling_drive = self._coupling_drive()
        phase_drive = self._frequency + coupling_drive
        activity = np.abs(phase_drive)
        metabolite_laplacian = self._metabolite_laplacian()

        self._phase = np.mod(self._phase + self.config.dt * phase_drive, TAU)
        self._metabolite = np.clip(
            self._metabolite
            + self.config.dt
            * (
                self.config.metabolite_recovery * (1.0 - self._metabolite)
                - self.config.metabolite_cost * activity
                + self.config.metabolite_diffusion * metabolite_laplacian
            ),
            0.0,
            1.0,
        )
        self._trace += (
            self.config.dt * self.config.trace_rate * (activity - self._trace)
        )
        self._frequency = np.clip(
            self._frequency
            + self.config.dt
            * self.config.frequency_plasticity_rate
            * self._trace
            * coupling_drive,
            self.config.min_frequency,
            self.config.max_frequency,
        )

        return self.state

    def local_synchrony(self) -> FloatArray:
        neighbor_vectors = (
            np.exp(1j * np.roll(self._phase, 1, axis=0))
            + np.exp(1j * np.roll(self._phase, -1, axis=0))
            + np.exp(1j * np.roll(self._phase, 1, axis=1))
            + np.exp(1j * np.roll(self._phase, -1, axis=1))
        )
        return np.abs((np.exp(1j * self._phase) + neighbor_vectors) / 5.0)

    def global_synchrony(self) -> float:
        return float(np.abs(np.mean(np.exp(1j * self._phase))))

    def metrics(self, step: int = 0) -> FieldMetrics:
        local = self.local_synchrony()
        return FieldMetrics(
            step=step,
            mean_metabolite=float(np.mean(self._metabolite)),
            min_metabolite=float(np.min(self._metabolite)),
            mean_trace=float(np.mean(self._trace)),
            max_trace=float(np.max(self._trace)),
            global_synchrony=self.global_synchrony(),
            mean_local_synchrony=float(np.mean(local)),
            max_local_synchrony=float(np.max(local)),
        )

    def _coupling_drive(self) -> FloatArray:
        neighbors = (
            np.sin(np.roll(self._phase, 1, axis=0) - self._phase)
            + np.sin(np.roll(self._phase, -1, axis=0) - self._phase)
            + np.sin(np.roll(self._phase, 1, axis=1) - self._phase)
            + np.sin(np.roll(self._phase, -1, axis=1) - self._phase)
        )
        return self._metabolite * self._coupling * neighbors / 4.0

    def _metabolite_laplacian(self) -> FloatArray:
        return (
            np.roll(self._metabolite, 1, axis=0)
            + np.roll(self._metabolite, -1, axis=0)
            + np.roll(self._metabolite, 1, axis=1)
            + np.roll(self._metabolite, -1, axis=1)
            - 4.0 * self._metabolite
        )
