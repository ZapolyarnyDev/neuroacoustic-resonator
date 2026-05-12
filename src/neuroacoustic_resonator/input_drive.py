from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from neuroacoustic_resonator.field import TAU, OscillatorField
from neuroacoustic_resonator.regions import RegionMasks

InputMode = Literal["sine", "pulse", "noise"]


@dataclass(frozen=True)
class SyntheticInputConfig:
    enabled: bool = False
    mode: InputMode = "sine"
    strength: float = 0.15
    period_steps: int = 64
    duty_cycle: float = 0.25
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.strength < 0.0:
            msg = "strength must be non-negative"
            raise ValueError(msg)
        if self.period_steps < 1:
            msg = "period_steps must be positive"
            raise ValueError(msg)
        if not 0.0 < self.duty_cycle <= 1.0:
            msg = "duty_cycle must be in (0, 1]"
            raise ValueError(msg)


class SyntheticInputDrive:
    def __init__(self, config: SyntheticInputConfig, regions: RegionMasks) -> None:
        self.config = config
        self.regions = regions
        self._rng = np.random.default_rng(config.seed)

    def value(self, step: int) -> float:
        if not self.config.enabled:
            return 0.0
        if step < 0:
            msg = "step must be non-negative"
            raise ValueError(msg)

        if self.config.mode == "sine":
            phase = TAU * (step % self.config.period_steps) / self.config.period_steps
            return self.config.strength * np.sin(phase)
        if self.config.mode == "pulse":
            active_steps = max(
                1, round(self.config.period_steps * self.config.duty_cycle)
            )
            return (
                self.config.strength
                if step % self.config.period_steps < active_steps
                else 0.0
            )
        if self.config.mode == "noise":
            return float(self._rng.normal(0.0, self.config.strength))

        msg = f"unsupported input mode: {self.config.mode}"
        raise ValueError(msg)

    def apply(self, field: OscillatorField, step: int) -> float:
        value = self.value(step)
        if value != 0.0:
            field.apply_phase_impulse(self.regions.input, value)
        return value
