from __future__ import annotations

from dataclasses import dataclass

from neuroacoustic_resonator.field import (
    FieldConfig,
    FieldMetrics,
    FieldState,
    FloatArray,
    OscillatorField,
)


@dataclass(frozen=True)
class SimulationFrame:
    state: FieldState
    metrics: FieldMetrics
    local_synchrony: FloatArray


class Simulation:
    def __init__(
        self,
        config: FieldConfig | None = None,
        field: OscillatorField | None = None,
    ) -> None:
        if config is not None and field is not None:
            msg = "provide either config or field, not both"
            raise ValueError(msg)

        self.field = (
            field if field is not None else OscillatorField(config or FieldConfig())
        )
        self.step_index = 0

    def snapshot(self) -> SimulationFrame:
        return SimulationFrame(
            state=self.field.state,
            metrics=self.field.metrics(step=self.step_index),
            local_synchrony=self.field.local_synchrony(),
        )

    def step(self) -> SimulationFrame:
        self.step_index += 1
        state = self.field.step()
        return SimulationFrame(
            state=state,
            metrics=self.field.metrics(step=self.step_index),
            local_synchrony=self.field.local_synchrony(),
        )

    def run(self, steps: int) -> list[SimulationFrame]:
        if steps < 0:
            msg = "steps must be non-negative"
            raise ValueError(msg)
        return [self.step() for _ in range(steps)]
