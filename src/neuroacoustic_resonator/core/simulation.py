from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from neuroacoustic_resonator.core.config import SimulationConfig
from neuroacoustic_resonator.core.field import (
    FieldConfig,
    FieldMetrics,
    FieldState,
    FloatArray,
    OscillatorField,
)
from neuroacoustic_resonator.core.input_drive import (
    SyntheticInputConfig,
    SyntheticInputDrive,
)
from neuroacoustic_resonator.core.regions import RegionMasks


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
        synthetic_input: SyntheticInputConfig | None = None,
    ) -> None:
        if config is not None and field is not None:
            msg = "provide either config or field, not both"
            raise ValueError(msg)

        self.field = (
            field if field is not None else OscillatorField(config or FieldConfig())
        )
        regions = RegionMasks.from_size(self.field.config.size)
        self._configure_memory_drive_regions(regions)
        input_config = synthetic_input or SyntheticInputConfig()
        self.input_drive = SyntheticInputDrive(
            input_config,
            regions,
        )
        self.step_index = 0
        self.last_input_value = 0.0

    @classmethod
    def from_config(cls, config: SimulationConfig) -> Simulation:
        return cls(
            config=config.to_field_config(),
            synthetic_input=config.to_synthetic_input_config(),
        )

    @classmethod
    def from_config_file(cls, path: str | Path) -> Simulation:
        return cls.from_config(SimulationConfig.from_file(path))

    def _configure_memory_drive_regions(self, regions: RegionMasks) -> None:
        self.field.set_memory_drive_gain(
            regions.input,
            self.field.config.memory_drive_input_gain,
        )
        self.field.set_memory_drive_gain(
            regions.assoc,
            self.field.config.memory_drive_assoc_gain,
        )
        self.field.set_memory_drive_gain(
            regions.output,
            self.field.config.memory_drive_output_gain,
        )

    def snapshot(self) -> SimulationFrame:
        return SimulationFrame(
            state=self.field.state,
            metrics=self.field.metrics(step=self.step_index),
            local_synchrony=self.field.local_synchrony(),
        )

    def step(self) -> SimulationFrame:
        self.last_input_value = self.input_drive.apply(self.field, self.step_index)
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
