from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from neuroacoustic_resonator.core.field import FieldConfig
from neuroacoustic_resonator.core.input_drive import InputMode, SyntheticInputConfig
from neuroacoustic_resonator.core.simulation import Simulation
from neuroacoustic_resonator.core.topology import BoundaryMode

if TYPE_CHECKING:
    from neuroacoustic_resonator.analysis.pattern_detector import (
        PatternDetectorConfig,
    )
    from neuroacoustic_resonator.encoding import ProtocolEncoderConfig


class FieldConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    size: int = Field(default=64, gt=1)
    dt: float = Field(default=0.02, gt=0.0)
    base_frequency: float = 1.0
    frequency_spread: float = Field(default=0.05, ge=0.0)
    coupling_strength: float = Field(default=0.16, ge=0.0)
    metabolite_recovery: float = Field(default=0.03, ge=0.0)
    metabolite_cost: float = Field(default=0.02, ge=0.0)
    metabolite_diffusion: float = Field(default=0.0, ge=0.0)
    metabolite_adaptive_recovery: float = Field(default=0.0, ge=0.0)
    trace_rate: float = Field(default=0.08, ge=0.0)
    trace_decay: float = Field(default=0.0, ge=0.0)
    memory_drive_strength: float = Field(default=0.0, ge=0.0)
    memory_drive_input_gain: float = Field(default=1.0, ge=0.0)
    memory_drive_assoc_gain: float = Field(default=1.0, ge=0.0)
    memory_drive_output_gain: float = Field(default=1.0, ge=0.0)
    frequency_plasticity_rate: float = Field(default=0.01, ge=0.0)
    frequency_homeostasis_rate: float = Field(default=0.01, ge=0.0)
    synchrony_target_low: float = Field(default=0.05, ge=0.0, le=1.0)
    synchrony_target_high: float = Field(default=0.8, ge=0.0, le=1.0)
    coupling_homeostasis_rate: float = Field(default=0.05, ge=0.0)
    min_coupling: float = Field(default=0.0, ge=0.0)
    max_coupling: float = Field(default=1.0, gt=0.0)
    min_frequency: float = Field(default=0.2, gt=0.0)
    max_frequency: float = Field(default=3.0, gt=0.0)
    boundary_x: BoundaryMode = "periodic"
    boundary_y: BoundaryMode = "periodic"
    seed: int | None = None

    @model_validator(mode="after")
    def validate_ranges(self) -> Self:
        if self.synchrony_target_high <= self.synchrony_target_low:
            msg = "synchrony_target_high must be greater than synchrony_target_low"
            raise ValueError(msg)
        if self.max_coupling <= self.min_coupling:
            msg = "max_coupling must be greater than min_coupling"
            raise ValueError(msg)
        if self.max_frequency <= self.min_frequency:
            msg = "max_frequency must be greater than min_frequency"
            raise ValueError(msg)
        return self

    def to_runtime(self) -> FieldConfig:
        return FieldConfig(**self.model_dump())


class SyntheticInputConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    mode: InputMode = "sine"
    strength: float = Field(default=0.15, ge=0.0)
    period_steps: int = Field(default=64, ge=1)
    duty_cycle: float = Field(default=0.25, gt=0.0, le=1.0)
    seed: int | None = None

    def to_runtime(self) -> SyntheticInputConfig:
        return SyntheticInputConfig(**self.model_dump())


class ProtocolConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frame_interval_steps: int = Field(default=1, ge=1)
    activity_threshold: float = Field(default=0.025, gt=0.0, le=1.0)
    confidence_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    confirmation_frames: int = Field(default=3, ge=1)
    minimum_active_frames: int = Field(default=3, ge=1)
    hysteresis_margin: float = Field(default=0.08, ge=0.0, le=1.0)
    novelty_threshold: float = Field(default=0.15, ge=0.0, le=1.0)

    def to_encoder_config(self) -> ProtocolEncoderConfig:
        from neuroacoustic_resonator.encoding import ProtocolEncoderConfig

        return ProtocolEncoderConfig(
            frame_interval_steps=self.frame_interval_steps,
        )

    def to_detector_config(self) -> PatternDetectorConfig:
        from neuroacoustic_resonator.analysis.pattern_detector import (
            PatternDetectorConfig,
        )

        return PatternDetectorConfig(
            activity_threshold=self.activity_threshold,
            confidence_threshold=self.confidence_threshold,
            confirmation_frames=self.confirmation_frames,
            minimum_active_frames=self.minimum_active_frames,
            hysteresis_margin=self.hysteresis_margin,
            novelty_threshold=self.novelty_threshold,
        )


class SimulationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: FieldConfigModel = Field(default_factory=FieldConfigModel)
    synthetic_input: SyntheticInputConfigModel = Field(
        default_factory=lambda: SyntheticInputConfigModel(enabled=False)
    )
    protocol: ProtocolConfigModel = Field(default_factory=ProtocolConfigModel)
    steps: int = Field(default=32, ge=1)
    preview_path: Path = Path("outputs/field-preview.png")

    def create_simulation(self) -> Simulation:
        return Simulation(
            config=self.field.to_runtime(),
            synthetic_input=self.synthetic_input.to_runtime(),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> SimulationConfig:
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream) or {}
        return cls.model_validate(data)
