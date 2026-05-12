from __future__ import annotations

from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from neuroacoustic_resonator.field import FieldConfig


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
    trace_rate: float = Field(default=0.08, ge=0.0)
    frequency_plasticity_rate: float = Field(default=0.01, ge=0.0)
    frequency_homeostasis_rate: float = Field(default=0.01, ge=0.0)
    synchrony_target_low: float = Field(default=0.05, ge=0.0, le=1.0)
    synchrony_target_high: float = Field(default=0.8, ge=0.0, le=1.0)
    coupling_homeostasis_rate: float = Field(default=0.05, ge=0.0)
    min_coupling: float = Field(default=0.0, ge=0.0)
    max_coupling: float = Field(default=1.0, gt=0.0)
    min_frequency: float = Field(default=0.2, gt=0.0)
    max_frequency: float = Field(default=3.0, gt=0.0)
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

    def to_field_config(self) -> FieldConfig:
        return FieldConfig(**self.model_dump())


class SimulationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: FieldConfigModel = Field(default_factory=FieldConfigModel)
    steps: int = Field(default=32, ge=1)
    preview_path: Path = Path("outputs/field-preview.png")

    def to_field_config(self) -> FieldConfig:
        return self.field.to_field_config()

    @classmethod
    def from_file(cls, path: str | Path) -> SimulationConfig:
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream) or {}
        return cls.model_validate(data)
