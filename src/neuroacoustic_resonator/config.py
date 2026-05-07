from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

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
    trace_rate: float = Field(default=0.08, ge=0.0)
    seed: int | None = None

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
