from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from neuroacoustic_resonator.analysis.pattern_calibration import (
    PatternCalibrationConfig,
    SyntheticKind,
    SyntheticStimulusSpec,
    run_pattern_calibration,
)


class CorpusStimulusConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str = Field(min_length=1)
    kind: SyntheticKind
    frequency_hz: float = Field(default=220.0, gt=0.0)
    end_frequency_hz: float = Field(default=660.0, gt=0.0)
    duration_seconds: float = Field(default=0.5, gt=0.0)
    amplitude: float = Field(default=0.65, ge=0.0)


class DistinguishabilityCorpusConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    simulation_config: Path
    output_dir: Path
    output_csv: Path
    output_summary: Path
    output_manifest: Path
    seed_roots: tuple[int, ...] = Field(min_length=1)
    repeats: int = Field(ge=1)
    sample_rate: int = Field(default=8_000, ge=1)
    output_frame_size: int = Field(default=256, ge=1)
    input_frame_size: int = Field(default=256, ge=1)
    input_hop_size: int = Field(default=128, ge=1)
    drive_strength: float = Field(default=0.45, ge=0.0)
    input_assoc_gain: float = Field(default=0.8, ge=0.0)
    input_output_gain: float = Field(default=0.0, ge=0.0)
    response_seconds: float = Field(default=0.35, gt=0.0)
    warmup_steps: int = Field(default=16, ge=0)
    gain: float = Field(default=0.35, ge=0.0)
    stimuli: tuple[CorpusStimulusConfig, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_corpus(self) -> Self:
        required_kinds = {"silence", "tone", "chirp", "pulse", "noise"}
        kinds = [stimulus.kind for stimulus in self.stimuli]
        labels = [stimulus.label for stimulus in self.stimuli]
        if set(kinds) != required_kinds or len(kinds) != len(required_kinds):
            msg = "corpus must contain exactly silence, tone, chirp, pulse, and noise"
            raise ValueError(msg)
        if len(set(labels)) != len(labels):
            msg = "corpus stimulus labels must be unique"
            raise ValueError(msg)
        if len(set(self.seed_roots)) != len(self.seed_roots):
            msg = "corpus seed_roots must be unique"
            raise ValueError(msg)
        if any(seed < 0 for seed in self.seed_roots):
            msg = "corpus seed_roots must be non-negative"
            raise ValueError(msg)
        return self

    @classmethod
    def from_file(cls, path: str | Path) -> DistinguishabilityCorpusConfig:
        with Path(path).open(encoding="utf-8") as stream:
            data = yaml.safe_load(stream) or {}
        return cls.model_validate(data)

    def to_calibration_config(self) -> PatternCalibrationConfig:
        return PatternCalibrationConfig(
            config_path=self.simulation_config,
            synthetic_stimuli=tuple(
                SyntheticStimulusSpec(
                    label=stimulus.label,
                    kind=stimulus.kind,
                    frequency_hz=stimulus.frequency_hz,
                    end_frequency_hz=stimulus.end_frequency_hz,
                    duration_seconds=stimulus.duration_seconds,
                    amplitude=stimulus.amplitude,
                    sample_rate=self.sample_rate,
                )
                for stimulus in self.stimuli
            ),
            output_dir=self.output_dir,
            output_csv=self.output_csv,
            output_summary=self.output_summary,
            output_manifest=self.output_manifest,
            seed_roots=self.seed_roots,
            repeats=self.repeats,
            sample_rate=self.sample_rate,
            output_frame_size=self.output_frame_size,
            input_frame_size=self.input_frame_size,
            input_hop_size=self.input_hop_size,
            drive_strength=self.drive_strength,
            input_assoc_gain=self.input_assoc_gain,
            input_output_gain=self.input_output_gain,
            response_seconds=self.response_seconds,
            warmup_steps=self.warmup_steps,
            gain=self.gain,
        )


def run_seeded_distinguishability_corpus(
    config_path: str | Path,
) -> dict[str, object]:
    config = DistinguishabilityCorpusConfig.from_file(config_path)
    return run_pattern_calibration(config.to_calibration_config())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the seeded sound distinguishability corpus.",
    )
    parser.add_argument(
        "--corpus-config",
        type=Path,
        default=Path("configs") / "distinguishability_corpus.yaml",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_seeded_distinguishability_corpus(args.corpus_config)
    print(json.dumps(summary, indent=2))
    return 0
