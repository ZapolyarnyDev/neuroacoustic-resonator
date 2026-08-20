from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Self

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from neuroacoustic_resonator.analysis.distinguishability_corpus import (
    CorpusSplitsConfig,
    CorpusStimulusConfig,
)
from neuroacoustic_resonator.analysis.distinguishability_diagnostics import (
    diagnose_protocol_embeddings,
)
from neuroacoustic_resonator.analysis.paired_causal_evidence import (
    evaluate_paired_causal_evidence,
)
from neuroacoustic_resonator.analysis.pattern_calibration import (
    CalibrationTrial,
    PatternCalibrationConfig,
    SyntheticStimulusSpec,
    calibration_manifest_entry,
    derive_trial_seed,
    materialize_synthetic_stimuli,
    run_calibration_trial,
    write_calibration_manifest,
)
from neuroacoustic_resonator.analysis.protocol_embeddings import (
    EVENT_FEATURES,
    FEATURE_COLUMNS,
    SIGNAL_GETTERS,
    EmbeddingRow,
    response_embedding,
    select_response_frames,
    temporal_statistics,
    write_embedding_rows,
)
from neuroacoustic_resonator.configuration import SimulationConfig
from neuroacoustic_resonator.core.equilibration import (
    FieldEquilibrationConfig,
    equilibrate_simulation,
)
from neuroacoustic_resonator.io.persistence import (
    checkpoint_fingerprint,
    load_checkpoint_metadata,
    save_simulation_checkpoint,
)
from neuroacoustic_resonator.protocol import SoundProtocolFrame, read_protocol_jsonl


class EquilibrationConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    neutral_steps: int = Field(default=512, ge=0)
    phase_damping: float = Field(default=0.2, ge=0.0, le=1.0)
    metabolite_baseline: float | None = Field(default=1.0, ge=0.0, le=1.0)

    def to_runtime(self) -> FieldEquilibrationConfig:
        return FieldEquilibrationConfig(**self.model_dump())


class ControlledEquilibrationCorpusConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    simulation_config: Path
    output_dir: Path
    output_manifest: Path
    output_pairs: Path
    output_embeddings: Path
    output_summary: Path
    seed_roots: tuple[int, ...] = Field(min_length=1)
    splits: CorpusSplitsConfig
    repeats: int = Field(default=2, ge=1)
    sample_rate: int = Field(default=8_000, ge=1)
    output_frame_size: int = Field(default=256, ge=1)
    input_frame_size: int = Field(default=256, ge=1)
    input_hop_size: int = Field(default=128, ge=1)
    drive_strength: float = Field(default=0.45, ge=0.0)
    input_assoc_gain: float = Field(default=0.8, ge=0.0)
    input_output_gain: float = Field(default=0.0, ge=0.0)
    response_seconds: float = Field(default=0.35, gt=0.0)
    gain: float = Field(default=0.35, ge=0.0)
    equilibration: EquilibrationConfigModel = Field(
        default_factory=EquilibrationConfigModel
    )
    stimuli: tuple[CorpusStimulusConfig, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def validate_experiment(self) -> Self:
        labels = [stimulus.label for stimulus in self.stimuli]
        kinds = [stimulus.kind for stimulus in self.stimuli]
        if len(set(labels)) != len(labels):
            msg = "stimulus labels must be unique"
            raise ValueError(msg)
        if kinds.count("silence") != 1:
            msg = "controlled corpus requires exactly one silence control"
            raise ValueError(msg)
        split_roots = [item.seed_root for item in self.splits.assignments()]
        if set(split_roots) != set(self.seed_roots):
            msg = "splits must assign every seed root exactly once"
            raise ValueError(msg)
        if len(set(self.seed_roots)) != len(self.seed_roots):
            msg = "seed_roots must be unique"
            raise ValueError(msg)
        return self

    @classmethod
    def from_file(cls, path: str | Path) -> ControlledEquilibrationCorpusConfig:
        with Path(path).open(encoding="utf-8") as stream:
            value = yaml.safe_load(stream) or {}
        return cls.model_validate(value)

    def calibration_config(self) -> PatternCalibrationConfig:
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
            output_csv=self.output_dir / "branches.csv",
            output_summary=self.output_dir / "branches.json",
            output_manifest=self.output_manifest,
            seed_roots=self.seed_roots,
            seed_splits=self.splits.assignments(),
            repeats=self.repeats,
            sample_rate=self.sample_rate,
            output_frame_size=self.output_frame_size,
            input_frame_size=self.input_frame_size,
            input_hop_size=self.input_hop_size,
            drive_strength=self.drive_strength,
            input_assoc_gain=self.input_assoc_gain,
            input_output_gain=self.input_output_gain,
            response_seconds=self.response_seconds,
            warmup_steps=0,
            gain=self.gain,
        )


def run_controlled_equilibration_corpus(
    config_path: str | Path,
    *,
    permutation_samples: int = 10_000,
    bootstrap_samples: int = 5_000,
) -> dict[str, Any]:
    config = ControlledEquilibrationCorpusConfig.from_file(config_path)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    calibration = config.calibration_config()
    stimuli = materialize_synthetic_stimuli(calibration)
    splits = {item.seed_root: item.split for item in config.splits.assignments()}
    checkpoints = create_equilibrated_checkpoints(config, splits)
    manifest: list[dict[str, Any]] = []
    by_checkpoint: dict[str, dict[str, dict[str, Any]]] = {}
    for stimulus in stimuli:
        for seed_root in config.seed_roots:
            for repeat_index in range(1, config.repeats + 1):
                checkpoint = checkpoints[(seed_root, repeat_index)]
                trial = CalibrationTrial(
                    trial_id=(
                        f"{safe_label(stimulus.label)}-s{seed_root}-r{repeat_index:02d}"
                    ),
                    stimulus=stimulus,
                    seed_root=seed_root,
                    field_seed=derive_trial_seed(seed_root, repeat_index),
                    repeat_index=repeat_index,
                    split=splits[seed_root],
                )
                run_calibration_trial(
                    calibration,
                    trial,
                    initial_checkpoint=checkpoint["path"],
                    expected_checkpoint_fingerprint=checkpoint["fingerprint"],
                )
                entry = {
                    **calibration_manifest_entry(trial, calibration),
                    "checkpoint": str(checkpoint["path"]),
                    "checkpoint_fingerprint": checkpoint["fingerprint"],
                }
                manifest.append(entry)
                by_checkpoint.setdefault(checkpoint["fingerprint"], {})[
                    stimulus.label
                ] = entry
    write_calibration_manifest(config.output_manifest, manifest)
    pairs = build_pair_manifest(config, by_checkpoint)
    write_json(config.output_pairs, pairs)
    extract_causal_embeddings(pairs, config.output_embeddings)
    active_manifest = config.output_dir / "active_branches_manifest.json"
    write_json(
        active_manifest,
        [entry for entry in manifest if entry["source_type"] != "silence"],
    )
    from neuroacoustic_resonator.analysis.protocol_embeddings import (
        extract_protocol_embeddings,
    )

    absolute_embeddings = config.output_dir / "absolute_embeddings.csv"
    extract_protocol_embeddings(
        active_manifest,
        absolute_embeddings,
        config.output_dir / "absolute_embedding_schema.json",
    )
    absolute_diagnostics = diagnose_protocol_embeddings(
        absolute_embeddings,
        config.output_dir / "absolute_diagnostics.json",
        config.output_dir / "absolute_diagnostic_features.csv",
        config.output_dir / "absolute_diagnostic_distances.csv",
        config.output_dir / "absolute_diagnostics.png",
        permutation_samples=permutation_samples,
    )
    causal_diagnostics = diagnose_protocol_embeddings(
        config.output_embeddings,
        config.output_dir / "causal_diagnostics.json",
        config.output_dir / "causal_diagnostic_features.csv",
        config.output_dir / "causal_diagnostic_distances.csv",
        config.output_dir / "causal_diagnostics.png",
        permutation_samples=permutation_samples,
    )
    evidence = evaluate_paired_causal_evidence(
        config.output_embeddings,
        config.output_dir / "causal_evidence.json",
        permutation_samples=permutation_samples,
        permutation_seed=20_260_818,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=20_260_819,
    )
    absolute_variance = absolute_diagnostics["variance"]["aggregate"]
    causal_variance = causal_diagnostics["variance"]["aggregate"]
    causal_distances = causal_diagnostics["distances"]
    report = {
        "config": str(config_path),
        "branch_trials": len(manifest),
        "causal_pairs": len(pairs),
        "checkpoints": len(checkpoints),
        "fingerprints_verified": True,
        "absolute": absolute_variance,
        "causal_difference": causal_variance,
        "field_seed_variance_reduction_fraction": variance_reduction(
            absolute_variance["field_seed_fraction"],
            causal_variance["field_seed_fraction"],
        ),
        "stimulus_to_seed_ratio_gain": ratio_gain(
            absolute_variance["stimulus_to_seed_ratio"],
            causal_variance["stimulus_to_seed_ratio"],
        ),
        "cross_seed_classification": evidence["classification"],
        "cross_seed_distance": {
            "within_stimulus_mean": causal_distances["groups"][
                "cross_seed_within_stimulus"
            ]["mean"],
            "between_stimulus_mean": causal_distances["groups"][
                "cross_seed_between_stimulus"
            ]["mean"],
            "separation_margin": causal_distances["cross_seed_separation_margin"],
            "cliffs_delta": causal_distances["cross_seed_cliffs_delta"],
        },
        "paired_stimulus_permutation": causal_diagnostics["paired_permutation"],
        "outputs": {
            "manifest": str(config.output_manifest),
            "pairs": str(config.output_pairs),
            "embeddings": str(config.output_embeddings),
            "causal_evidence": str(config.output_dir / "causal_evidence.json"),
        },
    }
    write_json(config.output_summary, report)
    return report


def analyze_controlled_equilibration(
    config_path: str | Path,
    *,
    permutation_samples: int = 10_000,
    bootstrap_samples: int = 5_000,
) -> dict[str, Any]:
    config = ControlledEquilibrationCorpusConfig.from_file(config_path)
    evidence_path = config.output_dir / "causal_evidence.json"
    evidence = evaluate_paired_causal_evidence(
        config.output_embeddings,
        evidence_path,
        permutation_samples=permutation_samples,
        bootstrap_samples=bootstrap_samples,
    )
    if config.output_summary.exists():
        summary = read_json_object(config.output_summary)
        summary["cross_seed_classification"] = evidence["classification"]
        outputs = summary.setdefault("outputs", {})
        if not isinstance(outputs, dict):
            msg = "controlled equilibration summary outputs must be an object"
            raise ValueError(msg)
        outputs["causal_evidence"] = str(evidence_path)
        write_json(config.output_summary, summary)
    return evidence


def create_equilibrated_checkpoints(
    config: ControlledEquilibrationCorpusConfig,
    splits: Mapping[int, str],
) -> dict[tuple[int, int], dict[str, Any]]:
    base = SimulationConfig.from_file(config.simulation_config)
    checkpoint_dir = config.output_dir / "checkpoints"
    checkpoints: dict[tuple[int, int], dict[str, Any]] = {}
    for seed_root in config.seed_roots:
        for repeat_index in range(1, config.repeats + 1):
            field_seed = derive_trial_seed(seed_root, repeat_index)
            seeded = base.model_copy(
                update={
                    "field": base.field.model_copy(update={"seed": field_seed}),
                }
            )
            simulation = seeded.create_simulation()
            equilibration = equilibrate_simulation(
                simulation,
                config.equilibration.to_runtime(),
            )
            path = checkpoint_dir / f"s{seed_root}-r{repeat_index:02d}.npz"
            save_simulation_checkpoint(
                path,
                simulation,
                metadata={
                    "purpose": "controlled_equilibration_branch_point",
                    "seed_root": seed_root,
                    "field_seed": field_seed,
                    "repeat_index": repeat_index,
                    "split": splits[seed_root],
                    "equilibration": equilibration,
                },
            )
            fingerprint = checkpoint_fingerprint(path)
            checkpoints[(seed_root, repeat_index)] = {
                "path": path,
                "fingerprint": fingerprint,
            }
    return checkpoints


def build_pair_manifest(
    config: ControlledEquilibrationCorpusConfig,
    by_checkpoint: dict[str, dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    control_label = next(
        stimulus.label for stimulus in config.stimuli if stimulus.kind == "silence"
    )
    pairs: list[dict[str, Any]] = []
    for fingerprint, branches in sorted(by_checkpoint.items()):
        control = branches.get(control_label)
        if control is None:
            msg = f"checkpoint {fingerprint} has no silence control"
            raise ValueError(msg)
        for stimulus in config.stimuli:
            if stimulus.kind == "silence":
                continue
            branch = branches.get(stimulus.label)
            if branch is None:
                msg = f"checkpoint {fingerprint} has no {stimulus.label} branch"
                raise ValueError(msg)
            pair = {
                "pair_id": branch["trial_id"],
                "stimulus_label": branch["stimulus_label"],
                "source_type": branch["source_type"],
                "seed_root": branch["seed_root"],
                "field_seed": branch["field_seed"],
                "repeat_index": branch["repeat_index"],
                "split": branch["split"],
                "checkpoint": branch["checkpoint"],
                "checkpoint_fingerprint": fingerprint,
                "stimulus_metadata_json": branch["metadata_json"],
                "control_metadata_json": control["metadata_json"],
            }
            validate_pair_entry(pair)
            pairs.append(pair)
    return pairs


def validate_pair_entry(pair: dict[str, Any]) -> None:
    expected = require_string(pair, "checkpoint_fingerprint")
    checkpoint = Path(require_string(pair, "checkpoint"))
    if checkpoint_fingerprint(checkpoint) != expected:
        msg = "pair checkpoint fingerprint does not match checkpoint"
        raise ValueError(msg)
    checkpoint_metadata = load_checkpoint_metadata(checkpoint)
    if checkpoint_metadata.get("fingerprint") != expected:
        msg = "pair checkpoint metadata fingerprint mismatch"
        raise ValueError(msg)
    stimulus = read_json_object(pair["stimulus_metadata_json"])
    control = read_json_object(pair["control_metadata_json"])
    for metadata in (stimulus, control):
        if metadata.get("checkpoint_fingerprint") != expected:
            msg = "paired branch fingerprint mismatch"
            raise ValueError(msg)
        for key in ("seed_root", "field_seed", "repeat_index", "split"):
            if metadata.get(key) != pair.get(key):
                msg = f"paired branch mismatch for {key}"
                raise ValueError(msg)
    if control.get("source_type") != "silence":
        msg = "paired control branch must be silence"
        raise ValueError(msg)


def extract_causal_embeddings(
    pairs: list[dict[str, Any]],
    output_csv: str | Path,
) -> Path:
    rows = [causal_embedding_from_pair(pair) for pair in pairs]
    return write_embedding_rows(output_csv, rows)


def causal_embedding_from_pair(pair: dict[str, Any]) -> EmbeddingRow:
    validate_pair_entry(pair)
    stimulus_metadata = read_json_object(pair["stimulus_metadata_json"])
    control_metadata = read_json_object(pair["control_metadata_json"])
    stimulus_frames = response_frames(stimulus_metadata)
    control_frames = response_frames(control_metadata)
    if len(stimulus_frames) != len(control_frames):
        msg = "paired response branches must contain the same number of frames"
        raise ValueError(msg)
    row: EmbeddingRow = {
        "trial_id": require_string(pair, "pair_id"),
        "stimulus_label": require_string(pair, "stimulus_label"),
        "source_type": require_string(pair, "source_type"),
        "seed_root": require_int(pair, "seed_root"),
        "field_seed": require_int(pair, "field_seed"),
        "repeat_index": require_int(pair, "repeat_index"),
        "split": require_string(pair, "split"),
        "protocol_version": require_string(stimulus_metadata, "protocol_version"),
        "response_frames": len(stimulus_frames),
    }
    row.update(causal_response_embedding(stimulus_frames, control_frames))
    return row


def causal_response_embedding(
    stimulus_frames: list[SoundProtocolFrame],
    control_frames: list[SoundProtocolFrame],
) -> dict[str, float]:
    features: dict[str, float] = {}
    for signal_name, getter in SIGNAL_GETTERS.items():
        values = np.asarray(
            [
                getter(stimulus) - getter(control)
                for stimulus, control in zip(
                    stimulus_frames,
                    control_frames,
                    strict=True,
                )
            ],
            dtype=np.float64,
        )
        for statistic, value in temporal_statistics(values).items():
            features[f"{signal_name}_{statistic}"] = value
    stimulus_events = response_embedding(stimulus_frames)
    control_events = response_embedding(control_frames)
    for name in EVENT_FEATURES:
        features[name] = stimulus_events[name] - control_events[name]
    if set(features) != set(FEATURE_COLUMNS):
        msg = "causal response embedding schema mismatch"
        raise ValueError(msg)
    return features


def response_frames(metadata: dict[str, Any]) -> list[SoundProtocolFrame]:
    frames = read_protocol_jsonl(Path(require_string(metadata, "protocol_jsonl")))
    return select_response_frames(frames, metadata)


def variance_reduction(before: float, after: float) -> float:
    if before <= 0.0:
        return 0.0
    return 1.0 - after / before


def ratio_gain(before: float, after: float) -> float:
    if before <= 0.0:
        return float("inf") if after > 0.0 else 1.0
    return after / before


def read_json_object(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        msg = f"expected JSON object: {path}"
        raise ValueError(msg)
    return value


def require_string(value: dict[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        msg = f"{key} must be a non-empty string"
        raise ValueError(msg)
    return item


def require_int(value: dict[str, Any], key: str) -> int:
    item = value.get(key)
    if isinstance(item, bool) or not isinstance(item, int):
        msg = f"{key} must be an integer"
        raise ValueError(msg)
    return item


def safe_label(value: str) -> str:
    return "".join(character if character.isalnum() else "-" for character in value)


def write_json(path: str | Path, value: Any) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2), encoding="utf-8")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run paired stimulus/control branches from equilibrated checkpoints."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/controlled_equilibration.yaml"),
    )
    parser.add_argument("--permutation-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-samples", type=int, default=5_000)
    parser.add_argument("--analysis-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.analysis_only:
        report = analyze_controlled_equilibration(
            args.config,
            permutation_samples=args.permutation_samples,
            bootstrap_samples=args.bootstrap_samples,
        )
    else:
        report = run_controlled_equilibration_corpus(
            args.config,
            permutation_samples=args.permutation_samples,
            bootstrap_samples=args.bootstrap_samples,
        )
    print(json.dumps(report, indent=2))
    return 0
