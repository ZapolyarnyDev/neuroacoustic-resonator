from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np

from neuroacoustic_resonator.analysis.distinguishability_diagnostics import (
    diagnose_protocol_embeddings,
)
from neuroacoustic_resonator.analysis.paired_causal_evidence import (
    evaluate_paired_causal_evidence,
)
from neuroacoustic_resonator.analysis.protocol_embeddings import (
    IDENTITY_COLUMNS,
    EmbeddingRow,
    read_embedding_rows,
    select_response_frames,
    write_embedding_rows,
)
from neuroacoustic_resonator.protocol import (
    RegionSnapshot,
    SoundProtocolFrame,
    read_protocol_jsonl,
)

RegionName = Literal["input", "assoc", "output"]

REGION_NAMES: tuple[RegionName, ...] = ("input", "assoc", "output")
REGION_SIGNAL_ATTRIBUTES = (
    "phase_coherence",
    "mean_local_synchrony",
    "mean_metabolite",
    "min_metabolite",
    "mean_trace",
    "max_trace",
    "mean_frequency",
    "frequency_spread",
    "mean_coupling",
    "coupling_spread",
)
CAUSAL_TEMPORAL_STATISTICS = (
    "mean",
    "std",
    "peak_absolute",
    "peak_time_fraction",
    "onset_time_fraction",
    "delta",
    "trend",
    "signed_auc",
    "absolute_auc",
    "retention",
)


def regional_feature_columns(regions: Sequence[RegionName]) -> tuple[str, ...]:
    return tuple(
        f"{region}_{signal}_{statistic}"
        for region in regions
        for signal in REGION_SIGNAL_ATTRIBUTES
        for statistic in CAUSAL_TEMPORAL_STATISTICS
    )


def causal_temporal_statistics(values: np.ndarray) -> dict[str, float]:
    if values.ndim != 1 or values.size == 0:
        msg = "causal temporal values must be a non-empty vector"
        raise ValueError(msg)
    magnitudes = np.abs(values)
    peak_index = int(np.argmax(magnitudes))
    peak = float(magnitudes[peak_index])
    if values.size == 1:
        time = np.asarray([0.0])
        trend = 0.0
        signed_auc = float(values[0])
        absolute_auc = peak
    else:
        time = np.linspace(0.0, 1.0, values.size)
        trend = float(np.polyfit(time, values, deg=1)[0])
        signed_auc = float(np.trapezoid(values, time))
        absolute_auc = float(np.trapezoid(magnitudes, time))
    if peak <= 1e-12:
        onset = 1.0
        retention = 0.0
    else:
        onset = float(time[int(np.flatnonzero(magnitudes >= peak * 0.2)[0])])
        retention = float(magnitudes[-1] / peak)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "peak_absolute": peak,
        "peak_time_fraction": float(time[peak_index]),
        "onset_time_fraction": onset,
        "delta": float(values[-1] - values[0]),
        "trend": trend,
        "signed_auc": signed_auc,
        "absolute_auc": absolute_auc,
        "retention": retention,
    }


def regional_causal_embedding(
    stimulus_frames: Sequence[SoundProtocolFrame],
    control_frames: Sequence[SoundProtocolFrame],
    regions: Sequence[RegionName],
) -> dict[str, float]:
    if not stimulus_frames or len(stimulus_frames) != len(control_frames):
        msg = "paired regional trajectories must be non-empty and equally sized"
        raise ValueError(msg)
    features: dict[str, float] = {}
    for region in regions:
        for signal in REGION_SIGNAL_ATTRIBUTES:
            values = np.asarray(
                [
                    getattr(region_snapshot(stimulus, region), signal)
                    - getattr(region_snapshot(control, region), signal)
                    for stimulus, control in zip(
                        stimulus_frames,
                        control_frames,
                        strict=True,
                    )
                ],
                dtype=np.float64,
            )
            for statistic, value in causal_temporal_statistics(values).items():
                features[f"{region}_{signal}_{statistic}"] = value
    if tuple(features) != regional_feature_columns(regions):
        msg = "regional causal embedding schema mismatch"
        raise ValueError(msg)
    return features


def extract_regional_causal_embeddings(
    pairs: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
) -> dict[str, Path]:
    if not pairs:
        msg = "regional causal extraction requires paired branches"
        raise ValueError(msg)
    output = Path(output_dir)
    frames_by_pair = [paired_response_frames(pair) for pair in pairs]
    paths: dict[str, Path] = {}
    region_sets: dict[str, tuple[RegionName, ...]] = {
        region: (region,) for region in REGION_NAMES
    }
    region_sets["combined"] = REGION_NAMES
    for name, regions in region_sets.items():
        rows: list[EmbeddingRow] = []
        for pair, (metadata, stimulus_frames, control_frames) in zip(
            pairs,
            frames_by_pair,
            strict=True,
        ):
            row = pair_identity(pair, metadata, len(stimulus_frames))
            row.update(
                regional_causal_embedding(stimulus_frames, control_frames, regions)
            )
            rows.append(row)
        path = output / f"causal_{name}_embeddings.csv"
        write_embedding_rows(
            path,
            rows,
            feature_columns=regional_feature_columns(regions),
        )
        paths[name] = path
    return paths


def analyze_regional_causal_representations(
    pairs: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    output_pattern_embeddings: str | Path,
    *,
    permutation_samples: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    output = Path(output_dir)
    paths = extract_regional_causal_embeddings(pairs, output)
    representation_paths = {
        "output_pattern": Path(output_pattern_embeddings),
        **paths,
    }
    representations: dict[str, dict[str, Any]] = {}
    for name, embeddings in representation_paths.items():
        prefix = output / f"causal_{name}"
        diagnostics = diagnose_protocol_embeddings(
            embeddings,
            prefix.with_name(f"{prefix.name}_diagnostics.json"),
            prefix.with_name(f"{prefix.name}_diagnostic_features.csv"),
            prefix.with_name(f"{prefix.name}_diagnostic_distances.csv"),
            prefix.with_name(f"{prefix.name}_diagnostics.png"),
            permutation_samples=permutation_samples,
        )
        evidence = evaluate_paired_causal_evidence(
            embeddings,
            prefix.with_name(f"{prefix.name}_evidence.json"),
            permutation_samples=permutation_samples,
            bootstrap_samples=bootstrap_samples,
        )
        variance = diagnostics["variance"]["aggregate"]
        distances = diagnostics["distances"]
        rows = read_embedding_rows(embeddings)
        representations[name] = {
            "embeddings": str(embeddings),
            "feature_count": len(rows[0]) - len(IDENTITY_COLUMNS),
            "stimulus_fraction": variance["stimulus_fraction"],
            "field_seed_fraction": variance["field_seed_fraction"],
            "stimulus_to_seed_ratio": variance["stimulus_to_seed_ratio"],
            "balanced_accuracy": evidence["classification"]["balanced_accuracy"],
            "chance_level": evidence["classification"]["chance_level"],
            "roots_above_chance": evidence["classification"]["roots_above_chance"],
            "bootstrap_ci": evidence["classification"][
                "checkpoint_clustered_bootstrap"
            ]["balanced_accuracy_ci"],
            "permutation_p_value": evidence["classification"][
                "checkpoint_label_permutation"
            ]["p_value"],
            "within_stimulus_mean": distances["groups"]["cross_seed_within_stimulus"][
                "mean"
            ],
            "between_stimulus_mean": distances["groups"]["cross_seed_between_stimulus"][
                "mean"
            ],
            "separation_margin": distances["cross_seed_separation_margin"],
            "cliffs_delta": distances["cross_seed_cliffs_delta"],
            "top_stimulus_features": diagnostics["variance"]["top_stimulus_features"],
        }
    report = {
        "design": {
            "causal_pair_count": len(pairs),
            "regions": list(REGION_NAMES),
            "signals_per_region": len(REGION_SIGNAL_ATTRIBUTES),
            "temporal_statistics": list(CAUSAL_TEMPORAL_STATISTICS),
            "protocol_schema_changed": False,
        },
        "representations": representations,
    }
    write_json(output / "regional_causal_report.json", report)
    return report


def paired_response_frames(
    pair: Mapping[str, Any],
) -> tuple[dict[str, Any], list[SoundProtocolFrame], list[SoundProtocolFrame]]:
    stimulus_metadata = read_json_object(require_string(pair, "stimulus_metadata_json"))
    control_metadata = read_json_object(require_string(pair, "control_metadata_json"))
    stimulus_frames = response_frames(stimulus_metadata)
    control_frames = response_frames(control_metadata)
    if len(stimulus_frames) != len(control_frames):
        msg = "paired response branches must contain the same number of frames"
        raise ValueError(msg)
    return stimulus_metadata, stimulus_frames, control_frames


def pair_identity(
    pair: Mapping[str, Any],
    metadata: Mapping[str, Any],
    response_frame_count: int,
) -> EmbeddingRow:
    return {
        "trial_id": require_string(pair, "pair_id"),
        "stimulus_label": require_string(pair, "stimulus_label"),
        "source_type": require_string(pair, "source_type"),
        "seed_root": require_int(pair, "seed_root"),
        "field_seed": require_int(pair, "field_seed"),
        "repeat_index": require_int(pair, "repeat_index"),
        "split": require_string(pair, "split"),
        "protocol_version": require_string(metadata, "protocol_version"),
        "response_frames": response_frame_count,
    }


def response_frames(metadata: Mapping[str, Any]) -> list[SoundProtocolFrame]:
    frames = read_protocol_jsonl(Path(require_string(metadata, "protocol_jsonl")))
    return select_response_frames(frames, dict(metadata))


def region_snapshot(frame: SoundProtocolFrame, region: RegionName) -> RegionSnapshot:
    if region == "input":
        return frame.input_region
    if region == "assoc":
        return frame.assoc_region
    return frame.output_region


def read_json_object(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        msg = f"expected JSON object in {path}"
        raise ValueError(msg)
    return value


def require_string(value: Mapping[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        msg = f"{key} must be a non-empty string"
        raise ValueError(msg)
    return item


def require_int(value: Mapping[str, Any], key: str) -> int:
    item = value.get(key)
    if isinstance(item, bool) or not isinstance(item, int):
        msg = f"{key} must be an integer"
        raise ValueError(msg)
    return item


def write_json(path: str | Path, value: Mapping[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2), encoding="utf-8")
    return output
