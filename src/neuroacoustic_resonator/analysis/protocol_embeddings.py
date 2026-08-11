from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Callable, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from neuroacoustic_resonator.protocol import SoundProtocolFrame, read_protocol_jsonl

EmbeddingRow = dict[str, str | int | float]
SignalGetter = Callable[[SoundProtocolFrame], float]

IDENTITY_COLUMNS = (
    "trial_id",
    "stimulus_label",
    "source_type",
    "seed_root",
    "field_seed",
    "repeat_index",
    "split",
    "protocol_version",
    "response_frames",
)
TEMPORAL_STATISTICS = ("mean", "std", "max", "delta", "trend", "auc")
SIGNAL_GETTERS: dict[str, SignalGetter] = {
    "output_phase_coherence": lambda frame: frame.output_region.phase_coherence,
    "output_local_synchrony": lambda frame: frame.output_region.mean_local_synchrony,
    "output_metabolite": lambda frame: frame.output_region.mean_metabolite,
    "output_trace": lambda frame: frame.output_region.mean_trace,
    "output_frequency": lambda frame: frame.output_region.mean_frequency,
    "output_frequency_spread": lambda frame: frame.output_region.frequency_spread,
    "pattern_phase_order_1": lambda frame: frame.pattern.phase_order_1,
    "pattern_phase_order_2": lambda frame: frame.pattern.phase_order_2,
    "pattern_phase_order_3": lambda frame: frame.pattern.phase_order_3,
    "pattern_trace_contrast": lambda frame: frame.pattern.trace_contrast,
    "pattern_metabolite_stress": lambda frame: frame.pattern.metabolite_stress,
    "pattern_frequency_spread": lambda frame: frame.pattern.frequency_spread,
}
EVENT_FEATURES = (
    "active_fraction",
    "active_confidence_mean",
    "active_intensity_mean",
    "active_novelty_mean",
    "transition_started_rate",
    "transition_changed_rate",
    "transition_ended_rate",
)
FEATURE_COLUMNS = (
    tuple(
        f"{signal}_{statistic}"
        for signal in SIGNAL_GETTERS
        for statistic in TEMPORAL_STATISTICS
    )
    + EVENT_FEATURES
)


def extract_protocol_embeddings(
    manifest_path: str | Path,
    output_csv: str | Path,
    output_schema: str | Path,
) -> dict[str, Any]:
    manifest = read_manifest(manifest_path)
    rows = [embedding_from_trial(entry) for entry in manifest]
    write_embedding_rows(output_csv, rows)
    schema = {
        "source_manifest": str(manifest_path),
        "rows": len(rows),
        "identity_columns": list(IDENTITY_COLUMNS),
        "feature_columns": list(FEATURE_COLUMNS),
        "feature_count": len(FEATURE_COLUMNS),
        "splits": split_counts(rows),
    }
    write_json(output_schema, schema)
    return schema


def read_manifest(path: str | Path) -> list[dict[str, Any]]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, list) or not value:
        msg = "distinguishability manifest must be a non-empty list"
        raise ValueError(msg)
    entries: list[dict[str, Any]] = []
    for index, entry in enumerate(value):
        if not isinstance(entry, dict):
            msg = f"manifest entry {index} must be an object"
            raise ValueError(msg)
        entries.append(entry)
    return entries


def embedding_from_trial(entry: dict[str, Any]) -> EmbeddingRow:
    metadata = read_trial_metadata(entry)
    frames = read_protocol_jsonl(require_path(entry, "protocol_jsonl"))
    response_frames = select_response_frames(frames, metadata)
    version = require_string(metadata, "protocol_version")
    if any(frame.version != version for frame in response_frames):
        msg = f"trial {require_string(entry, 'trial_id')} mixes protocol versions"
        raise ValueError(msg)
    row: EmbeddingRow = {
        "trial_id": require_string(entry, "trial_id"),
        "stimulus_label": require_string(entry, "stimulus_label"),
        "source_type": require_string(entry, "source_type"),
        "seed_root": require_int(entry, "seed_root"),
        "field_seed": require_int(entry, "field_seed"),
        "repeat_index": require_int(entry, "repeat_index"),
        "split": require_split(entry),
        "protocol_version": version,
        "response_frames": len(response_frames),
    }
    row.update(response_embedding(response_frames))
    return row


def read_trial_metadata(entry: dict[str, Any]) -> dict[str, Any]:
    path = require_path(entry, "metadata_json")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        msg = f"trial metadata must be an object: {path}"
        raise ValueError(msg)
    for key in (
        "trial_id",
        "stimulus_label",
        "seed_root",
        "field_seed",
        "repeat_index",
        "split",
        "protocol_jsonl",
    ):
        if value.get(key) != entry.get(key):
            msg = f"trial metadata mismatch for {key}: {path}"
            raise ValueError(msg)
    return value


def select_response_frames(
    frames: Sequence[SoundProtocolFrame],
    metadata: dict[str, Any],
) -> list[SoundProtocolFrame]:
    if not frames:
        msg = "protocol stream must not be empty"
        raise ValueError(msg)
    sequences = [frame.sequence for frame in frames]
    if any(right <= left for left, right in pairwise(sequences)):
        msg = "protocol frame sequences must be strictly increasing"
        raise ValueError(msg)
    segments = require_object(metadata, "segments")
    response = require_object(segments, "response")
    start = require_int(response, "sequence_start")
    end = require_int(response, "sequence_end")
    expected_count = require_int(response, "frames")
    selected = [frame for frame in frames if start <= frame.sequence <= end]
    if len(selected) != expected_count or not selected:
        msg = "response protocol segment does not match recorded metadata"
        raise ValueError(msg)
    if selected[0].sequence != start or selected[-1].sequence != end:
        msg = "response protocol segment boundaries are missing"
        raise ValueError(msg)
    return selected


def response_embedding(frames: Sequence[SoundProtocolFrame]) -> dict[str, float]:
    if not frames:
        msg = "response frames must not be empty"
        raise ValueError(msg)
    features: dict[str, float] = {}
    for signal_name, getter in SIGNAL_GETTERS.items():
        values = np.asarray([getter(frame) for frame in frames], dtype=np.float64)
        for statistic, value in temporal_statistics(values).items():
            features[f"{signal_name}_{statistic}"] = value
    active = [frame.active_pattern for frame in frames]
    frame_count = len(frames)
    features.update(
        {
            "active_fraction": sum(item is not None for item in active) / frame_count,
            "active_confidence_mean": float(
                np.mean([0.0 if item is None else item.confidence for item in active])
            ),
            "active_intensity_mean": float(
                np.mean([0.0 if item is None else item.intensity for item in active])
            ),
            "active_novelty_mean": float(
                np.mean([0.0 if item is None else item.novelty for item in active])
            ),
        }
    )
    for kind in ("started", "changed", "ended"):
        features[f"transition_{kind}_rate"] = (
            sum(
                frame.transition is not None and frame.transition.kind == kind
                for frame in frames
            )
            / frame_count
        )
    return features


def temporal_statistics(values: np.ndarray) -> dict[str, float]:
    if values.ndim != 1 or values.size == 0:
        msg = "temporal feature values must be a non-empty vector"
        raise ValueError(msg)
    if values.size == 1:
        trend = 0.0
        auc = float(values[0])
    else:
        time = np.linspace(0.0, 1.0, values.size)
        trend = float(np.polyfit(time, values, deg=1)[0])
        auc = float(np.trapezoid(values, time))
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "max": float(np.max(values)),
        "delta": float(values[-1] - values[0]),
        "trend": trend,
        "auc": auc,
    }


def write_embedding_rows(path: str | Path, rows: Sequence[EmbeddingRow]) -> Path:
    if not rows:
        msg = "embedding rows must not be empty"
        raise ValueError(msg)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [*IDENTITY_COLUMNS, *FEATURE_COLUMNS]
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output


def read_embedding_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        msg = "embedding table must not be empty"
        raise ValueError(msg)
    expected = [*IDENTITY_COLUMNS, *FEATURE_COLUMNS]
    if list(rows[0]) != expected:
        msg = "embedding table schema does not match the protocol feature schema"
        raise ValueError(msg)
    return rows


def split_counts(rows: Sequence[EmbeddingRow]) -> dict[str, int]:
    return {
        split: sum(row["split"] == split for row in rows)
        for split in ("train", "validation", "test")
    }


def require_object(value: dict[str, Any], key: str) -> dict[str, Any]:
    item = value.get(key)
    if not isinstance(item, dict):
        msg = f"{key} must be an object"
        raise ValueError(msg)
    return item


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


def require_path(value: dict[str, Any], key: str) -> Path:
    return Path(require_string(value, key))


def require_split(value: dict[str, Any]) -> str:
    split = require_string(value, "split")
    if split not in {"train", "validation", "test"}:
        msg = f"unsupported trial split: {split}"
        raise ValueError(msg)
    return split


def write_json(path: str | Path, value: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2), encoding="utf-8")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract response embeddings from recorded Sound Protocol streams.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("experiments/logs/distinguishability_corpus_manifest.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("experiments/logs/distinguishability_embeddings.csv"),
    )
    parser.add_argument(
        "--output-schema",
        type=Path,
        default=Path("experiments/logs/distinguishability_embedding_schema.json"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    schema = extract_protocol_embeddings(
        args.manifest,
        args.output_csv,
        args.output_schema,
    )
    print(json.dumps(schema, indent=2))
    return 0
