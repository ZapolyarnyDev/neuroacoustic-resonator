from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from neuroacoustic_resonator.analysis.protocol_embeddings import (
    FEATURE_COLUMNS,
    extract_protocol_embeddings,
    read_embedding_rows,
)
from neuroacoustic_resonator.protocol import (
    FieldSnapshot,
    PatternSnapshot,
    RegionName,
    RegionSnapshot,
    SoundProtocolFrame,
    write_protocol_jsonl,
)


def region(name: str, value: float) -> RegionSnapshot:
    return RegionSnapshot(
        name=cast(RegionName, name),
        phase_coherence=value,
        mean_local_synchrony=value,
        mean_metabolite=0.8,
        min_metabolite=0.7,
        mean_trace=value,
        max_trace=value + 0.1,
        mean_frequency=1.0 + value,
        frequency_spread=value / 2.0,
        mean_coupling=0.2,
        coupling_spread=0.01,
    )


def frame(sequence: int, value: float) -> SoundProtocolFrame:
    return SoundProtocolFrame(
        sequence=sequence,
        step=sequence,
        time_seconds=sequence * 0.02,
        field=FieldSnapshot(
            global_synchrony=value,
            mean_local_synchrony=value,
            mean_metabolite=0.8,
            min_metabolite=0.7,
            mean_trace=value,
            max_trace=value + 0.1,
        ),
        input_region=region("input", value),
        assoc_region=region("assoc", value),
        output_region=region("output", value),
        pattern=PatternSnapshot(
            phase_order_1=value,
            phase_order_2=value / 2.0,
            phase_order_3=value / 3.0,
            trace_mean=value,
            trace_contrast=value / 4.0,
            metabolite_stress=value,
            metabolite_contrast=value / 5.0,
            trace_phase_lock=value,
            metabolite_phase_lock=value,
            frequency_mean=1.0 + value,
            frequency_spread=value / 2.0,
        ),
        active_pattern=None,
        transition=None,
    )


def write_trial(
    root: Path,
    trial_id: str,
    stimulus_label: str,
    seed_root: int,
    split: str,
    offset: float,
) -> dict[str, object]:
    protocol_path = root / f"{trial_id}.jsonl"
    metadata_path = root / f"{trial_id}.json"
    frames = [frame(index, offset + index * 0.1) for index in range(4)]
    write_protocol_jsonl(protocol_path, frames)
    entry: dict[str, object] = {
        "trial_id": trial_id,
        "stimulus_label": stimulus_label,
        "source_type": "tone",
        "seed_root": seed_root,
        "field_seed": seed_root * 10,
        "repeat_index": 1,
        "split": split,
        "protocol_jsonl": str(protocol_path),
        "metadata_json": str(metadata_path),
    }
    metadata = {
        **entry,
        "protocol_version": "0.1",
        "protocol_frames": 4,
        "segments": {
            "input": {"sequence_start": 0, "sequence_end": 0, "frames": 1},
            "response": {"sequence_start": 1, "sequence_end": 3, "frames": 3},
        },
    }
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    return entry


def test_extract_protocol_embeddings_uses_only_response_segment(tmp_path) -> None:
    manifest_path = tmp_path / "manifest.json"
    output_csv = tmp_path / "embeddings.csv"
    output_schema = tmp_path / "schema.json"
    manifest = [
        write_trial(tmp_path, "tone-train", "tone", 11, "train", 0.1),
        write_trial(tmp_path, "noise-test", "noise", 29, "test", 0.2),
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    schema = extract_protocol_embeddings(
        manifest_path,
        output_csv,
        output_schema,
    )
    rows = read_embedding_rows(output_csv)

    assert schema["rows"] == 2
    assert schema["feature_count"] == len(FEATURE_COLUMNS)
    assert schema["splits"] == {"train": 1, "validation": 0, "test": 1}
    assert rows[0]["response_frames"] == "3"
    assert float(rows[0]["output_phase_coherence_mean"]) == pytest.approx(0.3)
    assert float(rows[0]["output_phase_coherence_delta"]) == pytest.approx(0.2)
    assert float(rows[0]["output_phase_coherence_max"]) == pytest.approx(0.4)


def test_extract_protocol_embeddings_rejects_incomplete_response(tmp_path) -> None:
    entry = write_trial(tmp_path, "tone", "tone", 11, "train", 0.1)
    metadata_path = Path(str(entry["metadata_json"]))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["segments"]["response"]["frames"] = 4
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps([entry]), encoding="utf-8")

    with pytest.raises(ValueError, match="segment"):
        extract_protocol_embeddings(
            manifest_path,
            tmp_path / "embeddings.csv",
            tmp_path / "schema.json",
        )
