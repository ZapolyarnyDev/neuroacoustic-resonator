from __future__ import annotations

import json

import pytest

from neuroacoustic_resonator.analysis.controlled_equilibration import (
    ControlledEquilibrationCorpusConfig,
    leave_one_seed_root_out_classification,
    validate_pair_entry,
)
from neuroacoustic_resonator.analysis.protocol_embeddings import (
    FEATURE_COLUMNS,
    EmbeddingRow,
    write_embedding_rows,
)
from neuroacoustic_resonator.core.field import FieldConfig
from neuroacoustic_resonator.core.simulation import Simulation
from neuroacoustic_resonator.io.persistence import (
    checkpoint_fingerprint,
    save_simulation_checkpoint,
)


def test_small_controlled_corpus_fixes_30_branches() -> None:
    config = ControlledEquilibrationCorpusConfig.from_file(
        "configs/controlled_equilibration.yaml"
    )

    assert config.seed_roots == (101, 401, 601)
    assert config.repeats == 2
    assert len(config.stimuli) == 5
    assert len(config.seed_roots) * config.repeats * len(config.stimuli) == 30
    assert config.equilibration.neutral_steps == 512


def test_pair_validation_requires_identical_checkpoint_fingerprints(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.npz"
    save_simulation_checkpoint(checkpoint, Simulation(FieldConfig(size=5, seed=7)))
    fingerprint = checkpoint_fingerprint(checkpoint)
    common = {
        "seed_root": 101,
        "field_seed": 77,
        "repeat_index": 1,
        "split": "train",
        "checkpoint_fingerprint": fingerprint,
    }
    stimulus_metadata = tmp_path / "stimulus.json"
    control_metadata = tmp_path / "control.json"
    stimulus_metadata.write_text(
        json.dumps({**common, "source_type": "tone"}),
        encoding="utf-8",
    )
    control_metadata.write_text(
        json.dumps({**common, "source_type": "silence"}),
        encoding="utf-8",
    )
    pair = {
        **common,
        "checkpoint": str(checkpoint),
        "stimulus_metadata_json": str(stimulus_metadata),
        "control_metadata_json": str(control_metadata),
    }

    validate_pair_entry(pair)
    control_metadata.write_text(
        json.dumps(
            {
                **common,
                "source_type": "silence",
                "checkpoint_fingerprint": "wrong",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="branch fingerprint"):
        validate_pair_entry(pair)


def test_leave_one_seed_root_out_classification_uses_all_roots(tmp_path) -> None:
    rows: list[EmbeddingRow] = []
    for root_index, seed_root in enumerate((101, 401, 601)):
        for repeat_index in (1, 2):
            for label_index, label in enumerate(("tone", "chirp", "pulse", "noise")):
                row: EmbeddingRow = {
                    "trial_id": f"{label}-{seed_root}-{repeat_index}",
                    "stimulus_label": label,
                    "source_type": label,
                    "seed_root": seed_root,
                    "field_seed": seed_root * 10 + repeat_index,
                    "repeat_index": repeat_index,
                    "split": ("train", "validation", "test")[root_index],
                    "protocol_version": "0.1",
                    "response_frames": 8,
                }
                row.update({name: 0.0 for name in FEATURE_COLUMNS})
                row[FEATURE_COLUMNS[0]] = label_index * 5.0 + repeat_index * 0.01
                rows.append(row)
    embeddings = tmp_path / "embeddings.csv"
    write_embedding_rows(embeddings, rows)

    report = leave_one_seed_root_out_classification(
        embeddings,
        permutation_samples=20,
        permutation_seed=7,
    )

    assert report["seed_roots"] == [101, 401, 601]
    assert report["balanced_accuracy"] == 1.0
    assert report["chance_level"] == 0.25
