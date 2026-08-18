from __future__ import annotations

from pathlib import Path

import pytest

from neuroacoustic_resonator.analysis.protocol_embeddings import (
    FEATURE_COLUMNS,
    EmbeddingRow,
    ResponseRepresentation,
    write_embedding_rows,
)
from neuroacoustic_resonator.analysis.seed_invariant_representations import (
    REPRESENTATIONS,
    score_representation,
    select_representation,
)


def representation_rows(*, stimulus_dominant: bool) -> list[EmbeddingRow]:
    rows: list[EmbeddingRow] = []
    roots = (
        (11, "train"),
        (17, "train"),
        (23, "train"),
        (29, "validation"),
        (31, "validation"),
        (37, "test"),
        (41, "test"),
    )
    labels = ("silence", "tone", "noise")
    for root_index, (seed_root, split) in enumerate(roots):
        for repeat in (1, 2):
            field_seed = seed_root * 10 + repeat
            for label_index, label in enumerate(labels):
                row: EmbeddingRow = {
                    "trial_id": f"{label}-{seed_root}-{repeat}",
                    "stimulus_label": label,
                    "source_type": label,
                    "seed_root": seed_root,
                    "field_seed": field_seed,
                    "repeat_index": repeat,
                    "split": split,
                    "protocol_version": "0.1",
                    "response_frames": 8,
                }
                row.update({name: 0.0 for name in FEATURE_COLUMNS})
                if stimulus_dominant:
                    value = label_index * 5.0 + root_index * 0.01 + repeat * 0.001
                else:
                    value = root_index * 5.0 + repeat + label_index * 0.001
                row[FEATURE_COLUMNS[0]] = value
                row[FEATURE_COLUMNS[1]] = value * 0.5
                rows.append(row)
    return rows


def score(
    tmp_path: Path,
    representation: ResponseRepresentation,
    *,
    stimulus_dominant: bool,
) -> dict[str, object]:
    path = tmp_path / f"{representation}.csv"
    write_embedding_rows(path, representation_rows(stimulus_dominant=stimulus_dominant))
    return score_representation(path, representation)


def test_representation_selection_uses_validation_and_seed_variance(tmp_path) -> None:
    candidates = [
        score(
            tmp_path,
            representation,
            stimulus_dominant=representation == "pre_input_delta",
        )
        for representation in REPRESENTATIONS
    ]

    selected = select_representation(candidates)

    assert selected["representation"] == "pre_input_delta"
    assert selected["validation_balanced_accuracy"] == 1.0
    assert selected["stimulus_to_seed_variance_ratio"] > 1.0
    assert all(candidate["test_rows_used"] == 0 for candidate in candidates)


def test_representation_selection_requires_predeclared_candidates() -> None:
    candidates = [
        {
            "representation": "absolute",
            "validation_balanced_accuracy": 1.0,
            "stimulus_to_seed_variance_ratio": 1.0,
        }
    ]

    with pytest.raises(ValueError, match="every predefined"):
        select_representation(candidates)
