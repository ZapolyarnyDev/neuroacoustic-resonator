from __future__ import annotations

from pathlib import Path

from neuroacoustic_resonator.analysis.distinguishability_classification import (
    classify_protocol_embeddings,
)
from neuroacoustic_resonator.analysis.protocol_embeddings import (
    FEATURE_COLUMNS,
    EmbeddingRow,
    write_embedding_rows,
)


def classification_rows() -> list[EmbeddingRow]:
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
    for seed_root, split in roots:
        for label, center in (
            ("silence", 0.0),
            ("tone", 3.0),
            ("chirp", 6.0),
            ("pulse", 9.0),
            ("noise", 12.0),
        ):
            for repeat in (1, 2):
                row: EmbeddingRow = {
                    "trial_id": f"{label}-{seed_root}-{repeat}",
                    "stimulus_label": label,
                    "source_type": label,
                    "seed_root": seed_root,
                    "field_seed": seed_root * 10 + repeat,
                    "repeat_index": repeat,
                    "split": split,
                    "protocol_version": "0.1",
                    "response_frames": 8,
                }
                row.update({name: 0.0 for name in FEATURE_COLUMNS})
                row[FEATURE_COLUMNS[0]] = (
                    center + (seed_root % 4) * 0.03 + repeat * 0.02
                )
                row[FEATURE_COLUMNS[1]] = 100.0 if split == "test" else 0.0
                rows.append(row)
    return rows


def test_classification_selects_on_validation_and_scores_held_out_test(
    tmp_path: Path,
) -> None:
    embeddings = tmp_path / "embeddings.csv"
    report_path = tmp_path / "classification.json"
    predictions_path = tmp_path / "predictions.csv"
    plot_path = tmp_path / "confusion.png"
    write_embedding_rows(embeddings, classification_rows())

    report = classify_protocol_embeddings(
        embeddings,
        report_path,
        predictions_path,
        plot_path,
        permutation_samples=500,
        permutation_seed=5,
        bootstrap_samples=200,
        bootstrap_seed=7,
    )

    assert report["selection"]["split"] == "validation"
    assert report["selection"]["selected_metric"] == "euclidean"
    assert report["test"]["balanced_accuracy"] == 1.0
    assert report["test"]["balanced_accuracy_ci"] == [1.0, 1.0]
    assert report["permutation_baseline"]["mean_balanced_accuracy"] < 0.6
    assert report["permutation_baseline"]["p_value"] < 0.05
    assert FEATURE_COLUMNS[1] in report["standardization"]["dropped_features"]
    assert report_path.exists()
    assert predictions_path.exists()
    assert plot_path.exists()
