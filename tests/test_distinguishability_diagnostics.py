from __future__ import annotations

from pathlib import Path

from neuroacoustic_resonator.analysis.distinguishability_diagnostics import (
    diagnose_protocol_embeddings,
)
from neuroacoustic_resonator.analysis.protocol_embeddings import (
    FEATURE_COLUMNS,
    EmbeddingRow,
    write_embedding_rows,
)


def crossed_rows(
    *,
    stimulus_scale: float,
    seed_scale: float,
) -> list[EmbeddingRow]:
    rows: list[EmbeddingRow] = []
    labels = ("silence", "tone", "noise")
    field_seeds = (1011, 1012, 2111, 2112)
    for seed_index, field_seed in enumerate(field_seeds):
        for label_index, label in enumerate(labels):
            row: EmbeddingRow = {
                "trial_id": f"{label}-{field_seed}",
                "stimulus_label": label,
                "source_type": label,
                "seed_root": field_seed // 10,
                "field_seed": field_seed,
                "repeat_index": field_seed % 10,
                "split": "train",
                "protocol_version": "0.1",
                "response_frames": 8,
            }
            row.update({name: 0.0 for name in FEATURE_COLUMNS})
            row[FEATURE_COLUMNS[0]] = (
                label_index * stimulus_scale + seed_index * seed_scale
            )
            row[FEATURE_COLUMNS[1]] = (
                label_index * stimulus_scale * 0.5 + seed_index * seed_scale * 1.5
            )
            rows.append(row)
    return rows


def run_diagnostic(
    tmp_path: Path,
    rows: list[EmbeddingRow],
    prefix: str,
) -> dict:
    embeddings = tmp_path / f"{prefix}-embeddings.csv"
    report_path = tmp_path / f"{prefix}-diagnostics.json"
    features_path = tmp_path / f"{prefix}-features.csv"
    distances_path = tmp_path / f"{prefix}-distances.csv"
    plot_path = tmp_path / f"{prefix}.png"
    write_embedding_rows(embeddings, rows)
    report = diagnose_protocol_embeddings(
        embeddings,
        report_path,
        features_path,
        distances_path,
        plot_path,
        permutation_samples=500,
        permutation_seed=7,
    )
    assert report_path.exists()
    assert features_path.exists()
    assert distances_path.exists()
    assert plot_path.exists()
    return report


def test_diagnostics_identify_stimulus_dominant_embeddings(tmp_path: Path) -> None:
    report = run_diagnostic(
        tmp_path,
        crossed_rows(stimulus_scale=5.0, seed_scale=0.1),
        "stimulus",
    )

    variance = report["variance"]["aggregate"]
    assert variance["stimulus_fraction"] > variance["field_seed_fraction"]
    assert variance["stimulus_to_seed_ratio"] > 1.0
    assert report["distances"]["matched_stimulus_to_seed_ratio"] > 1.0
    assert report["distances"]["cross_seed_cliffs_delta"] > 0.0
    assert report["paired_permutation"]["p_value"] < 0.05
    assert report["diagnosis"]["stimulus_dominant"]


def test_diagnostics_identify_seed_dominant_embeddings(tmp_path: Path) -> None:
    report = run_diagnostic(
        tmp_path,
        crossed_rows(stimulus_scale=0.1, seed_scale=5.0),
        "seed",
    )

    variance = report["variance"]["aggregate"]
    assert variance["field_seed_fraction"] > variance["stimulus_fraction"]
    assert variance["stimulus_to_seed_ratio"] < 1.0
    assert report["distances"]["matched_stimulus_to_seed_ratio"] < 1.0
    assert not report["diagnosis"]["stimulus_dominant"]
