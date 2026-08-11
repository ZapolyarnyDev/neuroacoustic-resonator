from __future__ import annotations

from pathlib import Path

from neuroacoustic_resonator.analysis.distinguishability_statistics import (
    fit_train_standardizer,
    quantify_response_separation,
)
from neuroacoustic_resonator.analysis.protocol_embeddings import (
    FEATURE_COLUMNS,
    EmbeddingRow,
    write_embedding_rows,
)


def embedding_row(
    trial_id: str,
    label: str,
    seed_root: int,
    repeat: int,
    split: str,
    value: float,
    test_only_value: float = 0.0,
) -> EmbeddingRow:
    row: EmbeddingRow = {
        "trial_id": trial_id,
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
    row[FEATURE_COLUMNS[0]] = value
    row[FEATURE_COLUMNS[1]] = test_only_value
    return row


def synthetic_rows() -> list[EmbeddingRow]:
    rows: list[EmbeddingRow] = []
    for seed_root, split in ((11, "train"), (17, "train"), (29, "test")):
        seed_offset = (seed_root % 5) * 0.02
        for label, center in (("tone", 0.0), ("noise", 5.0)):
            for repeat in (1, 2):
                rows.append(
                    embedding_row(
                        f"{label}-{seed_root}-{repeat}",
                        label,
                        seed_root,
                        repeat,
                        split,
                        center + seed_offset + repeat * 0.05,
                        test_only_value=100.0 if split == "test" else 0.0,
                    )
                )
    return rows


def test_standardizer_uses_only_varying_train_features() -> None:
    rows = synthetic_rows()
    string_rows = [{key: str(value) for key, value in row.items()} for row in rows]

    standardizer = fit_train_standardizer(string_rows)

    assert standardizer.feature_names == (FEATURE_COLUMNS[0],)
    assert FEATURE_COLUMNS[1] in standardizer.dropped_features


def test_quantify_response_separation_reports_seed_bootstrap(tmp_path: Path) -> None:
    embeddings = tmp_path / "embeddings.csv"
    report_path = tmp_path / "separation.json"
    pairs_path = tmp_path / "pairs.csv"
    plot_path = tmp_path / "separation.png"
    write_embedding_rows(embeddings, synthetic_rows())

    report = quantify_response_separation(
        embeddings,
        report_path,
        pairs_path,
        plot_path,
        bootstrap_samples=200,
        bootstrap_seed=7,
    )

    assert report["bootstrap"]["unit"] == "seed_root"
    assert report["standardization"]["fit_split"] == "train"
    assert report["overall"]["separation_margin"] > 0.0
    assert report["overall"]["separation_margin_ci"][0] > 0.0
    assert report["overall"]["cliffs_delta"] == 1.0
    assert report["by_split"]["validation"] == {"available": False}
    assert len(report["by_seed"]) == 3
    assert report_path.exists()
    assert pairs_path.exists()
    assert plot_path.exists()
