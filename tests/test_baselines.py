from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
from scipy.io import wavfile  # type: ignore[import-untyped]

from neuroacoustic_resonator.analysis.baselines import (
    classification_summary,
    input_features,
    uncoupled_corpus_config,
)
from neuroacoustic_resonator.analysis.controlled_equilibration import (
    ControlledEquilibrationCorpusConfig,
)

CONFIG = Path("configs/controlled_equilibration.yaml")


def write_tone(path: Path, *, amplitude: float, frequency: float) -> Path:
    sample_rate = 8_000
    time = np.arange(sample_rate // 2) / sample_rate
    samples = amplitude * np.sin(2.0 * np.pi * frequency * time)
    wavfile.write(path, sample_rate, samples.astype(np.float32))
    return path


def test_drive_features_summarize_drive_series(tmp_path) -> None:
    config = ControlledEquilibrationCorpusConfig.from_file(CONFIG)
    tone = write_tone(tmp_path / "tone.wav", amplitude=0.65, frequency=220.0)

    features = input_features(tone, config, signals=("drive",), include_bands=False)

    assert set(features) == {
        "drive_mean",
        "drive_std",
        "drive_max",
        "drive_delta",
        "drive_trend",
        "drive_auc",
    }
    assert all(np.isfinite(value) for value in features.values())


def test_audio_features_separate_bands(tmp_path) -> None:
    config = ControlledEquilibrationCorpusConfig.from_file(CONFIG)
    low = write_tone(tmp_path / "low.wav", amplitude=0.65, frequency=220.0)
    high = write_tone(tmp_path / "high.wav", amplitude=0.65, frequency=3_000.0)

    low_features = input_features(low, config, signals=(), include_bands=True)
    high_features = input_features(high, config, signals=(), include_bands=True)

    assert len(low_features) == 8 * 6
    assert low_features["band_0_mean"] > high_features["band_0_mean"]
    assert high_features["band_7_mean"] > low_features["band_7_mean"]


def test_uncoupled_corpus_disables_coupling_and_isolates_outputs(tmp_path) -> None:
    config = ControlledEquilibrationCorpusConfig.from_file(CONFIG)

    corpus_path = uncoupled_corpus_config(config, tmp_path / "uncoupled")

    corpus = ControlledEquilibrationCorpusConfig.from_file(corpus_path)
    field = yaml.safe_load(corpus.simulation_config.read_text(encoding="utf-8"))
    assert field["field"]["coupling_strength"] == 0.0
    assert field["field"]["coupling_homeostasis_rate"] == 0.0
    assert corpus.output_dir == tmp_path / "uncoupled"
    assert corpus.output_summary == tmp_path / "uncoupled" / "summary.json"
    assert corpus.stimuli == config.stimuli
    assert corpus.seed_roots == config.seed_roots


def test_classification_summary_extracts_headline() -> None:
    classification = {
        "balanced_accuracy": 0.5,
        "chance_level": 0.25,
        "checkpoint_clustered_bootstrap": {"balanced_accuracy_ci": [0.3, 0.6]},
        "checkpoint_label_permutation": {"p_value": 0.01},
    }

    assert classification_summary(classification) == {
        "balanced_accuracy": 0.5,
        "chance_level": 0.25,
        "balanced_accuracy_ci": [0.3, 0.6],
        "p_value": 0.01,
        "no_signal": False,
    }


def test_uncoupled_control_without_signal_reports_chance(tmp_path) -> None:
    from neuroacoustic_resonator.analysis.baselines import evaluate_embeddings
    from neuroacoustic_resonator.analysis.protocol_embeddings import (
        FEATURE_COLUMNS,
        EmbeddingRow,
        write_embedding_rows,
    )

    rows: list[EmbeddingRow] = []
    for seed_root in (101, 401, 601):
        for repeat_index in (1, 2):
            for label in ("tone", "chirp", "pulse", "noise"):
                row: EmbeddingRow = {
                    "trial_id": f"{label}-{seed_root}-{repeat_index}",
                    "stimulus_label": label,
                    "source_type": label,
                    "seed_root": seed_root,
                    "field_seed": seed_root * 10 + repeat_index,
                    "repeat_index": repeat_index,
                    "split": "train",
                    "protocol_version": "0.1",
                    "response_frames": 8,
                }
                row.update({name: 0.0 for name in FEATURE_COLUMNS})
                rows.append(row)
    embeddings = tmp_path / "embeddings.csv"
    write_embedding_rows(embeddings, rows)

    result = evaluate_embeddings(
        embeddings,
        tmp_path / "evidence.json",
        labels=4,
        permutation_samples=5,
        bootstrap_samples=5,
    )

    assert result == {
        "balanced_accuracy": 0.25,
        "chance_level": 0.25,
        "balanced_accuracy_ci": [0.25, 0.25],
        "p_value": 1.0,
        "no_signal": True,
    }
