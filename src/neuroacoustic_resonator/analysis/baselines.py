from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from neuroacoustic_resonator.analysis.controlled_equilibration import (
    ControlledEquilibrationCorpusConfig,
    read_json_object,
    simulate_paired_corpus,
    write_json,
)
from neuroacoustic_resonator.analysis.paired_causal_evidence import (
    evaluate_paired_causal_evidence,
)
from neuroacoustic_resonator.analysis.protocol_embeddings import (
    IDENTITY_COLUMNS,
    EmbeddingRow,
    temporal_statistics,
    write_embedding_rows,
)
from neuroacoustic_resonator.audio.input import (
    extract_audio_input_features,
    extract_channel_audio_features,
)

DRIVE_SIGNALS = ("drive",)
AUDIO_SIGNALS = ("rms", "onset", "spectral_centroid")
AUDIO_BANDS = 8


def input_feature_rows(
    pairs: Sequence[Mapping[str, Any]],
    config: ControlledEquilibrationCorpusConfig,
    *,
    signals: Sequence[str],
    include_bands: bool,
) -> list[EmbeddingRow]:
    rows: list[EmbeddingRow] = []
    for pair in pairs:
        metadata = read_json_object(pair["stimulus_metadata_json"])
        row: EmbeddingRow = {
            "trial_id": pair["pair_id"],
            "stimulus_label": pair["stimulus_label"],
            "source_type": pair["source_type"],
            "seed_root": pair["seed_root"],
            "field_seed": pair["field_seed"],
            "repeat_index": pair["repeat_index"],
            "split": pair["split"],
            "protocol_version": metadata["protocol_version"],
            "response_frames": 0,
        }
        row.update(
            input_features(
                Path(metadata["input_wav"]),
                config,
                signals=signals,
                include_bands=include_bands,
            )
        )
        rows.append(row)
    return rows


def input_features(
    wav_path: Path,
    config: ControlledEquilibrationCorpusConfig,
    *,
    signals: Sequence[str],
    include_bands: bool,
) -> dict[str, float]:
    features = extract_audio_input_features(
        wav_path,
        frame_size=config.input_frame_size,
        hop_size=config.input_hop_size,
        drive_strength=config.drive_strength,
    )
    values: dict[str, float] = {}
    for signal in signals:
        series = np.asarray(getattr(features, signal), dtype=np.float64)
        for statistic, value in temporal_statistics(series).items():
            values[f"{signal}_{statistic}"] = value
    if include_bands:
        bands = extract_channel_audio_features(
            wav_path,
            frame_size=config.input_frame_size,
            hop_size=config.input_hop_size,
            bands=AUDIO_BANDS,
        ).band_energy.mean(axis=1)
        for band in range(AUDIO_BANDS):
            for statistic, value in temporal_statistics(bands[:, band]).items():
                values[f"band_{band}_{statistic}"] = value
    return values


def evaluate_input_baseline(
    name: str,
    pairs: Sequence[Mapping[str, Any]],
    config: ControlledEquilibrationCorpusConfig,
    output_dir: Path,
    *,
    signals: Sequence[str],
    include_bands: bool,
    permutation_samples: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    rows = input_feature_rows(
        pairs,
        config,
        signals=signals,
        include_bands=include_bands,
    )
    embeddings = output_dir / f"{name}_embeddings.csv"
    write_embedding_rows(
        embeddings,
        rows,
        feature_columns=tuple(key for key in rows[0] if key not in IDENTITY_COLUMNS),
    )
    return evaluate_embeddings(
        embeddings,
        output_dir / f"{name}_evidence.json",
        labels=len({row["stimulus_label"] for row in rows}),
        permutation_samples=permutation_samples,
        bootstrap_samples=bootstrap_samples,
    )


def evaluate_embeddings(
    embeddings: Path,
    output_report: Path,
    *,
    labels: int,
    permutation_samples: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    try:
        evidence = evaluate_paired_causal_evidence(
            embeddings,
            output_report,
            permutation_samples=permutation_samples,
            bootstrap_samples=bootstrap_samples,
        )
    except ValueError as error:
        if "no varying" not in str(error):
            raise
        return no_signal_result(1.0 / labels)
    return classification_summary(evidence["classification"])


def field_result(summary: Mapping[str, Any]) -> dict[str, Any]:
    if "no_signal" in summary:
        return no_signal_result(summary["no_signal"]["chance_level"])
    return classification_summary(summary["cross_seed_classification"])


def no_signal_result(chance: float) -> dict[str, Any]:
    return {
        "balanced_accuracy": chance,
        "chance_level": chance,
        "balanced_accuracy_ci": [chance, chance],
        "p_value": 1.0,
        "no_signal": True,
    }


UNCOUPLED_FIELD = {"coupling_strength": 0.0, "coupling_homeostasis_rate": 0.0}


def uncoupled_corpus_config(
    config: ControlledEquilibrationCorpusConfig,
    output_dir: Path,
) -> Path:
    return variant_corpus_config(config, output_dir, field_updates=UNCOUPLED_FIELD)


def variant_corpus_config(
    config: ControlledEquilibrationCorpusConfig,
    output_dir: Path,
    *,
    field_updates: Mapping[str, Any] | None = None,
    equilibration: Mapping[str, Any] | None = None,
) -> Path:
    field_config = yaml.safe_load(config.simulation_config.read_text(encoding="utf-8"))
    field_config["field"].update(field_updates or {})
    output_dir.mkdir(parents=True, exist_ok=True)
    field_path = output_dir / "field.yaml"
    field_path.write_text(yaml.safe_dump(field_config, sort_keys=False), "utf-8")
    values = config.model_dump(mode="json")
    values.update(
        simulation_config=field_path.as_posix(),
        output_dir=output_dir.as_posix(),
        output_manifest=(output_dir / "manifest.json").as_posix(),
        output_pairs=(output_dir / "pairs.json").as_posix(),
        output_embeddings=(output_dir / "causal_embeddings.csv").as_posix(),
        output_summary=(output_dir / "summary.json").as_posix(),
    )
    if equilibration is not None:
        values["equilibration"] = dict(equilibration)
    corpus_path = output_dir / "corpus.yaml"
    corpus_path.write_text(yaml.safe_dump(values, sort_keys=False), "utf-8")
    return corpus_path


def run_baselines(
    config_path: str | Path,
    *,
    permutation_samples: int = 10_000,
    bootstrap_samples: int = 5_000,
) -> dict[str, Any]:
    config = ControlledEquilibrationCorpusConfig.from_file(config_path)
    pairs = json.loads(config.output_pairs.read_text(encoding="utf-8"))
    output_dir = config.output_dir / "baselines"
    output_dir.mkdir(parents=True, exist_ok=True)
    field_summary = read_json_object(config.output_summary)
    uncoupled = ControlledEquilibrationCorpusConfig.from_file(
        uncoupled_corpus_config(config, output_dir / "uncoupled")
    )
    simulate_paired_corpus(uncoupled)
    report = {
        "field": field_result(field_summary),
        "drive": evaluate_input_baseline(
            "drive",
            pairs,
            config,
            output_dir,
            signals=DRIVE_SIGNALS,
            include_bands=False,
            permutation_samples=permutation_samples,
            bootstrap_samples=bootstrap_samples,
        ),
        "audio": evaluate_input_baseline(
            "audio",
            pairs,
            config,
            output_dir,
            signals=(*DRIVE_SIGNALS, *AUDIO_SIGNALS),
            include_bands=True,
            permutation_samples=permutation_samples,
            bootstrap_samples=bootstrap_samples,
        ),
        "uncoupled_field": evaluate_embeddings(
            uncoupled.output_embeddings,
            uncoupled.output_dir / "causal_evidence.json",
            labels=len({pair["stimulus_label"] for pair in pairs}),
            permutation_samples=permutation_samples,
            bootstrap_samples=bootstrap_samples,
        ),
    }
    write_json(output_dir / "baselines.json", report)
    field_summary["baselines"] = report
    write_json(config.output_summary, field_summary)
    return report


def classification_summary(classification: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "balanced_accuracy": classification["balanced_accuracy"],
        "chance_level": classification["chance_level"],
        "balanced_accuracy_ci": classification["checkpoint_clustered_bootstrap"][
            "balanced_accuracy_ci"
        ],
        "p_value": classification["checkpoint_label_permutation"]["p_value"],
        "no_signal": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare the field response with input and uncoupled baselines."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/controlled_equilibration.yaml"),
    )
    parser.add_argument("--permutation-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-samples", type=int, default=5_000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_baselines(
        args.config,
        permutation_samples=args.permutation_samples,
        bootstrap_samples=args.bootstrap_samples,
    )
    print(json.dumps(report, indent=2))
    return 0
