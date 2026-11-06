from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from neuroacoustic_resonator.analysis.baselines import (
    UNCOUPLED_FIELD,
    evaluate_embeddings,
    variant_corpus_config,
)
from neuroacoustic_resonator.analysis.controlled_equilibration import (
    ControlledEquilibrationCorpusConfig,
    simulate_paired_corpus,
    write_json,
)
from neuroacoustic_resonator.analysis.distinguishability_statistics import (
    NOISE_FLOOR,
)
from neuroacoustic_resonator.analysis.protocol_embeddings import (
    embedding_feature_columns,
    read_embedding_rows,
)
from neuroacoustic_resonator.core.regions import RegionMasks
from neuroacoustic_resonator.core.simulation import Simulation

NO_EQUILIBRATION = {
    "neutral_steps": 0,
    "phase_damping": 0.0,
    "metabolite_baseline": None,
}
PURE_KURAMOTO = {
    "metabolite_cost": 0.0,
    "metabolite_diffusion": 0.0,
    "trace_rate": 0.0,
    "frequency_plasticity_rate": 0.0,
    "frequency_homeostasis_rate": 0.0,
    "coupling_homeostasis_rate": 0.0,
}


@dataclass(frozen=True)
class FieldPart:
    slug: str
    title: str
    field_updates: Mapping[str, Any] = field(default_factory=dict)
    equilibration: Mapping[str, Any] | None = None
    cut_output: bool = False


FIELD_PARTS = (
    FieldPart("full", "full field"),
    FieldPart("no-coupling", "no coupling", UNCOUPLED_FIELD),
    FieldPart(
        "no-coupling-homeostasis",
        "no coupling homeostasis",
        {"coupling_homeostasis_rate": 0.0},
    ),
    FieldPart("no-metabolite-cost", "no metabolite cost", {"metabolite_cost": 0.0}),
    FieldPart(
        "no-metabolite-diffusion",
        "no metabolite diffusion",
        {"metabolite_diffusion": 0.0},
    ),
    FieldPart("no-trace", "no trace", {"trace_rate": 0.0}),
    FieldPart(
        "no-frequency-plasticity",
        "no frequency plasticity",
        {"frequency_plasticity_rate": 0.0},
    ),
    FieldPart(
        "no-frequency-homeostasis",
        "no frequency homeostasis",
        {"frequency_homeostasis_rate": 0.0},
    ),
    FieldPart("pure-kuramoto", "pure kuramoto", PURE_KURAMOTO),
    FieldPart("no-equilibration", "no equilibration", equilibration=NO_EQUILIBRATION),
    FieldPart("output-cut", "output cut", cut_output=True),
    FieldPart(
        "output-cut-no-diffusion",
        "output cut, no metabolite diffusion",
        {"metabolite_diffusion": 0.0},
        cut_output=True,
    ),
    FieldPart(
        "output-cut-no-diffusion-no-homeostasis",
        "output cut, no diffusion, no coupling homeostasis",
        {"metabolite_diffusion": 0.0, "coupling_homeostasis_rate": 0.0},
        cut_output=True,
    ),
)


def cut_output_region(simulation: Simulation) -> None:
    field_config = simulation.field.config
    simulation.field.set_region_coupling(
        RegionMasks.from_size(field_config.size).output,
        field_config.min_coupling,
    )


def run_field_parts(
    config_path: str | Path,
    *,
    parts: tuple[FieldPart, ...] = FIELD_PARTS,
    permutation_samples: int = 2_000,
    bootstrap_samples: int = 1_000,
) -> dict[str, Any]:
    config = ControlledEquilibrationCorpusConfig.from_file(config_path)
    output_dir = config.output_dir.parent / "parts"
    results: dict[str, Any] = {}
    for part in parts:
        variant = ControlledEquilibrationCorpusConfig.from_file(
            variant_corpus_config(
                config,
                output_dir / part.slug,
                field_updates=part.field_updates,
                equilibration=part.equilibration,
            )
        )
        _, _, pairs = simulate_paired_corpus(
            variant,
            prepare_branch_point=cut_output_region if part.cut_output else None,
        )
        result = evaluate_embeddings(
            variant.output_embeddings,
            variant.output_dir / "causal_evidence.json",
            labels=len({pair["stimulus_label"] for pair in pairs}),
            permutation_samples=permutation_samples,
            bootstrap_samples=bootstrap_samples,
        )
        result.update(signal_strength(variant.output_embeddings))
        results[part.slug] = {"title": part.title, **result}
        print(format_row(part.title, results[part.slug]), flush=True)
    output = output_dir / "parts.json"
    report = {"config": str(config_path), "output": str(output), "parts": results}
    write_json(output, report)
    return report


def signal_strength(embeddings: Path) -> dict[str, Any]:
    rows = read_embedding_rows(embeddings)
    columns = embedding_feature_columns(rows)
    matrix = np.asarray(
        [[float(row[column]) for column in columns] for row in rows],
        dtype=np.float64,
    )
    labels = np.asarray([row["stimulus_label"] for row in rows])
    scale = np.std(matrix, axis=0)
    active = scale > NOISE_FLOOR * np.maximum(1.0, np.abs(np.mean(matrix, axis=0)))
    changing = int(np.count_nonzero(np.max(np.abs(matrix), axis=0) > 0.0))
    if not np.any(active):
        return {"stimulus_eta": 0.0, "changing_features": changing}
    standardized = (matrix[:, active] - np.mean(matrix[:, active], axis=0)) / scale[
        active
    ]
    between = sum(
        np.count_nonzero(labels == label)
        * np.square(np.mean(standardized[labels == label], axis=0))
        for label in np.unique(labels)
    )
    return {
        "stimulus_eta": float(np.mean(between / len(labels))),
        "changing_features": changing,
    }


def format_row(title: str, result: Mapping[str, Any]) -> str:
    if result.get("no_signal"):
        return (
            f"| {title} | no signal | — | — | 0.000 | {result['changing_features']} |"
        )
    low, high = result["balanced_accuracy_ci"]
    return (
        f"| {title} | {result['balanced_accuracy']:.3f} | [{low:.3f}, {high:.3f}] | "
        f"{result['p_value']:.4f} | {result['stimulus_eta']:.3f} | "
        f"{result['changing_features']} |"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure the response with each field component disabled."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/controlled_equilibration.yaml"),
    )
    parser.add_argument("--permutation-samples", type=int, default=2_000)
    parser.add_argument("--bootstrap-samples", type=int, default=1_000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print("| Variant | Accuracy | 95% CI | p | Stimulus eta | Changing features |")
    print("|---|---|---|---|---|---|")
    report = run_field_parts(
        args.config,
        permutation_samples=args.permutation_samples,
        bootstrap_samples=args.bootstrap_samples,
    )
    print(report["output"])
    return 0
