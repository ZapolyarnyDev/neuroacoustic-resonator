from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import numpy as np

from neuroacoustic_resonator.analysis.distinguishability_classification import (
    DistanceMetric,
    balanced_accuracy,
    classify_protocol_embeddings,
    fit_nearest_centroid,
    labels_for_indices,
    validate_splits,
)
from neuroacoustic_resonator.analysis.distinguishability_diagnostics import (
    aggregate_variance,
    diagnose_protocol_embeddings,
    variance_decomposition,
)
from neuroacoustic_resonator.analysis.distinguishability_statistics import (
    fit_train_standardizer,
)
from neuroacoustic_resonator.analysis.protocol_embeddings import (
    ResponseRepresentation,
    extract_protocol_embeddings,
    read_embedding_rows,
)

REPRESENTATIONS: tuple[ResponseRepresentation, ...] = (
    "absolute",
    "pre_input_delta",
    "input_end_delta",
    "response_velocity",
)


def derive_seed_invariant_representations(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    permutation_samples: int = 2_000,
    permutation_seed: int = 20_260_814,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    candidates: list[dict[str, Any]] = []
    paths: dict[ResponseRepresentation, Path] = {}
    for representation in REPRESENTATIONS:
        embeddings = output / f"{representation}_embeddings.csv"
        schema = output / f"{representation}_schema.json"
        extract_protocol_embeddings(
            manifest_path,
            embeddings,
            schema,
            representation=representation,
        )
        paths[representation] = embeddings
        candidates.append(score_representation(embeddings, representation))
    selected = select_representation(candidates)
    selected_name = selected["representation"]
    if selected_name not in REPRESENTATIONS:
        msg = f"unsupported selected representation: {selected_name}"
        raise ValueError(msg)
    selected_representation = cast(ResponseRepresentation, selected_name)
    evaluation = evaluate_selected_representation(
        selected_embeddings=paths[selected_representation],
        output_dir=output,
        permutation_samples=permutation_samples,
        permutation_seed=permutation_seed,
    )
    report = {
        "source_manifest": str(manifest_path),
        "selection_policy": {
            "data": "train_and_validation_only",
            "primary_metric": "validation_balanced_accuracy",
            "secondary_metric": "stimulus_to_seed_variance_ratio",
            "test_feature_values_used_for_selection": False,
        },
        "candidates": candidates,
        "selected_representation": selected_representation,
        **evaluation,
    }
    write_json(output / "representation_report.json", report)
    return report


def evaluate_selected_representation(
    *,
    selected_embeddings: Path,
    output_dir: Path,
    permutation_samples: int,
    permutation_seed: int,
) -> dict[str, Any]:
    classification = classify_protocol_embeddings(
        selected_embeddings,
        output_dir / "selected_classification.json",
        output_dir / "selected_predictions.csv",
        output_dir / "selected_confusion.png",
        permutation_samples=permutation_samples,
        permutation_seed=permutation_seed,
        bootstrap_samples=5_000,
        bootstrap_seed=permutation_seed + 1,
    )
    diagnostics = diagnose_protocol_embeddings(
        selected_embeddings,
        output_dir / "selected_diagnostics.json",
        output_dir / "selected_diagnostic_features.csv",
        output_dir / "selected_diagnostic_distances.csv",
        output_dir / "selected_diagnostics.png",
        permutation_samples=permutation_samples,
        permutation_seed=permutation_seed + 2,
    )
    return {
        "selected_test": {
            "balanced_accuracy": classification["test"]["balanced_accuracy"],
            "balanced_accuracy_ci": classification["test"]["balanced_accuracy_ci"],
            "permutation_p_value": classification["permutation_baseline"]["p_value"],
            "stimulus_variance_fraction": diagnostics["variance"]["aggregate"][
                "stimulus_fraction"
            ],
            "field_seed_variance_fraction": diagnostics["variance"]["aggregate"][
                "field_seed_fraction"
            ],
            "stimulus_to_seed_variance_ratio": diagnostics["variance"]["aggregate"][
                "stimulus_to_seed_ratio"
            ],
        },
        "outputs": {
            "classification": str(output_dir / "selected_classification.json"),
            "diagnostics": str(output_dir / "selected_diagnostics.json"),
        },
    }


def score_representation(
    embeddings_csv: str | Path,
    representation: ResponseRepresentation,
) -> dict[str, Any]:
    rows = read_embedding_rows(embeddings_csv)
    validate_splits(rows)
    selection_rows = [row for row in rows if row["split"] != "test"]
    standardizer = fit_train_standardizer(selection_rows)
    matrix = standardizer.transform(selection_rows)
    train_indices = np.asarray(
        [index for index, row in enumerate(selection_rows) if row["split"] == "train"],
        dtype=np.int64,
    )
    validation_indices = np.asarray(
        [
            index
            for index, row in enumerate(selection_rows)
            if row["split"] == "validation"
        ],
        dtype=np.int64,
    )
    train_labels = labels_for_indices(selection_rows, train_indices)
    validation_labels = labels_for_indices(selection_rows, validation_indices)
    metrics: tuple[DistanceMetric, ...] = ("euclidean", "cosine")
    metric_scores: dict[str, float] = {}
    for metric in metrics:
        model = fit_nearest_centroid(matrix[train_indices], train_labels, metric)
        predicted, _ = model.predict(matrix[validation_indices])
        metric_scores[metric] = balanced_accuracy(
            validation_labels,
            predicted,
            model.labels,
        )
    selected_metric = max(
        metrics,
        key=lambda metric: (metric_scores[metric], metric == "euclidean"),
    )
    feature_rows = variance_decomposition(selection_rows, matrix, standardizer)
    variance = aggregate_variance(feature_rows)
    return {
        "representation": representation,
        "selection_rows": len(selection_rows),
        "test_rows_used": 0,
        "selected_metric": selected_metric,
        "validation_balanced_accuracy": metric_scores[selected_metric],
        "candidate_metric_scores": metric_scores,
        "stimulus_variance_fraction": variance["stimulus_fraction"],
        "field_seed_variance_fraction": variance["field_seed_fraction"],
        "stimulus_to_seed_variance_ratio": variance["stimulus_to_seed_ratio"],
        "active_feature_count": len(standardizer.feature_names),
        "dropped_feature_count": len(standardizer.dropped_features),
    }


def select_representation(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if {candidate["representation"] for candidate in candidates} != set(
        REPRESENTATIONS
    ):
        msg = "representation selection requires every predefined candidate"
        raise ValueError(msg)
    return max(
        candidates,
        key=lambda candidate: (
            float(candidate["validation_balanced_accuracy"]),
            float(candidate["stimulus_to_seed_variance_ratio"]),
            -REPRESENTATIONS.index(
                cast(ResponseRepresentation, candidate["representation"])
            ),
        ),
    )


def derive_stage_one_representations(
    stage_root: str | Path,
    output_report: str | Path,
    *,
    permutation_samples: int = 2_000,
    permutation_seed: int = 20_260_814,
) -> dict[str, Any]:
    root = Path(stage_root)
    scenario_dirs = sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and (path / "manifest.json").exists()
    )
    if not scenario_dirs:
        msg = f"no completed Stage 1 scenarios found in {root}"
        raise ValueError(msg)
    baseline_dir = next(
        (
            scenario_dir
            for scenario_dir in scenario_dirs
            if scenario_dir.name == "baseline"
        ),
        None,
    )
    if baseline_dir is None:
        msg = "Stage 1 representation selection requires the baseline scenario"
        raise ValueError(msg)
    baseline_report = derive_seed_invariant_representations(
        baseline_dir / "manifest.json",
        baseline_dir / "representations",
        permutation_samples=permutation_samples,
        permutation_seed=permutation_seed,
    )
    selected_name = baseline_report["selected_representation"]
    if selected_name not in REPRESENTATIONS:
        msg = f"unsupported baseline representation: {selected_name}"
        raise ValueError(msg)
    selected_representation = cast(ResponseRepresentation, selected_name)
    reports = []
    for index, scenario_dir in enumerate(scenario_dirs):
        if scenario_dir == baseline_dir:
            report = baseline_report
        else:
            representation_dir = scenario_dir / "representations"
            representation_dir.mkdir(parents=True, exist_ok=True)
            embeddings = (
                representation_dir / f"{selected_representation}_embeddings.csv"
            )
            extract_protocol_embeddings(
                scenario_dir / "manifest.json",
                embeddings,
                representation_dir / f"{selected_representation}_schema.json",
                representation=selected_representation,
            )
            evaluation = evaluate_selected_representation(
                selected_embeddings=embeddings,
                output_dir=representation_dir,
                permutation_samples=permutation_samples,
                permutation_seed=permutation_seed + index * 10,
            )
            report = {
                "source_manifest": str(scenario_dir / "manifest.json"),
                "selected_representation": selected_representation,
                "selection_source": "baseline_train_and_validation",
                **evaluation,
            }
            write_json(
                representation_dir / "representation_report.json",
                report,
            )
        reports.append(
            {
                "scenario": scenario_dir.name,
                "selected_representation": report["selected_representation"],
                "selected_test": report["selected_test"],
                "report": str(
                    scenario_dir / "representations" / "representation_report.json"
                ),
            }
        )
    summary = {
        "stage_root": str(root),
        "selection_scenario": "baseline",
        "selected_representation": selected_representation,
        "selection_policy": "baseline_train_and_validation_only",
        "scenario_count": len(reports),
        "scenarios": reports,
    }
    write_json(output_report, summary)
    return summary


def write_json(path: str | Path, value: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2), encoding="utf-8")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Derive seed-invariant representations from recorded protocols.",
    )
    parser.add_argument(
        "--stage-root",
        type=Path,
        default=Path("experiments/distinguishability_stage_one"),
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=Path(
            "experiments/logs/distinguishability_seed_invariant_representations.json"
        ),
    )
    parser.add_argument("--permutation-samples", type=int, default=2_000)
    parser.add_argument("--permutation-seed", type=int, default=20_260_814)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = derive_stage_one_representations(
        args.stage_root,
        args.output_report,
        permutation_samples=args.permutation_samples,
        permutation_seed=args.permutation_seed,
    )
    print(json.dumps(report, indent=2))
    return 0
