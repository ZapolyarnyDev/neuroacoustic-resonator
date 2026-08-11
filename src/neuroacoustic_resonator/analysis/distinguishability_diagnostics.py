from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from neuroacoustic_resonator.analysis.distinguishability_statistics import (
    TrainStandardizer,
    cliffs_delta,
    distribution_summary,
    fit_train_standardizer,
)
from neuroacoustic_resonator.analysis.protocol_embeddings import read_embedding_rows

StringRow = dict[str, str]
DiagnosticDistanceRow = dict[str, str | int | float]


def diagnose_protocol_embeddings(
    embeddings_csv: str | Path,
    output_report: str | Path,
    output_features: str | Path,
    output_distances: str | Path,
    output_plot: str | Path,
    *,
    permutation_samples: int = 2_000,
    permutation_seed: int = 20_260_813,
) -> dict[str, Any]:
    if permutation_samples < 1:
        msg = "permutation_samples must be positive"
        raise ValueError(msg)
    rows = read_embedding_rows(embeddings_csv)
    validate_crossed_design(rows)
    standardizer = fit_train_standardizer(rows)
    normalized = standardizer.transform(rows)
    feature_rows = variance_decomposition(rows, normalized, standardizer)
    aggregate = aggregate_variance(feature_rows)
    distance_rows = diagnostic_distances(rows, normalized)
    distances = summarize_diagnostic_distances(distance_rows)
    permutation = paired_stimulus_permutation(
        rows,
        normalized,
        samples=permutation_samples,
        seed=permutation_seed,
    )
    report = {
        "source_embeddings": str(embeddings_csv),
        "rows": len(rows),
        "stimuli": sorted({row["stimulus_label"] for row in rows}),
        "field_seeds": sorted({int(row["field_seed"]) for row in rows}),
        "standardization": standardizer.to_dict(),
        "variance": {
            "aggregate": aggregate,
            "top_stimulus_features": top_features(
                feature_rows,
                key="stimulus_fraction",
            ),
            "top_seed_features": top_features(
                feature_rows,
                key="field_seed_fraction",
            ),
        },
        "distances": distances,
        "paired_permutation": permutation,
        "diagnosis": diagnostic_verdict(aggregate, distances, permutation),
        "outputs": {
            "feature_metrics_csv": str(output_features),
            "distance_metrics_csv": str(output_distances),
            "plot": str(output_plot),
        },
    }
    write_rows(output_features, feature_rows)
    write_rows(output_distances, distance_rows)
    write_diagnostic_plot(output_plot, aggregate, distance_rows)
    write_json(output_report, report)
    return report


def diagnose_stage_one(
    stage_root: str | Path,
    output_report: str | Path,
    *,
    permutation_samples: int = 2_000,
    permutation_seed: int = 20_260_813,
) -> dict[str, Any]:
    root = Path(stage_root)
    scenario_dirs = sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and (path / "embeddings.csv").exists()
    )
    if not scenario_dirs:
        msg = f"no completed Stage 1 scenarios found in {root}"
        raise ValueError(msg)
    scenarios: list[dict[str, Any]] = []
    for index, scenario_dir in enumerate(scenario_dirs):
        report = diagnose_protocol_embeddings(
            scenario_dir / "embeddings.csv",
            scenario_dir / "diagnostics.json",
            scenario_dir / "diagnostic_features.csv",
            scenario_dir / "diagnostic_distances.csv",
            scenario_dir / "diagnostics.png",
            permutation_samples=permutation_samples,
            permutation_seed=permutation_seed + index,
        )
        scenarios.append(
            {
                "scenario": scenario_dir.name,
                "stimulus_variance_fraction": report["variance"]["aggregate"][
                    "stimulus_fraction"
                ],
                "field_seed_variance_fraction": report["variance"]["aggregate"][
                    "field_seed_fraction"
                ],
                "stimulus_to_seed_variance_ratio": report["variance"]["aggregate"][
                    "stimulus_to_seed_ratio"
                ],
                "matched_stimulus_to_seed_distance_ratio": report["distances"][
                    "matched_stimulus_to_seed_ratio"
                ],
                "cross_seed_cliffs_delta": report["distances"][
                    "cross_seed_cliffs_delta"
                ],
                "stimulus_permutation_p_value": report["paired_permutation"]["p_value"],
                "diagnosis": report["diagnosis"],
                "report": str(scenario_dir / "diagnostics.json"),
            }
        )
    summary = {
        "stage_root": str(root),
        "scenario_count": len(scenarios),
        "scenarios": scenarios,
        "consensus": diagnostic_consensus(scenarios),
    }
    write_json(output_report, summary)
    return summary


def validate_crossed_design(rows: Sequence[StringRow]) -> None:
    labels = {row["stimulus_label"] for row in rows}
    if len(labels) < 2:
        msg = "diagnostics require at least two stimulus labels"
        raise ValueError(msg)
    labels_by_seed: dict[int, set[str]] = defaultdict(set)
    counts: dict[tuple[int, str], int] = defaultdict(int)
    for row in rows:
        field_seed = int(row["field_seed"])
        label = row["stimulus_label"]
        labels_by_seed[field_seed].add(label)
        counts[(field_seed, label)] += 1
    if any(seed_labels != labels for seed_labels in labels_by_seed.values()):
        msg = "every field seed must contain every stimulus label"
        raise ValueError(msg)
    if any(count != 1 for count in counts.values()):
        msg = "diagnostics require one trial per stimulus and field seed"
        raise ValueError(msg)


def variance_decomposition(
    rows: Sequence[StringRow],
    matrix: np.ndarray,
    standardizer: TrainStandardizer,
) -> list[dict[str, str | float]]:
    labels = np.asarray([row["stimulus_label"] for row in rows])
    field_seeds = np.asarray([int(row["field_seed"]) for row in rows])
    unique_labels = np.unique(labels)
    unique_seeds = np.unique(field_seeds)
    feature_rows: list[dict[str, str | float]] = []
    for feature_index, feature_name in enumerate(standardizer.feature_names):
        values = matrix[:, feature_index]
        grand_mean = float(np.mean(values))
        label_means = {
            str(label): float(np.mean(values[labels == label]))
            for label in unique_labels
        }
        seed_means = {
            int(seed): float(np.mean(values[field_seeds == seed]))
            for seed in unique_seeds
        }
        stimulus_ss = float(
            unique_seeds.size
            * sum(
                (label_means[str(label)] - grand_mean) ** 2 for label in unique_labels
            )
        )
        seed_ss = float(
            unique_labels.size
            * sum((seed_means[int(seed)] - grand_mean) ** 2 for seed in unique_seeds)
        )
        residuals = np.asarray(
            [
                value
                - label_means[str(label)]
                - seed_means[int(field_seed)]
                + grand_mean
                for value, label, field_seed in zip(
                    values,
                    labels,
                    field_seeds,
                    strict=True,
                )
            ],
            dtype=np.float64,
        )
        residual_ss = float(np.sum(residuals**2))
        total_ss = stimulus_ss + seed_ss + residual_ss
        if total_ss <= 1e-12:
            continue
        feature_rows.append(
            {
                "feature": feature_name,
                "stimulus_ss": stimulus_ss,
                "field_seed_ss": seed_ss,
                "residual_ss": residual_ss,
                "total_ss": total_ss,
                "stimulus_fraction": stimulus_ss / total_ss,
                "field_seed_fraction": seed_ss / total_ss,
                "residual_fraction": residual_ss / total_ss,
                "stimulus_to_seed_ratio": safe_ratio(stimulus_ss, seed_ss),
            }
        )
    if not feature_rows:
        msg = "diagnostics found no variable embedding features"
        raise ValueError(msg)
    return feature_rows


def aggregate_variance(
    feature_rows: Sequence[dict[str, str | float]],
) -> dict[str, float]:
    stimulus_ss = sum(float(row["stimulus_ss"]) for row in feature_rows)
    seed_ss = sum(float(row["field_seed_ss"]) for row in feature_rows)
    residual_ss = sum(float(row["residual_ss"]) for row in feature_rows)
    total_ss = stimulus_ss + seed_ss + residual_ss
    return {
        "stimulus_fraction": stimulus_ss / total_ss,
        "field_seed_fraction": seed_ss / total_ss,
        "residual_fraction": residual_ss / total_ss,
        "stimulus_to_seed_ratio": safe_ratio(stimulus_ss, seed_ss),
    }


def diagnostic_distances(
    rows: Sequence[StringRow],
    matrix: np.ndarray,
) -> list[DiagnosticDistanceRow]:
    output: list[DiagnosticDistanceRow] = []
    for left_index in range(len(rows)):
        for right_index in range(left_index + 1, len(rows)):
            left = rows[left_index]
            right = rows[right_index]
            same_label = left["stimulus_label"] == right["stimulus_label"]
            same_field_seed = left["field_seed"] == right["field_seed"]
            if same_label and same_field_seed:
                continue
            if same_field_seed:
                group = "matched_between_stimulus"
            elif same_label:
                group = "cross_seed_within_stimulus"
            else:
                group = "cross_seed_between_stimulus"
            output.append(
                {
                    "group": group,
                    "left_trial": left["trial_id"],
                    "right_trial": right["trial_id"],
                    "left_label": left["stimulus_label"],
                    "right_label": right["stimulus_label"],
                    "left_field_seed": int(left["field_seed"]),
                    "right_field_seed": int(right["field_seed"]),
                    "distance": float(
                        np.linalg.norm(matrix[left_index] - matrix[right_index])
                    ),
                }
            )
    return output


def summarize_diagnostic_distances(
    rows: Sequence[DiagnosticDistanceRow],
) -> dict[str, Any]:
    groups = {
        name: np.asarray(
            [float(row["distance"]) for row in rows if row["group"] == name],
            dtype=np.float64,
        )
        for name in (
            "matched_between_stimulus",
            "cross_seed_within_stimulus",
            "cross_seed_between_stimulus",
        )
    }
    if any(not values.size for values in groups.values()):
        msg = "diagnostic distance groups must not be empty"
        raise ValueError(msg)
    matched = groups["matched_between_stimulus"]
    within = groups["cross_seed_within_stimulus"]
    between = groups["cross_seed_between_stimulus"]
    return {
        "groups": {
            name: distribution_summary(values) for name, values in groups.items()
        },
        "matched_stimulus_to_seed_ratio": safe_ratio(
            float(np.mean(matched)),
            float(np.mean(within)),
        ),
        "cross_seed_separation_margin": float(np.mean(between) - np.mean(within)),
        "cross_seed_cliffs_delta": cliffs_delta(between, within),
    }


def paired_stimulus_permutation(
    rows: Sequence[StringRow],
    matrix: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    labels = np.asarray([row["stimulus_label"] for row in rows])
    field_seeds = np.asarray([int(row["field_seed"]) for row in rows])
    observed = multivariate_stimulus_ss(labels, field_seeds, matrix)
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        permuted = labels.copy()
        for field_seed in np.unique(field_seeds):
            mask = field_seeds == field_seed
            permuted[mask] = rng.permutation(permuted[mask])
        estimates[index] = multivariate_stimulus_ss(
            permuted,
            field_seeds,
            matrix,
        )
    return {
        "method": "stimulus_labels_permuted_within_field_seed",
        "samples": samples,
        "seed": seed,
        "observed_stimulus_ss": observed,
        "null_mean_stimulus_ss": float(np.mean(estimates)),
        "null_stimulus_ss_ci": [
            float(np.quantile(estimates, 0.025)),
            float(np.quantile(estimates, 0.975)),
        ],
        "p_value": float((1 + np.sum(estimates >= observed)) / (samples + 1)),
    }


def multivariate_stimulus_ss(
    labels: np.ndarray,
    field_seeds: np.ndarray,
    matrix: np.ndarray,
) -> float:
    grand_mean = np.mean(matrix, axis=0)
    unique_labels = np.unique(labels)
    return float(
        np.unique(field_seeds).size
        * sum(
            float(np.sum((np.mean(matrix[labels == label], axis=0) - grand_mean) ** 2))
            for label in unique_labels
        )
    )


def top_features(
    rows: Sequence[dict[str, str | float]],
    *,
    key: str,
    count: int = 10,
) -> list[dict[str, str | float]]:
    return sorted(rows, key=lambda row: float(row[key]), reverse=True)[:count]


def diagnostic_verdict(
    variance: dict[str, float],
    distances: dict[str, Any],
    permutation: dict[str, Any],
) -> dict[str, Any]:
    checks = {
        "stimulus_variance_exceeds_seed_variance": (
            variance["stimulus_to_seed_ratio"] > 1.0
        ),
        "matched_stimulus_distance_exceeds_seed_distance": (
            distances["matched_stimulus_to_seed_ratio"] > 1.0
        ),
        "cross_seed_between_exceeds_within": (
            distances["cross_seed_separation_margin"] > 0.0
        ),
        "paired_stimulus_effect_detected": permutation["p_value"] <= 0.05,
    }
    return {
        "stimulus_dominant": all(checks.values()),
        "checks": checks,
    }


def diagnostic_consensus(scenarios: Sequence[dict[str, Any]]) -> dict[str, Any]:
    stimulus_dominant = sum(
        bool(scenario["diagnosis"]["stimulus_dominant"]) for scenario in scenarios
    )
    return {
        "stimulus_dominant_scenarios": stimulus_dominant,
        "seed_or_residual_dominant_scenarios": len(scenarios) - stimulus_dominant,
        "all_scenarios_stimulus_dominant": stimulus_dominant == len(scenarios),
    }


def safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / max(denominator, 1e-12)


def write_rows(
    path: str | Path,
    rows: Sequence[dict[str, Any]],
) -> Path:
    if not rows:
        msg = "diagnostic rows must not be empty"
        raise ValueError(msg)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output


def write_diagnostic_plot(
    path: str | Path,
    variance: dict[str, float],
    distances: Sequence[DiagnosticDistanceRow],
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    axes[0].bar(
        ["stimulus", "field seed", "residual"],
        [
            variance["stimulus_fraction"],
            variance["field_seed_fraction"],
            variance["residual_fraction"],
        ],
    )
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("fraction of standardized variance")
    axes[0].set_title("Embedding variance decomposition")
    group_names = (
        "matched_between_stimulus",
        "cross_seed_within_stimulus",
        "cross_seed_between_stimulus",
    )
    values = [
        [float(row["distance"]) for row in distances if row["group"] == name]
        for name in group_names
    ]
    axes[1].boxplot(
        values,
        tick_labels=["matched\nbetween", "cross-seed\nwithin", "cross-seed\nbetween"],
    )
    axes[1].set_ylabel("train-standardized Euclidean distance")
    axes[1].set_title("Stimulus and seed distance scales")
    figure.savefig(output, dpi=140)
    plt.close(figure)
    return output


def write_json(path: str | Path, value: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2), encoding="utf-8")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnose stimulus and seed effects in recorded embeddings.",
    )
    parser.add_argument(
        "--stage-root",
        type=Path,
        default=Path("experiments/distinguishability_stage_one"),
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=Path("experiments/logs/distinguishability_stage_one_diagnostics.json"),
    )
    parser.add_argument("--permutation-samples", type=int, default=2_000)
    parser.add_argument("--permutation-seed", type=int, default=20_260_813)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = diagnose_stage_one(
        args.stage_root,
        args.output_report,
        permutation_samples=args.permutation_samples,
        permutation_seed=args.permutation_seed,
    )
    print(json.dumps(report, indent=2))
    return 0
