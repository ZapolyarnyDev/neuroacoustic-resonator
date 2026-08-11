from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np

from neuroacoustic_resonator.analysis.distinguishability_statistics import (
    fit_train_standardizer,
)
from neuroacoustic_resonator.analysis.protocol_embeddings import read_embedding_rows

DistanceMetric = Literal["euclidean", "cosine"]
StringRow = dict[str, str]
PredictionRow = dict[str, str | int | float | bool]


@dataclass(frozen=True)
class NearestCentroidModel:
    labels: tuple[str, ...]
    centroids: np.ndarray
    metric: DistanceMetric

    def predict(self, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        distances = distance_matrix(matrix, self.centroids, self.metric)
        indices = np.argmin(distances, axis=1)
        predictions = np.asarray([self.labels[index] for index in indices])
        selected_distances = distances[np.arange(matrix.shape[0]), indices]
        return predictions, selected_distances


def classify_protocol_embeddings(
    embeddings_csv: str | Path,
    output_report: str | Path,
    output_predictions: str | Path,
    output_plot: str | Path,
    *,
    permutation_samples: int = 2_000,
    permutation_seed: int = 20_260_811,
    bootstrap_samples: int = 5_000,
    bootstrap_seed: int = 20_260_812,
) -> dict[str, Any]:
    if permutation_samples < 1:
        msg = "permutation_samples must be positive"
        raise ValueError(msg)
    if bootstrap_samples < 1:
        msg = "bootstrap_samples must be positive"
        raise ValueError(msg)
    rows = read_embedding_rows(embeddings_csv)
    validate_splits(rows)
    standardizer = fit_train_standardizer(rows)
    matrix = standardizer.transform(rows)
    split_indices = {
        split: np.asarray(
            [index for index, row in enumerate(rows) if row["split"] == split],
            dtype=np.int64,
        )
        for split in ("train", "validation", "test")
    }
    train_indices = split_indices["train"]
    validation_indices = split_indices["validation"]
    test_indices = split_indices["test"]
    train_labels = np.asarray(
        [rows[index]["stimulus_label"] for index in train_indices]
    )
    candidate_metrics: tuple[DistanceMetric, ...] = ("euclidean", "cosine")
    candidate_scores: dict[str, float] = {}
    for metric in candidate_metrics:
        candidate = fit_nearest_centroid(
            matrix[train_indices],
            train_labels,
            metric,
        )
        predicted, _ = candidate.predict(matrix[validation_indices])
        candidate_scores[metric] = balanced_accuracy(
            labels_for_indices(rows, validation_indices),
            predicted,
            candidate.labels,
        )
    selected_metric = max(
        candidate_metrics,
        key=lambda metric: (candidate_scores[metric], metric == "euclidean"),
    )
    model = fit_nearest_centroid(
        matrix[train_indices],
        train_labels,
        selected_metric,
    )
    validation_predictions, validation_distances = model.predict(
        matrix[validation_indices]
    )
    test_predictions, test_distances = model.predict(matrix[test_indices])
    prediction_rows = [
        *build_prediction_rows(
            rows,
            validation_indices,
            validation_predictions,
            validation_distances,
        ),
        *build_prediction_rows(
            rows,
            test_indices,
            test_predictions,
            test_distances,
        ),
    ]
    validation_actual = labels_for_indices(rows, validation_indices)
    test_actual = labels_for_indices(rows, test_indices)
    test_confusion = confusion_matrix(test_actual, test_predictions, model.labels)
    test_balanced_accuracy = balanced_accuracy(
        test_actual,
        test_predictions,
        model.labels,
    )
    report = {
        "source_embeddings": str(embeddings_csv),
        "labels": list(model.labels),
        "standardization": standardizer.to_dict(),
        "selection": {
            "split": "validation",
            "candidate_balanced_accuracy": candidate_scores,
            "selected_metric": selected_metric,
        },
        "validation": classification_summary(
            validation_actual,
            validation_predictions,
            model.labels,
        ),
        "test": {
            **classification_summary(test_actual, test_predictions, model.labels),
            "seed_roots": sorted(
                {int(rows[index]["seed_root"]) for index in test_indices}
            ),
            "balanced_accuracy_ci": clustered_accuracy_ci(
                [rows[index] for index in test_indices],
                test_actual,
                test_predictions,
                model.labels,
                samples=bootstrap_samples,
                seed=bootstrap_seed,
            ),
            "confusion_matrix": test_confusion.tolist(),
        },
        "permutation_baseline": permutation_baseline(
            matrix[train_indices],
            train_labels,
            np.asarray([int(rows[index]["seed_root"]) for index in train_indices]),
            matrix[test_indices],
            test_actual,
            model.labels,
            selected_metric,
            observed=test_balanced_accuracy,
            samples=permutation_samples,
            seed=permutation_seed,
        ),
        "outputs": {
            "predictions_csv": str(output_predictions),
            "confusion_plot": str(output_plot),
        },
    }
    write_prediction_rows(output_predictions, prediction_rows)
    write_confusion_plot(output_plot, test_confusion, model.labels)
    write_json(output_report, report)
    return report


def validate_splits(rows: Sequence[StringRow]) -> None:
    split_labels = {
        split: {row["stimulus_label"] for row in rows if row["split"] == split}
        for split in ("train", "validation", "test")
    }
    if any(not labels for labels in split_labels.values()):
        msg = "classification requires non-empty train, validation, and test splits"
        raise ValueError(msg)
    if not (
        split_labels["train"] == split_labels["validation"] == split_labels["test"]
    ):
        msg = "every data split must contain the same stimulus labels"
        raise ValueError(msg)
    roots_by_split = {
        split: {row["seed_root"] for row in rows if row["split"] == split}
        for split in ("train", "validation", "test")
    }
    if any(
        roots_by_split[left] & roots_by_split[right]
        for left, right in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        )
    ):
        msg = "seed roots must not cross classification splits"
        raise ValueError(msg)


def fit_nearest_centroid(
    matrix: np.ndarray,
    labels: np.ndarray,
    metric: DistanceMetric,
) -> NearestCentroidModel:
    unique_labels = tuple(sorted(str(label) for label in np.unique(labels)))
    if len(unique_labels) < 2:
        msg = "nearest-centroid classification requires at least two labels"
        raise ValueError(msg)
    centroids = np.asarray(
        [np.mean(matrix[labels == label], axis=0) for label in unique_labels],
        dtype=np.float64,
    )
    return NearestCentroidModel(unique_labels, centroids, metric)


def distance_matrix(
    samples: np.ndarray,
    centroids: np.ndarray,
    metric: DistanceMetric,
) -> np.ndarray:
    if metric == "euclidean":
        return np.linalg.norm(samples[:, None, :] - centroids[None, :, :], axis=2)
    if metric == "cosine":
        numerator = samples @ centroids.T
        denominator = (
            np.linalg.norm(samples, axis=1)[:, None]
            * np.linalg.norm(
                centroids,
                axis=1,
            )[None, :]
        )
        similarity = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 1e-12,
        )
        return 1.0 - similarity
    msg = f"unsupported distance metric: {metric}"
    raise ValueError(msg)


def labels_for_indices(rows: Sequence[StringRow], indices: np.ndarray) -> np.ndarray:
    return np.asarray([rows[index]["stimulus_label"] for index in indices])


def balanced_accuracy(
    actual: np.ndarray,
    predicted: np.ndarray,
    labels: Sequence[str],
) -> float:
    recalls = []
    for label in labels:
        mask = actual == label
        if not np.any(mask):
            msg = f"balanced accuracy is missing label {label}"
            raise ValueError(msg)
        recalls.append(float(np.mean(predicted[mask] == label)))
    return float(np.mean(recalls))


def classification_summary(
    actual: np.ndarray,
    predicted: np.ndarray,
    labels: Sequence[str],
) -> dict[str, Any]:
    correct = predicted == actual
    per_class = {
        label: float(np.mean(predicted[actual == label] == label)) for label in labels
    }
    return {
        "samples": int(actual.size),
        "accuracy": float(np.mean(correct)),
        "balanced_accuracy": float(np.mean(list(per_class.values()))),
        "per_class_recall": per_class,
    }


def confusion_matrix(
    actual: np.ndarray,
    predicted: np.ndarray,
    labels: Sequence[str],
) -> np.ndarray:
    label_indices = {label: index for index, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for actual_label, predicted_label in zip(actual, predicted, strict=True):
        matrix[
            label_indices[str(actual_label)], label_indices[str(predicted_label)]
        ] += 1
    return matrix


def build_prediction_rows(
    rows: Sequence[StringRow],
    indices: np.ndarray,
    predicted: np.ndarray,
    distances: np.ndarray,
) -> list[PredictionRow]:
    return [
        {
            "trial_id": rows[index]["trial_id"],
            "split": rows[index]["split"],
            "seed_root": int(rows[index]["seed_root"]),
            "repeat_index": int(rows[index]["repeat_index"]),
            "actual_label": rows[index]["stimulus_label"],
            "predicted_label": str(prediction),
            "distance": float(distance),
            "correct": rows[index]["stimulus_label"] == prediction,
        }
        for index, prediction, distance in zip(
            indices,
            predicted,
            distances,
            strict=True,
        )
    ]


def clustered_accuracy_ci(
    rows: Sequence[StringRow],
    actual: np.ndarray,
    predicted: np.ndarray,
    labels: Sequence[str],
    *,
    samples: int,
    seed: int,
) -> list[float]:
    by_root: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_root[int(row["seed_root"])].append(index)
    roots = np.asarray(sorted(by_root), dtype=np.int64)
    if roots.size < 2:
        msg = "test accuracy confidence interval requires at least two seed roots"
        raise ValueError(msg)
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    for sample_index in range(samples):
        selected_roots = rng.choice(roots, size=roots.size, replace=True)
        selected_indices = np.asarray(
            [index for root in selected_roots for index in by_root[int(root)]],
            dtype=np.int64,
        )
        estimates[sample_index] = balanced_accuracy(
            actual[selected_indices],
            predicted[selected_indices],
            labels,
        )
    return [
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    ]


def permutation_baseline(
    train_matrix: np.ndarray,
    train_labels: np.ndarray,
    train_seed_roots: np.ndarray,
    test_matrix: np.ndarray,
    test_labels: np.ndarray,
    labels: Sequence[str],
    metric: DistanceMetric,
    *,
    observed: float,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    scores = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        permuted = permute_labels_within_seed_roots(
            train_labels,
            train_seed_roots,
            rng,
        )
        model = fit_nearest_centroid(train_matrix, permuted, metric)
        predicted, _ = model.predict(test_matrix)
        scores[index] = balanced_accuracy(test_labels, predicted, labels)
    return {
        "method": "permuted_labels_within_seed_root",
        "samples": samples,
        "seed": seed,
        "mean_balanced_accuracy": float(np.mean(scores)),
        "balanced_accuracy_ci": [
            float(np.quantile(scores, 0.025)),
            float(np.quantile(scores, 0.975)),
        ],
        "p_value": float((1 + np.sum(scores >= observed)) / (samples + 1)),
    }


def permute_labels_within_seed_roots(
    labels: np.ndarray,
    seed_roots: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    if labels.shape != seed_roots.shape:
        msg = "train labels and seed roots must have matching shapes"
        raise ValueError(msg)
    permuted = labels.copy()
    expected_labels = tuple(sorted(str(label) for label in np.unique(labels)))
    for seed_root in np.unique(seed_roots):
        mask = seed_roots == seed_root
        root_labels = tuple(sorted(str(label) for label in np.unique(labels[mask])))
        if root_labels != expected_labels:
            msg = f"train seed root {seed_root} does not contain every stimulus label"
            raise ValueError(msg)
        shuffled = rng.permutation(root_labels)
        mapping = dict(zip(root_labels, shuffled, strict=True))
        permuted[mask] = [mapping[str(label)] for label in labels[mask]]
    return permuted


def write_prediction_rows(
    path: str | Path,
    rows: Sequence[PredictionRow],
) -> Path:
    if not rows:
        msg = "classification predictions must not be empty"
        raise ValueError(msg)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output


def write_confusion_plot(
    path: str | Path,
    matrix: np.ndarray,
    labels: Sequence[str],
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    axis.set_xlabel("predicted")
    axis.set_ylabel("actual")
    axis.set_title("Held-out Sound Protocol classification")
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            axis.text(
                column_index,
                row_index,
                str(matrix[row_index, column_index]),
                ha="center",
                va="center",
            )
    figure.colorbar(image, ax=axis)
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
        description="Run held-out classification of Sound Protocol embeddings.",
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=Path("experiments/logs/distinguishability_embeddings.csv"),
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=Path("experiments/logs/distinguishability_classification.json"),
    )
    parser.add_argument(
        "--output-predictions",
        type=Path,
        default=Path("experiments/logs/distinguishability_predictions.csv"),
    )
    parser.add_argument(
        "--output-plot",
        type=Path,
        default=Path("experiments/logs/distinguishability_confusion.png"),
    )
    parser.add_argument("--permutation-samples", type=int, default=2_000)
    parser.add_argument("--permutation-seed", type=int, default=20_260_811)
    parser.add_argument("--bootstrap-samples", type=int, default=5_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_812)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = classify_protocol_embeddings(
        args.embeddings,
        args.output_report,
        args.output_predictions,
        args.output_plot,
        permutation_samples=args.permutation_samples,
        permutation_seed=args.permutation_seed,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(json.dumps(report, indent=2))
    return 0
