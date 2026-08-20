from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from neuroacoustic_resonator.analysis.distinguishability_classification import (
    balanced_accuracy,
    fit_nearest_centroid,
)
from neuroacoustic_resonator.analysis.distinguishability_statistics import (
    fit_train_standardizer,
)
from neuroacoustic_resonator.analysis.protocol_embeddings import read_embedding_rows

StringRow = dict[str, str]


@dataclass(frozen=True)
class CrossSeedFold:
    train_indices: np.ndarray
    test_indices: np.ndarray
    train_matrix: np.ndarray
    test_matrix: np.ndarray


def evaluate_paired_causal_evidence(
    embeddings_csv: str | Path,
    output_report: str | Path,
    *,
    permutation_samples: int = 10_000,
    permutation_seed: int = 20_260_818,
    bootstrap_samples: int = 5_000,
    bootstrap_seed: int = 20_260_819,
) -> dict[str, Any]:
    if permutation_samples < 1:
        msg = "permutation_samples must be positive"
        raise ValueError(msg)
    if bootstrap_samples < 1:
        msg = "bootstrap_samples must be positive"
        raise ValueError(msg)
    rows = read_embedding_rows(embeddings_csv)
    design = validate_paired_causal_design(rows)
    roots = design["seed_roots"]
    labels = tuple(design["stimulus_labels"])
    actual = np.asarray([row["stimulus_label"] for row in rows])
    folds = prepare_cross_seed_folds(rows, roots)
    predicted = predict_cross_seed_folds(folds, actual)
    observed = balanced_accuracy(actual, predicted, labels)
    by_root = grouped_classification(rows, actual, predicted, labels, "seed_root")
    by_checkpoint = grouped_classification(
        rows,
        actual,
        predicted,
        labels,
        "field_seed",
    )
    bootstrap = checkpoint_clustered_bootstrap(
        rows,
        labels,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    permutation = checkpoint_label_permutation(
        rows,
        labels,
        observed=observed,
        samples=permutation_samples,
        seed=permutation_seed,
    )
    report = {
        "source_embeddings": str(embeddings_csv),
        "design": design,
        "classification": {
            "method": "leave_one_seed_root_out_nearest_centroid",
            "balanced_accuracy": observed,
            "chance_level": 1.0 / len(labels),
            "correct": int(np.count_nonzero(actual == predicted)),
            "samples": len(rows),
            "by_seed_root": by_root,
            "by_checkpoint": by_checkpoint,
            "roots_above_chance": sum(
                item["balanced_accuracy"] > 1.0 / len(labels) for item in by_root
            ),
            "checkpoint_clustered_bootstrap": bootstrap,
            "checkpoint_label_permutation": permutation,
        },
        "limitations": {
            "confirmatory": False,
            "seed_roots": len(roots),
            "checkpoint_clusters": design["checkpoint_count"],
            "reason": "pilot corpus requires replication on more unseen seed roots",
        },
    }
    write_json(output_report, report)
    return report


def validate_paired_causal_design(rows: Sequence[StringRow]) -> dict[str, Any]:
    labels = sorted({row["stimulus_label"] for row in rows})
    roots = sorted({int(row["seed_root"]) for row in rows})
    if len(labels) < 2:
        msg = "paired causal evidence requires at least two stimulus labels"
        raise ValueError(msg)
    if len(roots) < 3:
        msg = "paired causal evidence requires at least three seed roots"
        raise ValueError(msg)
    labels_by_checkpoint: dict[int, list[str]] = defaultdict(list)
    root_by_checkpoint: dict[int, set[int]] = defaultdict(set)
    checkpoints_by_root: dict[int, set[int]] = defaultdict(set)
    for row in rows:
        checkpoint = int(row["field_seed"])
        root = int(row["seed_root"])
        labels_by_checkpoint[checkpoint].append(row["stimulus_label"])
        root_by_checkpoint[checkpoint].add(root)
        checkpoints_by_root[root].add(checkpoint)
    if any(
        len(checkpoint_roots) != 1 for checkpoint_roots in root_by_checkpoint.values()
    ):
        msg = "each checkpoint must belong to exactly one seed root"
        raise ValueError(msg)
    if any(
        sorted(checkpoint_labels) != labels
        for checkpoint_labels in labels_by_checkpoint.values()
    ):
        msg = "every checkpoint must contain each stimulus label exactly once"
        raise ValueError(msg)
    counts = {
        root: len(checkpoints) for root, checkpoints in checkpoints_by_root.items()
    }
    if len(set(counts.values())) != 1:
        msg = "every seed root must contain the same number of checkpoints"
        raise ValueError(msg)
    return {
        "independence_unit": "field_seed_checkpoint",
        "shared_silence_control_within_checkpoint": True,
        "rows": len(rows),
        "stimulus_labels": labels,
        "seed_roots": roots,
        "checkpoint_count": len(labels_by_checkpoint),
        "checkpoints_per_seed_root": next(iter(counts.values())),
        "rows_per_checkpoint": len(labels),
    }


def cross_seed_predictions(
    rows: list[StringRow],
    roots: list[int],
    labels: np.ndarray,
) -> np.ndarray:
    return predict_cross_seed_folds(prepare_cross_seed_folds(rows, roots), labels)


def prepare_cross_seed_folds(
    rows: list[StringRow],
    roots: list[int],
) -> list[CrossSeedFold]:
    folds: list[CrossSeedFold] = []
    row_roots = np.asarray([int(row["seed_root"]) for row in rows])
    for held_out in roots:
        train_indices = np.flatnonzero(row_roots != held_out)
        test_indices = np.flatnonzero(row_roots == held_out)
        training_rows = [{**rows[index], "split": "train"} for index in train_indices]
        standardizer = fit_train_standardizer(training_rows)
        folds.append(
            CrossSeedFold(
                train_indices=train_indices,
                test_indices=test_indices,
                train_matrix=standardizer.transform(training_rows),
                test_matrix=standardizer.transform(
                    [rows[index] for index in test_indices]
                ),
            )
        )
    return folds


def predict_cross_seed_folds(
    folds: Sequence[CrossSeedFold],
    labels: np.ndarray,
) -> np.ndarray:
    predicted = np.empty(labels.shape, dtype=labels.dtype)
    for fold in folds:
        model = fit_nearest_centroid(
            fold.train_matrix,
            labels[fold.train_indices],
            "euclidean",
        )
        predicted[fold.test_indices], _ = model.predict(fold.test_matrix)
    return predicted


def grouped_classification(
    rows: Sequence[StringRow],
    actual: np.ndarray,
    predicted: np.ndarray,
    labels: Sequence[str],
    column: str,
) -> list[dict[str, Any]]:
    groups = sorted({int(row[column]) for row in rows})
    values = np.asarray([int(row[column]) for row in rows])
    return [
        {
            column: group,
            "samples": int(np.count_nonzero(values == group)),
            "correct": int(np.count_nonzero((values == group) & (actual == predicted))),
            "balanced_accuracy": balanced_accuracy(
                actual[values == group],
                predicted[values == group],
                labels,
            ),
        }
        for group in groups
    ]


def checkpoint_clustered_bootstrap(
    rows: list[StringRow],
    labels: Sequence[str],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    roots = sorted({int(row["seed_root"]) for row in rows})
    checkpoint_rows = rows_by_checkpoint(rows)
    checkpoints_by_root = {
        root: sorted(
            checkpoint
            for checkpoint, group in checkpoint_rows.items()
            if int(group[0]["seed_root"]) == root
        )
        for root in roots
    }
    root_resamples = [
        list(product(checkpoints_by_root[root], repeat=len(checkpoints_by_root[root])))
        for root in roots
    ]
    exhaustive_samples = int(np.prod([len(options) for options in root_resamples]))
    exhaustive = exhaustive_samples <= samples
    rng = np.random.default_rng(seed)
    if exhaustive:
        resamples = list(product(*root_resamples))
    else:
        resamples = [
            tuple(
                tuple(
                    int(item)
                    for item in rng.choice(
                        checkpoints_by_root[root],
                        size=len(checkpoints_by_root[root]),
                        replace=True,
                    )
                )
                for root in roots
            )
            for _ in range(samples)
        ]
    scores = np.empty(len(resamples), dtype=np.float64)
    for sample_index, root_samples in enumerate(resamples):
        sampled_rows: list[StringRow] = []
        for root, sampled in zip(roots, root_samples, strict=True):
            for copy_index, checkpoint in enumerate(sampled):
                for row in checkpoint_rows[int(checkpoint)]:
                    sampled_rows.append(
                        {
                            **row,
                            "field_seed": f"{root}{sample_index}{copy_index}",
                        }
                    )
        actual = np.asarray([row["stimulus_label"] for row in sampled_rows])
        predicted = cross_seed_predictions(sampled_rows, roots, actual)
        scores[sample_index] = balanced_accuracy(actual, predicted, labels)
    return {
        "method": (
            "exhaustive_checkpoint_cluster_bootstrap"
            if exhaustive
            else "monte_carlo_checkpoint_cluster_bootstrap"
        ),
        "requested_samples": samples,
        "samples": len(resamples),
        "seed": seed,
        "mean": float(np.mean(scores)),
        "confidence_level": 0.95,
        "balanced_accuracy_ci": [
            float(np.quantile(scores, 0.025)),
            float(np.quantile(scores, 0.975)),
        ],
        "resampling_unit": "field_seed_checkpoint",
    }


def checkpoint_label_permutation(
    rows: list[StringRow],
    labels: Sequence[str],
    *,
    observed: float,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    roots = sorted({int(row["seed_root"]) for row in rows})
    actual = np.asarray([row["stimulus_label"] for row in rows])
    checkpoints = np.asarray([int(row["field_seed"]) for row in rows])
    rng = np.random.default_rng(seed)
    folds = prepare_cross_seed_folds(rows, roots)
    scores = np.empty(samples, dtype=np.float64)
    for sample_index in range(samples):
        permuted = actual.copy()
        for checkpoint in np.unique(checkpoints):
            indices = np.flatnonzero(checkpoints == checkpoint)
            permuted[indices] = rng.permutation(permuted[indices])
        predicted = predict_cross_seed_folds(folds, permuted)
        scores[sample_index] = balanced_accuracy(permuted, predicted, labels)
    return {
        "method": "stimulus_labels_permuted_within_checkpoint",
        "preserves_shared_control_dependence": True,
        "samples": samples,
        "seed": seed,
        "observed_balanced_accuracy": observed,
        "null_mean": float(np.mean(scores)),
        "null_balanced_accuracy_ci": [
            float(np.quantile(scores, 0.025)),
            float(np.quantile(scores, 0.975)),
        ],
        "p_value": float((1 + np.count_nonzero(scores >= observed)) / (samples + 1)),
    }


def rows_by_checkpoint(rows: Sequence[StringRow]) -> dict[int, list[StringRow]]:
    grouped: dict[int, list[StringRow]] = defaultdict(list)
    for row in rows:
        grouped[int(row["field_seed"])].append(row)
    return grouped


def write_json(path: str | Path, value: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2), encoding="utf-8")
    return output
