from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from neuroacoustic_resonator.analysis.protocol_embeddings import (
    FEATURE_COLUMNS,
    read_embedding_rows,
)

StringRow = dict[str, str]
DistanceRow = dict[str, str | int | float]


@dataclass(frozen=True)
class TrainStandardizer:
    feature_names: tuple[str, ...]
    dropped_features: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray

    def transform(self, rows: Sequence[StringRow]) -> np.ndarray:
        matrix = feature_matrix(rows, self.feature_names)
        return (matrix - self.mean) / self.scale

    def to_dict(self) -> dict[str, Any]:
        return {
            "fit_split": "train",
            "feature_names": list(self.feature_names),
            "dropped_features": list(self.dropped_features),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
        }


def quantify_response_separation(
    embeddings_csv: str | Path,
    output_report: str | Path,
    output_pairs: str | Path,
    output_plot: str | Path,
    *,
    bootstrap_samples: int = 5_000,
    bootstrap_seed: int = 20_260_811,
) -> dict[str, Any]:
    rows = read_embedding_rows(embeddings_csv)
    standardizer = fit_train_standardizer(rows)
    normalized = standardizer.transform(rows)
    pairs = build_distance_rows(rows, normalized)
    report = separation_report(
        pairs,
        standardizer,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    report["source_embeddings"] = str(embeddings_csv)
    report["outputs"] = {
        "pairs_csv": str(output_pairs),
        "plot": str(output_plot),
    }
    write_distance_rows(output_pairs, pairs)
    write_distance_plot(output_plot, pairs)
    write_json(output_report, report)
    return report


def fit_train_standardizer(rows: Sequence[StringRow]) -> TrainStandardizer:
    train_rows = [row for row in rows if row["split"] == "train"]
    if not train_rows:
        msg = "embedding table must contain train rows"
        raise ValueError(msg)
    matrix = feature_matrix(train_rows, FEATURE_COLUMNS)
    mean = np.mean(matrix, axis=0)
    scale = np.std(matrix, axis=0)
    active = scale > 1e-12
    if not np.any(active):
        msg = "train embeddings have no varying protocol features"
        raise ValueError(msg)
    feature_names = tuple(
        name
        for name, is_active in zip(FEATURE_COLUMNS, active, strict=True)
        if is_active
    )
    dropped = tuple(
        name
        for name, is_active in zip(FEATURE_COLUMNS, active, strict=True)
        if not is_active
    )
    return TrainStandardizer(
        feature_names=feature_names,
        dropped_features=dropped,
        mean=mean[active],
        scale=scale[active],
    )


def feature_matrix(
    rows: Sequence[StringRow],
    feature_names: Sequence[str],
) -> np.ndarray:
    try:
        matrix = np.asarray(
            [[float(row[name]) for name in feature_names] for row in rows],
            dtype=np.float64,
        )
    except (KeyError, ValueError) as exc:
        raise ValueError("embedding table contains an invalid feature value") from exc
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        msg = "embedding feature matrix must be finite and two-dimensional"
        raise ValueError(msg)
    return matrix


def build_distance_rows(
    rows: Sequence[StringRow],
    normalized: np.ndarray,
) -> list[DistanceRow]:
    if normalized.shape[0] != len(rows):
        msg = "embedding rows and normalized matrix must have matching lengths"
        raise ValueError(msg)
    grouped: dict[int, list[tuple[StringRow, np.ndarray]]] = defaultdict(list)
    for row, vector in zip(rows, normalized, strict=True):
        grouped[int(row["seed_root"])].append((row, vector))
    pairs: list[DistanceRow] = []
    for seed_root, group in sorted(grouped.items()):
        splits = {row["split"] for row, _ in group}
        if len(splits) != 1:
            msg = f"seed root {seed_root} crosses data splits"
            raise ValueError(msg)
        split = next(iter(splits))
        by_label: dict[str, list[np.ndarray]] = defaultdict(list)
        for row, vector in group:
            by_label[row["stimulus_label"]].append(vector)
        if len(by_label) < 2:
            msg = f"seed root {seed_root} must contain at least two stimuli"
            raise ValueError(msg)
        for label, vectors in sorted(by_label.items()):
            if len(vectors) < 2:
                msg = f"seed root {seed_root} stimulus {label} needs repeated trials"
                raise ValueError(msg)
            for left_index in range(len(vectors)):
                for right_index in range(left_index + 1, len(vectors)):
                    pairs.append(
                        distance_row(
                            seed_root,
                            split,
                            "within",
                            label,
                            label,
                            vectors[left_index],
                            vectors[right_index],
                        )
                    )
        centroids = {
            label: np.mean(vectors, axis=0) for label, vectors in by_label.items()
        }
        labels = sorted(centroids)
        for left_index in range(len(labels)):
            for right_index in range(left_index + 1, len(labels)):
                left_label = labels[left_index]
                right_label = labels[right_index]
                pairs.append(
                    distance_row(
                        seed_root,
                        split,
                        "between",
                        left_label,
                        right_label,
                        centroids[left_label],
                        centroids[right_label],
                    )
                )
    return pairs


def distance_row(
    seed_root: int,
    split: str,
    pair_type: str,
    left_label: str,
    right_label: str,
    left: np.ndarray,
    right: np.ndarray,
) -> DistanceRow:
    return {
        "seed_root": seed_root,
        "split": split,
        "pair_type": pair_type,
        "left_label": left_label,
        "right_label": right_label,
        "distance": float(np.linalg.norm(left - right)),
    }


def separation_report(
    pairs: Sequence[DistanceRow],
    standardizer: TrainStandardizer,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    if bootstrap_samples < 1:
        msg = "bootstrap_samples must be positive"
        raise ValueError(msg)
    rng = np.random.default_rng(bootstrap_seed)
    return {
        "standardization": standardizer.to_dict(),
        "bootstrap": {
            "unit": "seed_root",
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
            "confidence": 0.95,
        },
        "overall": summarize_pairs(pairs, bootstrap_samples, rng),
        "by_split": {
            split: summarize_pairs(
                [pair for pair in pairs if pair["split"] == split],
                bootstrap_samples,
                rng,
            )
            for split in ("train", "validation", "test")
        },
        "by_seed": summarize_by_seed(pairs),
    }


def summarize_pairs(
    pairs: Sequence[DistanceRow],
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    if not pairs:
        return {"available": False}
    within = distance_values(pairs, "within")
    between = distance_values(pairs, "between")
    if not within.size or not between.size:
        return {"available": False}
    margin = float(np.mean(between) - np.mean(within))
    ratio = safe_ratio(float(np.mean(between)), float(np.mean(within)))
    effect = cliffs_delta(between, within)
    roots = sorted({int(pair["seed_root"]) for pair in pairs})
    by_root = {
        root: [pair for pair in pairs if int(pair["seed_root"]) == root]
        for root in roots
    }
    margin_ci = clustered_bootstrap_ci(
        by_root,
        bootstrap_samples,
        rng,
        separation_margin,
    )
    effect_ci = clustered_bootstrap_ci(
        by_root,
        bootstrap_samples,
        rng,
        separation_cliffs_delta,
    )
    return {
        "available": True,
        "seed_roots": roots,
        "within": distribution_summary(within),
        "between": distribution_summary(between),
        "separation_margin": margin,
        "separation_margin_ci": margin_ci,
        "separation_ratio": ratio,
        "cliffs_delta": effect,
        "cliffs_delta_ci": effect_ci,
    }


def summarize_by_seed(pairs: Sequence[DistanceRow]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for seed_root in sorted({int(pair["seed_root"]) for pair in pairs}):
        subset = [pair for pair in pairs if int(pair["seed_root"]) == seed_root]
        within = distance_values(subset, "within")
        between = distance_values(subset, "between")
        summaries.append(
            {
                "seed_root": seed_root,
                "split": str(subset[0]["split"]),
                "within_mean": float(np.mean(within)),
                "between_mean": float(np.mean(between)),
                "separation_margin": float(np.mean(between) - np.mean(within)),
                "separation_ratio": safe_ratio(
                    float(np.mean(between)),
                    float(np.mean(within)),
                ),
            }
        )
    return summaries


def distance_values(
    pairs: Sequence[DistanceRow],
    pair_type: str,
) -> np.ndarray:
    return np.asarray(
        [float(pair["distance"]) for pair in pairs if pair["pair_type"] == pair_type],
        dtype=np.float64,
    )


def distribution_summary(values: np.ndarray) -> dict[str, float | int]:
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def cliffs_delta(left: np.ndarray, right: np.ndarray) -> float:
    if not left.size or not right.size:
        return float("nan")
    differences = left[:, None] - right[None, :]
    return float((np.sum(differences > 0) - np.sum(differences < 0)) / differences.size)


def separation_margin(pairs: Sequence[DistanceRow]) -> float:
    within = distance_values(pairs, "within")
    between = distance_values(pairs, "between")
    return float(np.mean(between) - np.mean(within))


def separation_cliffs_delta(pairs: Sequence[DistanceRow]) -> float:
    return cliffs_delta(
        distance_values(pairs, "between"),
        distance_values(pairs, "within"),
    )


def clustered_bootstrap_ci(
    pairs_by_root: dict[int, list[DistanceRow]],
    samples: int,
    rng: np.random.Generator,
    statistic: Callable[[Sequence[DistanceRow]], float],
) -> list[float]:
    roots = np.asarray(sorted(pairs_by_root), dtype=np.int64)
    estimates = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        selected = rng.choice(roots, size=roots.size, replace=True)
        resampled = [pair for root in selected for pair in pairs_by_root[int(root)]]
        estimates[index] = statistic(resampled)
    return [
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    ]


def safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 1e-12:
        return None
    return numerator / denominator


def write_distance_rows(path: str | Path, pairs: Sequence[DistanceRow]) -> Path:
    if not pairs:
        msg = "distance rows must not be empty"
        raise ValueError(msg)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)
    return output


def write_distance_plot(path: str | Path, pairs: Sequence[DistanceRow]) -> Path:
    within = distance_values(pairs, "within")
    between = distance_values(pairs, "between")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
    axis.boxplot([within, between], tick_labels=["within", "between"])
    axis.set_title("Sound Protocol response separation")
    axis.set_ylabel("train-standardized Euclidean distance")
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
        description="Measure within- and between-stimulus protocol distances.",
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=Path("experiments/logs/distinguishability_embeddings.csv"),
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=Path("experiments/logs/distinguishability_separation.json"),
    )
    parser.add_argument(
        "--output-pairs",
        type=Path,
        default=Path("experiments/logs/distinguishability_distance_pairs.csv"),
    )
    parser.add_argument(
        "--output-plot",
        type=Path,
        default=Path("experiments/logs/distinguishability_separation.png"),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=5_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_811)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = quantify_response_separation(
        args.embeddings,
        args.output_report,
        args.output_pairs,
        args.output_plot,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(json.dumps(report, indent=2))
    return 0
