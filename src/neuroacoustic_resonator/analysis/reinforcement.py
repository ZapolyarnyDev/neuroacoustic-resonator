from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PatternReinforcementWeights:
    stability: float = 0.30
    repeatability: float = 0.25
    reactivity: float = 0.25
    diversity: float = 0.20
    anti_collapse: float = 0.30

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if value < 0.0:
                msg = f"{name} weight must be non-negative"
                raise ValueError(msg)


@dataclass(frozen=True)
class PatternReinforcementSignals:
    stability: float
    repeatability: float
    reactivity: float
    diversity: float
    anti_collapse_penalty: float
    reward: float
    sample_count: int
    stimulus_count: int
    dominant_label: str | None
    label_entropy: float
    label_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_pattern_reinforcement_signals(
    rows: list[dict[str, Any]],
    *,
    weights: PatternReinforcementWeights | None = None,
) -> PatternReinforcementSignals:
    if weights is None:
        weights = PatternReinforcementWeights()
    if not rows:
        return PatternReinforcementSignals(
            stability=0.0,
            repeatability=0.0,
            reactivity=0.0,
            diversity=0.0,
            anti_collapse_penalty=0.0,
            reward=0.0,
            sample_count=0,
            stimulus_count=0,
            dominant_label=None,
            label_entropy=0.0,
            label_counts={},
        )

    labels = [str(row.get("response_active_pattern_label") or "none") for row in rows]
    label_counts = count_labels(labels)
    dominant = dominant_label(label_counts)
    entropy = normalized_entropy(label_counts)
    stability = mean_bounded(
        float(row.get("response_pattern_confidence", 0.0)) for row in rows
    )
    reactivity = response_reactivity(rows)
    repeatability = response_repeatability(rows)
    diversity = entropy
    anti_collapse = anti_collapse_penalty(label_counts, total=len(rows))
    reward = (
        weights.stability * stability
        + weights.repeatability * repeatability
        + weights.reactivity * reactivity
        + weights.diversity * diversity
        - weights.anti_collapse * anti_collapse
    )
    reward = float(np.clip(reward, -1.0, 1.0))
    return PatternReinforcementSignals(
        stability=stability,
        repeatability=repeatability,
        reactivity=reactivity,
        diversity=diversity,
        anti_collapse_penalty=anti_collapse,
        reward=reward,
        sample_count=len(rows),
        stimulus_count=len({str(row.get("stimulus_label", "")) for row in rows}),
        dominant_label=dominant,
        label_entropy=entropy,
        label_counts=label_counts,
    )


def response_repeatability(rows: list[dict[str, Any]]) -> float:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get("stimulus_label", "")), []).append(row)
    scores: list[float] = []
    for group in groups.values():
        if len(group) < 2:
            continue
        label_score = repeated_label_score(group)
        metric_score = repeated_metric_score(group)
        scores.append(0.6 * label_score + 0.4 * metric_score)
    if not scores:
        return 0.0
    return mean_bounded(scores)


def response_reactivity(rows: list[dict[str, Any]]) -> float:
    values = []
    for row in rows:
        peak = float(row.get("peak_response_score", 0.0))
        active_ratio = safe_ratio(
            float(row.get("response_active_pattern_frames", 0.0)),
            float(row.get("response_pattern_frames", 0.0)),
        )
        rms = float(row.get("response_audio_rms", 0.0))
        values.append(
            np.clip(
                0.45 * saturating_score(peak, scale=0.01)
                + 0.35 * active_ratio
                + 0.20 * saturating_score(rms, scale=0.08),
                0.0,
                1.0,
            )
        )
    return mean_bounded(float(value) for value in values)


def repeated_label_score(rows: list[dict[str, Any]]) -> float:
    labels = [str(row.get("response_active_pattern_label") or "none") for row in rows]
    counts = count_labels(labels)
    return max(counts.values()) / len(labels)


def repeated_metric_score(rows: list[dict[str, Any]]) -> float:
    keys = (
        "peak_response_score",
        "response_audio_rms",
        "response_audio_spectral_centroid_hz",
    )
    scores: list[float] = []
    for key in keys:
        values = np.asarray([float(row.get(key, 0.0)) for row in rows], dtype=float)
        mean = float(np.mean(np.abs(values)))
        if mean <= 1e-12:
            scores.append(1.0)
            continue
        coefficient = float(np.std(values) / mean)
        scores.append(float(np.exp(-coefficient)))
    return mean_bounded(scores)


def anti_collapse_penalty(label_counts: dict[str, int], *, total: int) -> float:
    if total <= 0:
        return 0.0
    dominant_fraction = max(label_counts.values()) / total if label_counts else 0.0
    idle_fraction = label_counts.get("idle", 0) / total
    none_fraction = label_counts.get("none", 0) / total
    return float(
        np.clip(
            max(0.0, dominant_fraction - 0.55) / 0.45 + idle_fraction + none_fraction,
            0.0,
            1.0,
        )
    )


def normalized_entropy(label_counts: dict[str, int]) -> float:
    total = sum(label_counts.values())
    if total <= 0 or len(label_counts) <= 1:
        return 0.0
    probabilities = [count / total for count in label_counts.values()]
    entropy = -sum(p * math.log(p) for p in probabilities if p > 0.0)
    return float(entropy / math.log(len(label_counts)))


def count_labels(labels: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return counts


def dominant_label(label_counts: dict[str, int]) -> str | None:
    if not label_counts:
        return None
    return max(label_counts, key=lambda label: label_counts[label])


def saturating_score(value: float, *, scale: float) -> float:
    if scale <= 0.0:
        msg = "scale must be positive"
        raise ValueError(msg)
    return float(1.0 - math.exp(-max(0.0, value) / scale))


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 1e-12:
        return 0.0
    return float(numerator / denominator)


def mean_bounded(values: Any) -> float:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return 0.0
    return float(np.clip(np.mean(array), 0.0, 1.0))
