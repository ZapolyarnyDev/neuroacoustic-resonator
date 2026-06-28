from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from neuroacoustic_resonator.core.field import FieldState
from neuroacoustic_resonator.core.regions import RegionMasks


@dataclass(frozen=True)
class OutputPatternSignature:
    label: str
    confidence: float
    features: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "features": self.features,
        }


PATTERN_FEATURE_KEYS = (
    "synchrony",
    "phase_spread",
    "phase_order_2",
    "phase_order_3",
    "trace",
    "trace_contrast",
    "metabolite_stress",
    "metabolite_contrast",
    "trace_phase_lock",
    "metabolite_phase_lock",
    "frequency_mean",
    "frequency_spread",
    "brightness",
    "roughness",
)


def output_pattern_signature(
    state: FieldState,
    regions: RegionMasks,
) -> OutputPatternSignature:
    if state.phase.shape != regions.shape:
        msg = "state and regions must have matching shapes"
        raise ValueError(msg)

    features = output_pattern_features(state, regions)
    label, confidence = classify_output_pattern(features)
    return OutputPatternSignature(
        label=label,
        confidence=confidence,
        features=features,
    )


def output_pattern_features(
    state: FieldState,
    regions: RegionMasks,
) -> dict[str, float]:
    mask = regions.output
    if not np.any(mask):
        return {key: 0.0 for key in PATTERN_FEATURE_KEYS}

    phase = state.phase[mask]
    order = np.mean(np.exp(1j * phase))
    second_order = np.mean(np.exp(2j * phase))
    third_order = np.mean(np.exp(3j * phase))
    synchrony = float(np.abs(order))
    phase_order_2 = float(np.abs(second_order))
    phase_order_3 = float(np.abs(third_order))
    phase_spread = float(np.clip(1.0 - synchrony, 0.0, 1.0))
    trace_values = state.trace[mask]
    metabolite_stress_values = 1.0 - state.metabolite[mask]
    trace = float(np.clip(np.mean(trace_values), 0.0, 1.0))
    trace_contrast = float(np.clip(np.std(trace_values), 0.0, 1.0))
    metabolite_stress = float(np.clip(np.mean(metabolite_stress_values), 0.0, 1.0))
    metabolite_contrast = float(np.clip(np.std(metabolite_stress_values), 0.0, 1.0))
    frequency_spread = float(np.clip(np.std(state.frequency[mask]), 0.0, 1.0))
    frequency_mean = float(np.clip(np.mean(state.frequency[mask]), 0.0, 2.0) / 2.0)
    trace_phase_lock = weighted_phase_lock(phase, trace_values)
    metabolite_phase_lock = weighted_phase_lock(phase, metabolite_stress_values)
    brightness = float(
        np.clip(
            0.28 * synchrony
            + 0.24 * metabolite_stress
            + 0.20 * frequency_spread
            + 0.18 * trace_contrast
            + 0.10 * frequency_mean
            + 0.08 * phase_order_2,
            0.0,
            1.0,
        )
    )
    roughness = float(
        np.clip(
            0.45 * phase_spread
            + 0.25 * metabolite_contrast
            + 0.20 * trace_contrast
            + 0.10 * frequency_spread
            + 0.10 * phase_order_3,
            0.0,
            1.0,
        )
    )
    return {
        "synchrony": synchrony,
        "phase_spread": phase_spread,
        "phase_order_2": phase_order_2,
        "phase_order_3": phase_order_3,
        "trace": trace,
        "trace_contrast": trace_contrast,
        "metabolite_stress": metabolite_stress,
        "metabolite_contrast": metabolite_contrast,
        "trace_phase_lock": trace_phase_lock,
        "metabolite_phase_lock": metabolite_phase_lock,
        "frequency_mean": frequency_mean,
        "frequency_spread": frequency_spread,
        "brightness": brightness,
        "roughness": roughness,
    }


def weighted_phase_lock(phase: np.ndarray, weights: np.ndarray) -> float:
    clipped = np.clip(weights, 0.0, None)
    weight_sum = float(np.sum(clipped))
    if weight_sum <= 1e-12:
        return 0.0
    return float(np.abs(np.sum(clipped * np.exp(1j * phase)) / weight_sum))


def classify_output_pattern(features: dict[str, float]) -> tuple[str, float]:
    active = max(features["trace"], features["metabolite_stress"])
    if active < 0.025:
        return "idle", float(np.clip(1.0 - active / 0.025, 0.0, 1.0))

    scores = {
        "coherent": features["synchrony"],
        "split": features["phase_order_2"] * (1.0 - 0.35 * features["synchrony"]),
        "triadic": features["phase_order_3"] * (1.0 - 0.25 * features["synchrony"]),
        "diffuse": 0.6 * features["roughness"] + 0.4 * features["phase_spread"],
        "imprinted": max(
            features["trace_phase_lock"],
            features["metabolite_phase_lock"],
        )
        * (0.5 + 0.5 * active),
    }
    label, best = max(scores.items(), key=lambda item: item[1])
    if best < 0.35:
        return "mixed", float(np.clip(best / 0.35, 0.0, 1.0))
    runner_up = max(value for key, value in scores.items() if key != label)
    confidence = 0.35 + 0.65 * np.clip(best - runner_up, 0.0, 1.0)
    return label, float(confidence)


def compare_output_patterns(
    left: OutputPatternSignature,
    right: OutputPatternSignature,
) -> dict[str, float]:
    left_values = np.asarray([left.features[key] for key in PATTERN_FEATURE_KEYS])
    right_values = np.asarray([right.features[key] for key in PATTERN_FEATURE_KEYS])
    return {
        "pattern_label_match": 1.0 if left.label == right.label else 0.0,
        "pattern_feature_distance": float(np.linalg.norm(right_values - left_values)),
        "pattern_confidence_delta": float(right.confidence - left.confidence),
    }
