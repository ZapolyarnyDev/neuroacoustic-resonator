from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from neuroacoustic_resonator.analysis.pattern_detector import classify_pattern
from neuroacoustic_resonator.analysis.protocol_features import pattern_snapshot
from neuroacoustic_resonator.core.field import FieldState
from neuroacoustic_resonator.core.regions import RegionMasks
from neuroacoustic_resonator.protocol import PatternSnapshot, SoundProtocolFrame


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


@dataclass
class OutputPatternHistory:
    signatures: list[OutputPatternSignature] = field(default_factory=list)
    activations: list[float] = field(default_factory=list)

    def update(
        self,
        signature: OutputPatternSignature,
        *,
        activation: float = 0.0,
    ) -> OutputPatternSignature:
        self.signatures.append(signature)
        self.activations.append(max(0.0, float(activation)))
        return signature

    def summary(self, *, active_threshold: float = 1e-6) -> dict[str, Any]:
        if active_threshold < 0.0:
            msg = "active_threshold must be non-negative"
            raise ValueError(msg)
        if not self.signatures:
            return empty_pattern_history_summary()

        labels = [signature.label for signature in self.signatures]
        counts = label_counts(labels)
        confidences = np.asarray(
            [signature.confidence for signature in self.signatures],
            dtype=np.float64,
        )
        activations = np.asarray(self.activations, dtype=np.float64)
        active_indices = [
            index
            for index, activation in enumerate(self.activations)
            if activation > active_threshold or self.signatures[index].label != "idle"
        ]
        active_labels = [labels[index] for index in active_indices]
        active_counts = label_counts(active_labels)
        peak_index = int(np.argmax(activations)) if activations.size else 0
        peak_signature = self.signatures[peak_index]

        return {
            "frames": len(self.signatures),
            "active_frames": len(active_indices),
            "dominant_label": dominant_label(counts),
            "active_dominant_label": dominant_label(active_counts),
            "label_counts": counts,
            "active_label_counts": active_counts,
            "mean_confidence": float(np.mean(confidences)),
            "peak_activation": float(np.max(activations)) if activations.size else 0.0,
            "peak_activation_label": peak_signature.label,
            "peak_activation_confidence": peak_signature.confidence,
            "peak_activation_features": peak_signature.features,
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


def empty_pattern_history_summary() -> dict[str, Any]:
    return {
        "frames": 0,
        "active_frames": 0,
        "dominant_label": None,
        "active_dominant_label": None,
        "label_counts": {},
        "active_label_counts": {},
        "mean_confidence": 0.0,
        "peak_activation": 0.0,
        "peak_activation_label": None,
        "peak_activation_confidence": 0.0,
        "peak_activation_features": {key: 0.0 for key in PATTERN_FEATURE_KEYS},
    }


def label_counts(labels: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return counts


def dominant_label(counts: dict[str, int]) -> str | None:
    if not counts:
        return None
    return max(counts, key=lambda label: counts[label])


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
    snapshot = pattern_snapshot(state, regions.output)
    return pattern_features(snapshot)


def pattern_features(snapshot: PatternSnapshot) -> dict[str, float]:
    synchrony = snapshot.phase_order_1
    phase_order_2 = snapshot.phase_order_2
    phase_order_3 = snapshot.phase_order_3
    phase_spread = float(np.clip(1.0 - synchrony, 0.0, 1.0))
    trace = float(np.clip(snapshot.trace_mean, 0.0, 1.0))
    trace_contrast = float(np.clip(snapshot.trace_contrast, 0.0, 1.0))
    metabolite_stress = snapshot.metabolite_stress
    metabolite_contrast = snapshot.metabolite_contrast
    frequency_spread = float(np.clip(snapshot.frequency_spread, 0.0, 1.0))
    frequency_mean = float(np.clip(snapshot.frequency_mean, 0.0, 2.0) / 2.0)
    trace_phase_lock = snapshot.trace_phase_lock
    metabolite_phase_lock = snapshot.metabolite_phase_lock
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


def protocol_pattern_signature(
    frame: SoundProtocolFrame,
) -> OutputPatternSignature:
    active = frame.active_pattern
    return OutputPatternSignature(
        label="idle" if active is None else active.label,
        confidence=0.0 if active is None else active.confidence,
        features=pattern_features(frame.pattern),
    )


def classify_output_pattern(features: dict[str, float]) -> tuple[str, float]:
    snapshot = PatternSnapshot(
        phase_order_1=features["synchrony"],
        phase_order_2=features["phase_order_2"],
        phase_order_3=features["phase_order_3"],
        trace_mean=features["trace"],
        trace_contrast=features["trace_contrast"],
        metabolite_stress=features["metabolite_stress"],
        metabolite_contrast=features["metabolite_contrast"],
        trace_phase_lock=features["trace_phase_lock"],
        metabolite_phase_lock=features["metabolite_phase_lock"],
        frequency_mean=2.0 * features["frequency_mean"],
        frequency_spread=features["frequency_spread"],
    )
    classification = classify_pattern(snapshot)
    return classification.label, classification.confidence


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
