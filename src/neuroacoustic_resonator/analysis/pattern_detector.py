from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neuroacoustic_resonator.protocol import (
    ActivePattern,
    PatternSnapshot,
    PatternTransition,
)


@dataclass(frozen=True, slots=True)
class PatternClassification:
    label: str
    confidence: float
    intensity: float
    scores: dict[str, float]


@dataclass(frozen=True, slots=True)
class PatternDetectorConfig:
    activity_threshold: float = 0.025
    confidence_threshold: float = 0.35
    confirmation_frames: int = 3
    minimum_active_frames: int = 3
    hysteresis_margin: float = 0.08
    novelty_threshold: float = 0.15

    def __post_init__(self) -> None:
        if not 0.0 < self.activity_threshold <= 1.0:
            msg = "activity_threshold must be in (0, 1]"
            raise ValueError(msg)
        for name, value in (
            ("confidence_threshold", self.confidence_threshold),
            ("hysteresis_margin", self.hysteresis_margin),
            ("novelty_threshold", self.novelty_threshold),
        ):
            if not 0.0 <= value <= 1.0:
                msg = f"{name} must be in [0, 1]"
                raise ValueError(msg)
        if self.confirmation_frames < 1:
            msg = "confirmation_frames must be positive"
            raise ValueError(msg)
        if self.minimum_active_frames < 1:
            msg = "minimum_active_frames must be positive"
            raise ValueError(msg)


class TemporalPatternDetector:
    def __init__(
        self,
        config: PatternDetectorConfig | None = None,
    ) -> None:
        self.config = config or PatternDetectorConfig()
        self._active_label: str | None = None
        self._active_age = 0
        self._candidate_label: str | None = None
        self._candidate_age = 0
        self._inactive_age = 0
        self._previous_snapshot: PatternSnapshot | None = None
        self.reset()

    def update(
        self,
        snapshot: PatternSnapshot,
    ) -> tuple[ActivePattern | None, PatternTransition | None]:
        novelty = pattern_novelty(self._previous_snapshot, snapshot)
        self._previous_snapshot = snapshot
        classification = classify_pattern(
            snapshot,
            activity_threshold=self.config.activity_threshold,
        )
        candidate_label = (
            classification.label
            if classification.label != "idle"
            and classification.confidence >= self.config.confidence_threshold
            else None
        )

        if self._active_label is None:
            return self._update_inactive(
                candidate_label,
                classification,
                novelty,
            )
        return self._update_active(
            candidate_label,
            classification,
            novelty,
        )

    def reset(self) -> None:
        self._active_label = None
        self._active_age = 0
        self._candidate_label = None
        self._candidate_age = 0
        self._inactive_age = 0
        self._previous_snapshot = None

    def _update_inactive(
        self,
        candidate_label: str | None,
        classification: PatternClassification,
        novelty: float,
    ) -> tuple[ActivePattern | None, PatternTransition | None]:
        if candidate_label is None:
            self._clear_candidate()
            return None, None

        self._advance_candidate(candidate_label)
        if self._candidate_age < self.config.confirmation_frames:
            return None, None

        self._active_label = candidate_label
        self._active_age = 1
        self._inactive_age = 0
        self._clear_candidate()
        active = self._active_pattern(classification, novelty)
        return active, PatternTransition("started", None, candidate_label)

    def _update_active(
        self,
        candidate_label: str | None,
        classification: PatternClassification,
        novelty: float,
    ) -> tuple[ActivePattern | None, PatternTransition | None]:
        self._active_age += 1
        if candidate_label is None:
            self._inactive_age += 1
            self._clear_candidate()
            if (
                self._inactive_age >= self.config.confirmation_frames
                and self._active_age >= self.config.minimum_active_frames
            ):
                ended_label = self._active_label
                self._clear_active()
                return None, PatternTransition("ended", ended_label, None)
            return self._active_pattern(classification, novelty), None

        self._inactive_age = 0
        if candidate_label == self._active_label:
            self._clear_candidate()
            return self._active_pattern(classification, novelty), None

        active_label = self._active_label
        if active_label is None:
            msg = "active pattern is not set"
            raise RuntimeError(msg)
        active_score = classification.scores.get(active_label, 0.0)
        candidate_score = classification.scores.get(
            candidate_label,
            classification.confidence,
        )
        if candidate_score < active_score + self.config.hysteresis_margin:
            self._clear_candidate()
            return self._active_pattern(classification, novelty), None

        self._advance_candidate(candidate_label)
        if (
            self._candidate_age < self.config.confirmation_frames
            or self._active_age < self.config.minimum_active_frames
        ):
            return self._active_pattern(classification, novelty), None

        previous_label = self._active_label
        self._active_label = candidate_label
        self._active_age = 1
        self._clear_candidate()
        active = self._active_pattern(classification, novelty)
        return active, PatternTransition("changed", previous_label, candidate_label)

    def _active_pattern(
        self,
        classification: PatternClassification,
        novelty: float,
    ) -> ActivePattern:
        if self._active_label is None:
            msg = "active pattern is not set"
            raise RuntimeError(msg)
        confidence = (
            classification.confidence
            if classification.label == self._active_label
            else classification.scores.get(self._active_label, 0.0)
        )
        return ActivePattern(
            label=self._active_label,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            intensity=classification.intensity,
            novelty=novelty,
            is_novel=novelty >= self.config.novelty_threshold,
            age_frames=self._active_age,
        )

    def _advance_candidate(self, label: str) -> None:
        if self._candidate_label == label:
            self._candidate_age += 1
        else:
            self._candidate_label = label
            self._candidate_age = 1

    def _clear_candidate(self) -> None:
        self._candidate_label = None
        self._candidate_age = 0

    def _clear_active(self) -> None:
        self._active_label = None
        self._active_age = 0
        self._inactive_age = 0
        self._clear_candidate()


def classify_pattern(
    snapshot: PatternSnapshot,
    *,
    activity_threshold: float = 0.025,
) -> PatternClassification:
    if not 0.0 < activity_threshold <= 1.0:
        msg = "activity_threshold must be in (0, 1]"
        raise ValueError(msg)

    trace = float(np.clip(snapshot.trace_mean, 0.0, 1.0))
    trace_contrast = float(np.clip(snapshot.trace_contrast, 0.0, 1.0))
    frequency_spread = float(np.clip(snapshot.frequency_spread, 0.0, 1.0))
    intensity = max(trace, snapshot.metabolite_stress)
    if intensity < activity_threshold:
        confidence = float(np.clip(1.0 - intensity / activity_threshold, 0.0, 1.0))
        return PatternClassification(
            label="idle",
            confidence=confidence,
            intensity=intensity,
            scores={"idle": confidence},
        )

    phase_spread = float(np.clip(1.0 - snapshot.phase_order_1, 0.0, 1.0))
    roughness = float(
        np.clip(
            0.45 * phase_spread
            + 0.25 * snapshot.metabolite_contrast
            + 0.20 * trace_contrast
            + 0.10 * frequency_spread
            + 0.10 * snapshot.phase_order_3,
            0.0,
            1.0,
        )
    )
    scores = {
        "coherent": snapshot.phase_order_1,
        "split": snapshot.phase_order_2 * (1.0 - 0.35 * snapshot.phase_order_1),
        "triadic": snapshot.phase_order_3 * (1.0 - 0.25 * snapshot.phase_order_1),
        "diffuse": 0.6 * roughness + 0.4 * phase_spread,
        "imprinted": max(
            snapshot.trace_phase_lock,
            snapshot.metabolite_phase_lock,
        )
        * (0.5 + 0.5 * intensity),
    }
    label, best = max(scores.items(), key=lambda item: item[1])
    if best < 0.35:
        confidence = float(np.clip(best / 0.35, 0.0, 1.0))
        return PatternClassification(
            label="mixed",
            confidence=confidence,
            intensity=intensity,
            scores=scores,
        )
    runner_up = max(value for key, value in scores.items() if key != label)
    confidence = float(0.35 + 0.65 * np.clip(best - runner_up, 0.0, 1.0))
    return PatternClassification(
        label=label,
        confidence=confidence,
        intensity=intensity,
        scores=scores,
    )


def pattern_novelty(
    previous: PatternSnapshot | None,
    current: PatternSnapshot,
) -> float:
    if previous is None:
        return 0.0
    left = normalized_pattern_vector(previous)
    right = normalized_pattern_vector(current)
    return float(np.clip(np.sqrt(np.mean(np.square(right - left))), 0.0, 1.0))


def normalized_pattern_vector(snapshot: PatternSnapshot) -> np.ndarray:
    return np.asarray(
        [
            snapshot.phase_order_1,
            snapshot.phase_order_2,
            snapshot.phase_order_3,
            saturate_non_negative(snapshot.trace_mean),
            saturate_non_negative(snapshot.trace_contrast),
            snapshot.metabolite_stress,
            snapshot.metabolite_contrast,
            snapshot.trace_phase_lock,
            snapshot.metabolite_phase_lock,
            saturate_non_negative(snapshot.frequency_mean),
            saturate_non_negative(snapshot.frequency_spread),
        ],
        dtype=np.float64,
    )


def saturate_non_negative(value: float) -> float:
    return value / (1.0 + value)
