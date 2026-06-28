from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from neuroacoustic_resonator.analysis.output_patterns import OutputPatternSignature
from neuroacoustic_resonator.analysis.reinforcement import saturating_score


@dataclass(frozen=True)
class PatternGuidedPlasticityConfig:
    enabled: bool = True
    output_gain: float = 1.0
    assoc_gain: float = 0.35
    min_confidence: float = 0.20
    response_scale: float = 0.01
    stability_weight: float = 0.45
    reactivity_weight: float = 0.45
    anti_collapse_weight: float = 0.35
    diversity_floor: float = 0.08
    max_signal: float = 1.0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if name == "enabled":
                continue
            if value < 0.0:
                msg = f"{name} must be non-negative"
                raise ValueError(msg)
        if self.min_confidence > 1.0:
            msg = "min_confidence must be between 0 and 1"
            raise ValueError(msg)
        if self.response_scale <= 0.0:
            msg = "response_scale must be positive"
            raise ValueError(msg)
        if self.max_signal <= 0.0:
            msg = "max_signal must be positive"
            raise ValueError(msg)


@dataclass(frozen=True)
class PatternPlasticityDecision:
    label: str
    confidence: float
    stability: float
    reactivity: float
    anti_collapse_penalty: float
    reward: float
    output_signal: float
    assoc_signal: float
    output_rate_multiplier: float
    assoc_rate_multiplier: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def pattern_guided_plasticity_decision(
    signature: OutputPatternSignature,
    *,
    response_score: float,
    config: PatternGuidedPlasticityConfig | None = None,
) -> PatternPlasticityDecision:
    if config is None:
        config = PatternGuidedPlasticityConfig()
    confidence = float(np.clip(signature.confidence, 0.0, 1.0))
    response_score = max(0.0, float(response_score))
    if not config.enabled:
        return PatternPlasticityDecision(
            label=signature.label,
            confidence=confidence,
            stability=confidence,
            reactivity=saturating_score(response_score, scale=config.response_scale),
            anti_collapse_penalty=0.0,
            reward=0.0,
            output_signal=response_score,
            assoc_signal=0.0,
            output_rate_multiplier=1.0,
            assoc_rate_multiplier=0.0,
        )

    stability = stability_score(signature, min_confidence=config.min_confidence)
    reactivity = saturating_score(response_score, scale=config.response_scale)
    penalty = frame_anti_collapse_penalty(signature)
    reward = (
        config.stability_weight * stability
        + config.reactivity_weight * reactivity
        + config.diversity_floor
        - config.anti_collapse_weight * penalty
    )
    reward = float(np.clip(reward, 0.0, 1.0))
    output_signal = min(
        config.max_signal,
        response_score * (1.0 + config.output_gain * reward),
    )
    assoc_signal = min(
        config.max_signal,
        response_score * config.assoc_gain * reward,
    )
    return PatternPlasticityDecision(
        label=signature.label,
        confidence=confidence,
        stability=stability,
        reactivity=reactivity,
        anti_collapse_penalty=penalty,
        reward=reward,
        output_signal=output_signal,
        assoc_signal=assoc_signal,
        output_rate_multiplier=1.0 + config.output_gain * reward,
        assoc_rate_multiplier=config.assoc_gain * reward,
    )


def stability_score(
    signature: OutputPatternSignature,
    *,
    min_confidence: float,
) -> float:
    if signature.label in {"idle", "none"}:
        return 0.0
    if signature.confidence <= min_confidence:
        return 0.0
    return float(
        np.clip(
            (signature.confidence - min_confidence) / max(1.0 - min_confidence, 1e-12),
            0.0,
            1.0,
        )
    )


def frame_anti_collapse_penalty(signature: OutputPatternSignature) -> float:
    if signature.label in {"idle", "none"}:
        return 1.0
    if signature.label == "diffuse" and signature.confidence > 0.75:
        return 0.6
    if signature.label == "coherent" and signature.confidence > 0.95:
        return 0.25
    return 0.0


def summarize_plasticity_decisions(
    decisions: list[PatternPlasticityDecision],
) -> dict[str, Any]:
    if not decisions:
        return {
            "frames": 0,
            "mean_reward": 0.0,
            "mean_output_signal": 0.0,
            "mean_assoc_signal": 0.0,
            "peak_reward": 0.0,
            "label_counts": {},
        }
    rewards = np.asarray([decision.reward for decision in decisions], dtype=float)
    output_signals = np.asarray(
        [decision.output_signal for decision in decisions],
        dtype=float,
    )
    assoc_signals = np.asarray(
        [decision.assoc_signal for decision in decisions],
        dtype=float,
    )
    label_counts: dict[str, int] = {}
    for decision in decisions:
        label_counts[decision.label] = label_counts.get(decision.label, 0) + 1
    return {
        "frames": len(decisions),
        "mean_reward": float(np.mean(rewards)),
        "mean_output_signal": float(np.mean(output_signals)),
        "mean_assoc_signal": float(np.mean(assoc_signals)),
        "peak_reward": float(np.max(rewards)),
        "label_counts": label_counts,
    }
