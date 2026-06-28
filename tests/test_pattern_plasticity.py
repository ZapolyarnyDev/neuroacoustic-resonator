from __future__ import annotations

import pytest

from neuroacoustic_resonator.analysis.output_patterns import OutputPatternSignature
from neuroacoustic_resonator.analysis.pattern_plasticity import (
    PatternGuidedPlasticityConfig,
    pattern_guided_plasticity_decision,
    summarize_plasticity_decisions,
)


def signature(label: str, confidence: float) -> OutputPatternSignature:
    return OutputPatternSignature(label=label, confidence=confidence, features={})


def test_pattern_guided_plasticity_rewards_confident_reactive_pattern() -> None:
    decision = pattern_guided_plasticity_decision(
        signature("split", 0.8),
        response_score=0.02,
        config=PatternGuidedPlasticityConfig(output_gain=1.0, assoc_gain=0.5),
    )

    assert decision.reward > 0.0
    assert decision.output_signal > 0.02
    assert decision.assoc_signal > 0.0
    assert decision.output_rate_multiplier > 1.0


def test_pattern_guided_plasticity_penalizes_idle_collapse() -> None:
    decision = pattern_guided_plasticity_decision(
        signature("idle", 0.95),
        response_score=0.02,
        config=PatternGuidedPlasticityConfig(),
    )

    assert decision.anti_collapse_penalty == 1.0
    assert decision.reward < 0.5


def test_disabled_pattern_guided_plasticity_preserves_output_signal() -> None:
    decision = pattern_guided_plasticity_decision(
        signature("split", 0.8),
        response_score=0.02,
        config=PatternGuidedPlasticityConfig(enabled=False),
    )

    assert decision.reward == 0.0
    assert decision.output_signal == 0.02
    assert decision.assoc_signal == 0.0


def test_summarize_plasticity_decisions_reports_means() -> None:
    decisions = [
        pattern_guided_plasticity_decision(
            signature("split", 0.8), response_score=0.02
        ),
        pattern_guided_plasticity_decision(
            signature("triadic", 0.7), response_score=0.01
        ),
    ]

    summary = summarize_plasticity_decisions(decisions)

    assert summary["frames"] == 2
    assert summary["mean_reward"] > 0.0
    assert summary["label_counts"] == {"split": 1, "triadic": 1}


def test_pattern_guided_plasticity_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="output_gain"):
        PatternGuidedPlasticityConfig(output_gain=-0.1)
