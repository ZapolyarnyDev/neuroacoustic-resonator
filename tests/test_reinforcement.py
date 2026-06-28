from __future__ import annotations

import pytest

from neuroacoustic_resonator.analysis.reinforcement import (
    PatternReinforcementWeights,
    compute_pattern_reinforcement_signals,
)


def test_compute_pattern_reinforcement_signals_rewards_repeatable_reactive_rows() -> (
    None
):
    rows = [
        {
            "stimulus_label": "tone",
            "response_active_pattern_label": "coherent",
            "response_pattern_confidence": 0.8,
            "response_active_pattern_frames": 6,
            "response_pattern_frames": 8,
            "peak_response_score": 0.02,
            "response_audio_rms": 0.08,
            "response_audio_spectral_centroid_hz": 400.0,
        },
        {
            "stimulus_label": "tone",
            "response_active_pattern_label": "coherent",
            "response_pattern_confidence": 0.75,
            "response_active_pattern_frames": 5,
            "response_pattern_frames": 8,
            "peak_response_score": 0.018,
            "response_audio_rms": 0.075,
            "response_audio_spectral_centroid_hz": 420.0,
        },
        {
            "stimulus_label": "pulse",
            "response_active_pattern_label": "split",
            "response_pattern_confidence": 0.7,
            "response_active_pattern_frames": 4,
            "response_pattern_frames": 8,
            "peak_response_score": 0.015,
            "response_audio_rms": 0.07,
            "response_audio_spectral_centroid_hz": 650.0,
        },
    ]

    signals = compute_pattern_reinforcement_signals(rows)

    assert signals.sample_count == 3
    assert signals.stimulus_count == 2
    assert signals.stability > 0.7
    assert signals.reactivity > 0.5
    assert signals.diversity > 0.0
    assert signals.reward > 0.0
    assert signals.label_counts == {"coherent": 2, "split": 1}


def test_reinforcement_penalizes_collapsed_idle_rows() -> None:
    rows = [
        {
            "stimulus_label": f"stimulus-{index}",
            "response_active_pattern_label": "idle",
            "response_pattern_confidence": 0.9,
            "response_active_pattern_frames": 0,
            "response_pattern_frames": 8,
            "peak_response_score": 0.0,
            "response_audio_rms": 0.0,
            "response_audio_spectral_centroid_hz": 0.0,
        }
        for index in range(4)
    ]

    signals = compute_pattern_reinforcement_signals(rows)

    assert signals.diversity == 0.0
    assert signals.anti_collapse_penalty == 1.0
    assert signals.reward < signals.stability


def test_reinforcement_weights_reject_negative_values() -> None:
    with pytest.raises(ValueError, match="stability"):
        PatternReinforcementWeights(stability=-0.1)
