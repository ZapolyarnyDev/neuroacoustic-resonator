from __future__ import annotations

import pytest

from neuroacoustic_resonator.audio.conversation import (
    VoiceConversationConfig,
    apply_voice_conversation_preset,
)
from neuroacoustic_resonator.audio.conversation_presets import (
    conversation_preset,
    preset_names,
)
from neuroacoustic_resonator.audio.live_conversation import (
    LiveConversationConfig,
    apply_live_conversation_preset,
)


def test_conversation_preset_names_include_user_modes() -> None:
    assert preset_names() == ("bright", "memory", "quiet", "reactive", "textured")


def test_apply_voice_conversation_preset_updates_response_profile(tmp_path) -> None:
    config = VoiceConversationConfig(
        config_path=tmp_path / "config.yaml",
        input_wavs=(tmp_path / "voice.wav",),
        preset_name="memory",
    )

    effective = apply_voice_conversation_preset(config)
    preset = conversation_preset("memory")

    assert effective.gain == preset.gain
    assert effective.response_seconds == preset.response_seconds
    assert effective.carrier_frequency == preset.carrier_frequency
    assert effective.frequency_scale == preset.frequency_scale
    assert effective.pattern_voice_depth == preset.pattern_voice_depth
    assert effective.response_mix == preset.response_mix
    assert effective.min_response_gain == preset.min_response_gain
    assert effective.target_response_rms == preset.target_response_rms
    assert effective.output_plasticity_rate == preset.output_plasticity_rate


def test_apply_live_conversation_preset_updates_response_profile(tmp_path) -> None:
    config = LiveConversationConfig(
        config_path=tmp_path / "config.yaml",
        preset_name="bright",
    )

    effective = apply_live_conversation_preset(config)
    preset = conversation_preset("bright")

    assert effective.gain == preset.gain
    assert effective.response_sensitivity == preset.response_sensitivity
    assert effective.carrier_frequency == preset.carrier_frequency
    assert effective.pattern_voice_depth == preset.pattern_voice_depth
    assert effective.response_mix == preset.response_mix
    assert effective.min_energy_gain == preset.min_energy_gain


def test_conversation_preset_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="unknown conversation preset"):
        conversation_preset("missing")
