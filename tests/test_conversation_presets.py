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
    assert effective.pattern_voice_depth == preset.pattern_voice_depth
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
    assert effective.pattern_voice_depth == preset.pattern_voice_depth


def test_conversation_preset_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="unknown conversation preset"):
        conversation_preset("missing")
