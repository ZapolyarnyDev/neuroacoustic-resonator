from typing import TYPE_CHECKING

from neuroacoustic_resonator.audio.input import (
    AudioInputFeatures,
    WavInputDrive,
    extract_audio_array_features,
    extract_audio_input_features,
    write_audio_input_features_csv,
)
from neuroacoustic_resonator.audio.io import write_wav
from neuroacoustic_resonator.audio.output import VoiceResponseSonificationRenderer
from neuroacoustic_resonator.audio.turn_detection import (
    TurnDetectionConfig,
    detect_and_write_turns,
    detect_voice_turns,
)

if TYPE_CHECKING:
    from neuroacoustic_resonator.audio.conversation import (
        VoiceConversationConfig,
        render_voice_conversation,
    )
    from neuroacoustic_resonator.audio.live_conversation import (
        LiveConversationConfig,
        LiveConversationEngine,
        run_live_conversation,
    )

__all__ = [
    "AudioInputFeatures",
    "LiveConversationConfig",
    "LiveConversationEngine",
    "TurnDetectionConfig",
    "VoiceConversationConfig",
    "VoiceResponseSonificationRenderer",
    "WavInputDrive",
    "detect_and_write_turns",
    "detect_voice_turns",
    "extract_audio_array_features",
    "extract_audio_input_features",
    "render_voice_conversation",
    "run_live_conversation",
    "write_audio_input_features_csv",
    "write_wav",
]


def __getattr__(name: str) -> object:
    if name in {
        "VoiceConversationConfig",
        "render_voice_conversation",
    }:
        from neuroacoustic_resonator.audio import conversation

        return getattr(conversation, name)
    if name in {
        "LiveConversationConfig",
        "LiveConversationEngine",
        "run_live_conversation",
    }:
        from neuroacoustic_resonator.audio import live_conversation

        return getattr(live_conversation, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
