from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ResponsiveConversationPreset:
    name: str
    gain: float
    response_seconds: float
    min_response_seconds: float
    max_response_seconds: float
    response_seed_gain: float
    response_seed_decay_seconds: float
    output_plasticity_rate: float
    output_frequency_plasticity_rate: float
    response_threshold: float
    response_sensitivity: float
    pattern_voice_depth: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


CONVERSATION_PRESETS: dict[str, ResponsiveConversationPreset] = {
    "reactive": ResponsiveConversationPreset(
        name="reactive",
        gain=0.38,
        response_seconds=1.0,
        min_response_seconds=0.35,
        max_response_seconds=2.2,
        response_seed_gain=0.85,
        response_seed_decay_seconds=0.45,
        output_plasticity_rate=0.018,
        output_frequency_plasticity_rate=0.004,
        response_threshold=0.0,
        response_sensitivity=1100.0,
        pattern_voice_depth=0.65,
    ),
    "textured": ResponsiveConversationPreset(
        name="textured",
        gain=0.36,
        response_seconds=1.8,
        min_response_seconds=0.8,
        max_response_seconds=3.4,
        response_seed_gain=0.70,
        response_seed_decay_seconds=0.9,
        output_plasticity_rate=0.026,
        output_frequency_plasticity_rate=0.006,
        response_threshold=0.0,
        response_sensitivity=850.0,
        pattern_voice_depth=1.0,
    ),
    "memory": ResponsiveConversationPreset(
        name="memory",
        gain=0.34,
        response_seconds=2.0,
        min_response_seconds=1.0,
        max_response_seconds=4.0,
        response_seed_gain=0.95,
        response_seed_decay_seconds=1.3,
        output_plasticity_rate=0.035,
        output_frequency_plasticity_rate=0.008,
        response_threshold=0.0,
        response_sensitivity=780.0,
        pattern_voice_depth=0.85,
    ),
    "quiet": ResponsiveConversationPreset(
        name="quiet",
        gain=0.22,
        response_seconds=1.2,
        min_response_seconds=0.5,
        max_response_seconds=2.4,
        response_seed_gain=0.55,
        response_seed_decay_seconds=0.8,
        output_plasticity_rate=0.012,
        output_frequency_plasticity_rate=0.002,
        response_threshold=0.00005,
        response_sensitivity=650.0,
        pattern_voice_depth=0.45,
    ),
    "bright": ResponsiveConversationPreset(
        name="bright",
        gain=0.40,
        response_seconds=1.4,
        min_response_seconds=0.6,
        max_response_seconds=2.8,
        response_seed_gain=0.75,
        response_seed_decay_seconds=0.6,
        output_plasticity_rate=0.020,
        output_frequency_plasticity_rate=0.010,
        response_threshold=0.0,
        response_sensitivity=980.0,
        pattern_voice_depth=0.75,
    ),
}


def conversation_preset(name: str) -> ResponsiveConversationPreset:
    try:
        return CONVERSATION_PRESETS[name]
    except KeyError as exc:
        valid = ", ".join(sorted(CONVERSATION_PRESETS))
        msg = f"unknown conversation preset {name!r}; expected one of: {valid}"
        raise ValueError(msg) from exc


def preset_names() -> tuple[str, ...]:
    return tuple(sorted(CONVERSATION_PRESETS))
