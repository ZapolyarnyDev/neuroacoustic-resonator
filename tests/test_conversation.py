from __future__ import annotations

import json
import wave

import numpy as np
from scipy.io import wavfile  # type: ignore[import-untyped]

from neuroacoustic_resonator.audio.conversation import (
    VoiceConversationConfig,
    main,
    render_voice_conversation,
)


def test_render_voice_conversation_writes_response_wav_and_summary(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
field:
  size: 6
  seed: 1
  coupling_strength: 0.2
synthetic_input:
  enabled: false
steps: 8
""",
        encoding="utf-8",
    )
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    wavfile.write(first, 8_000, np.ones(256, dtype=np.float32))
    wavfile.write(second, 8_000, np.linspace(0.0, 1.0, 256, dtype=np.float32))
    output = tmp_path / "conversation.wav"
    summary_path = tmp_path / "conversation.json"

    summary = render_voice_conversation(
        VoiceConversationConfig(
            config_path=config_path,
            input_wavs=(first, second),
            output_wav=output,
            output_summary=summary_path,
            sample_rate=8_000,
            output_frame_size=80,
            input_frame_size=128,
            input_hop_size=64,
            response_seconds=0.1,
            pause_seconds=0.05,
            warmup_steps=2,
        )
    )

    loaded = json.loads(summary_path.read_text(encoding="utf-8"))

    with wave.open(str(output), "rb") as stream:
        assert stream.getframerate() == 8_000
        assert stream.getnframes() == 2 * 800 + 2 * 400
    assert summary["utterance_count"] == 2
    assert len(summary["utterances"]) == 2
    assert summary["utterances"][0]["peak_input_value"] > 0.0
    assert loaded["output_wav"] == str(output)


def test_render_voice_conversation_produces_audible_response(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
field:
  size: 6
  seed: 1
  coupling_strength: 0.2
synthetic_input:
  enabled: false
steps: 8
""",
        encoding="utf-8",
    )
    wav_path = tmp_path / "voice.wav"
    samples = np.zeros(512, dtype=np.float32)
    samples[128:256] = 1.0
    wavfile.write(wav_path, 8_000, samples)
    output = tmp_path / "conversation.wav"
    summary_path = tmp_path / "conversation.json"

    render_voice_conversation(
        VoiceConversationConfig(
            config_path=config_path,
            input_wavs=(wav_path,),
            output_wav=output,
            output_summary=summary_path,
            sample_rate=8_000,
            output_frame_size=80,
            input_frame_size=128,
            input_hop_size=64,
            response_seconds=0.1,
            pause_seconds=0.0,
            warmup_steps=2,
            response_threshold=0.0,
            response_sensitivity=900.0,
        )
    )

    with wave.open(str(output), "rb") as stream:
        samples = np.frombuffer(stream.readframes(stream.getnframes()), dtype=np.int16)

    assert int(np.max(np.abs(samples))) > 0


def test_conversation_main_writes_outputs(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
field:
  size: 6
  seed: 1
synthetic_input:
  enabled: false
steps: 4
""",
        encoding="utf-8",
    )
    wav_path = tmp_path / "voice.wav"
    wavfile.write(wav_path, 8_000, np.ones(256, dtype=np.float32))
    output = tmp_path / "conversation.wav"
    summary = tmp_path / "conversation.json"

    exit_code = main(
        [
            "--config",
            str(config_path),
            "--inputs",
            str(wav_path),
            "--output",
            str(output),
            "--summary",
            str(summary),
            "--sample-rate",
            "8000",
            "--output-frame-size",
            "80",
            "--input-frame-size",
            "128",
            "--input-hop-size",
            "64",
            "--response-seconds",
            "0.1",
            "--pause-seconds",
            "0.0",
        ]
    )

    assert exit_code == 0
    assert output.exists()
    assert summary.exists()
