from __future__ import annotations

import numpy as np

from neuroacoustic_resonator.audio.live_conversation import (
    LiveConversationConfig,
    LiveConversationEngine,
    record_utterance,
)


class FakeSoundDevice:
    def __init__(self, blocks: list[np.ndarray]) -> None:
        self.blocks = blocks
        self.played: list[np.ndarray] = []

    def InputStream(
        self,
        *,
        samplerate: int,
        blocksize: int,
        channels: int,
        dtype: str,
        device: int | str | None = None,
    ) -> "FakeInputStream":
        del samplerate, dtype, device
        return FakeInputStream(self.blocks, blocksize=blocksize, channels=channels)

    def play(
        self,
        data: np.ndarray,
        *,
        samplerate: int,
        blocking: bool,
        device: int | str | None = None,
    ) -> None:
        del samplerate, blocking, device
        self.played.append(np.asarray(data))


class FakeInputStream:
    def __init__(
        self,
        blocks: list[np.ndarray],
        *,
        blocksize: int,
        channels: int,
    ) -> None:
        self.blocks = blocks
        self.blocksize = blocksize
        self.channels = channels
        self.index = 0

    def __enter__(self) -> "FakeInputStream":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None

    def read(self, frames: int) -> tuple[np.ndarray, bool]:
        assert frames == self.blocksize
        if self.index >= len(self.blocks):
            return np.zeros((frames, self.channels), dtype=np.float32), False
        block = self.blocks[self.index]
        self.index += 1
        return np.asarray(block, dtype=np.float32).reshape(frames, self.channels), False


def test_record_utterance_waits_for_voice_and_trims_tail_silence() -> None:
    config = LiveConversationConfig(
        sample_rate=8_000,
        input_block_size=80,
        start_rms=0.1,
        stop_rms=0.05,
        min_utterance_seconds=0.02,
        silence_seconds=0.02,
        max_utterance_seconds=0.2,
    )
    blocks = [
        np.zeros(80),
        np.full(80, 0.2),
        np.full(80, 0.15),
        np.zeros(80),
        np.zeros(80),
    ]

    utterance = record_utterance(config, FakeSoundDevice(blocks))

    assert utterance is not None
    assert utterance.shape == (160,)
    assert float(np.max(utterance)) == np.float32(0.2)


def test_live_conversation_engine_processes_utterance(tmp_path) -> None:
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
    config = LiveConversationConfig(
        config_path=config_path,
        sample_rate=8_000,
        input_block_size=80,
        output_frame_size=80,
        input_frame_size=128,
        input_hop_size=64,
        response_seconds=0.1,
        min_response_seconds=0.1,
        max_response_seconds=0.2,
        warmup_steps=2,
    )
    audio = np.zeros(512, dtype=np.float64)
    audio[128:256] = 0.8
    engine = LiveConversationEngine(config)

    result = engine.process_utterance(audio, index=1)

    assert result.index == 1
    assert result.response_audio.size > 0
    assert result.summary["peak_input_value"] > 0.0
    assert result.summary["initial_response_seed"] >= 0.0
    assert "label" in result.summary["input_end_output_pattern"]
    assert "features" in result.summary["response_end_output_pattern"]
    assert engine.session_summary()["turn_count"] == 1


def test_live_conversation_engine_records_turn_audio(tmp_path) -> None:
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
    record_dir = tmp_path / "records"
    config = LiveConversationConfig(
        config_path=config_path,
        sample_rate=8_000,
        output_frame_size=80,
        input_frame_size=128,
        input_hop_size=64,
        response_seconds=0.1,
        min_response_seconds=0.1,
        max_response_seconds=0.1,
        warmup_steps=1,
        record_dir=record_dir,
    )
    audio = np.zeros(256, dtype=np.float64)
    audio[64:128] = 0.7
    engine = LiveConversationEngine(config)

    result = engine.process_utterance(audio, index=1)

    assert (record_dir / "turn_001_input.wav").exists()
    assert (record_dir / "turn_001_response.wav").exists()
    assert result.summary["input_wav"] == str(record_dir / "turn_001_input.wav")
    assert result.summary["response_wav"] == str(record_dir / "turn_001_response.wav")
