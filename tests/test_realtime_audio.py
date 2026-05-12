from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from neuroacoustic_resonator.realtime_audio import (
    RealtimeAudioConfig,
    RealtimeAudioEngine,
    play_realtime_audio,
)


class FakeStream:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.entered = False

    def __enter__(self) -> FakeStream:
        self.entered = True
        callback = self.kwargs["callback"]
        frames = self.kwargs["blocksize"]
        outdata = np.zeros((frames, self.kwargs["channels"]), dtype=np.float32)
        callback(outdata, frames, None, None)
        self.outdata = outdata
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.entered = False


def test_realtime_audio_config_rejects_invalid_frame_size() -> None:
    with pytest.raises(ValueError, match="frame_size"):
        RealtimeAudioConfig(frame_size=0)


def test_realtime_audio_config_rejects_invalid_audio_mode() -> None:
    with pytest.raises(ValueError, match="audio_mode"):
        RealtimeAudioConfig(audio_mode="unknown")


def test_realtime_audio_engine_callback_fills_output_buffer() -> None:
    engine = RealtimeAudioEngine(
        RealtimeAudioConfig(
            config_path=Path("configs") / "audio_demo.yaml",
            sample_rate=8_000,
            frame_size=32,
        )
    )
    outdata = np.zeros((32, 1), dtype=np.float32)

    engine.callback(outdata, 32, None, None)

    assert engine.callback_count == 1
    assert outdata.shape == (32, 1)
    assert np.all(np.isfinite(outdata))
    assert np.max(np.abs(outdata)) > 0.0


def test_realtime_audio_engine_supports_gated_mode() -> None:
    engine = RealtimeAudioEngine(
        RealtimeAudioConfig(
            config_path=Path("configs") / "audio_demo.yaml",
            sample_rate=8_000,
            frame_size=32,
            audio_mode="gated",
        )
    )
    outdata = np.zeros((32, 1), dtype=np.float32)

    engine.callback(outdata, 32, None, None)

    assert engine.callback_count == 1
    assert np.all(np.isfinite(outdata))


def test_realtime_audio_engine_supports_event_mode() -> None:
    engine = RealtimeAudioEngine(
        RealtimeAudioConfig(
            config_path=Path("configs") / "audio_demo.yaml",
            sample_rate=8_000,
            frame_size=32,
            audio_mode="event",
        )
    )
    outdata = np.zeros((32, 1), dtype=np.float32)

    engine.callback(outdata, 32, None, None)

    assert engine.callback_count == 1
    assert np.all(np.isfinite(outdata))


def test_play_realtime_audio_uses_stream_factory() -> None:
    streams: list[FakeStream] = []

    def stream_factory(**kwargs: Any) -> FakeStream:
        stream = FakeStream(**kwargs)
        streams.append(stream)
        return stream

    engine = play_realtime_audio(
        RealtimeAudioConfig(
            config_path=Path("configs") / "audio_demo.yaml",
            sample_rate=8_000,
            frame_size=32,
            duration_seconds=0.01,
        ),
        stream_factory=stream_factory,
    )

    assert engine.callback_count == 1
    assert len(streams) == 1
    assert streams[0].kwargs["samplerate"] == 8_000
    assert streams[0].kwargs["blocksize"] == 32
