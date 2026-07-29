from __future__ import annotations

import json
import wave
from pathlib import Path

import pytest

from neuroacoustic_resonator.cli.protocol import (
    main,
    record_protocol,
    replay_protocol,
)
from neuroacoustic_resonator.configuration import SimulationConfig
from neuroacoustic_resonator.protocol import ProtocolConsumer, SoundProtocolFrame


def _config(tmp_path) -> Path:
    path = tmp_path / "protocol.yaml"
    path.write_text(
        """
field:
  size: 6
  seed: 7
synthetic_input:
  enabled: false
protocol:
  confirmation_frames: 1
steps: 8
""",
        encoding="utf-8",
    )
    return path


class SequenceConsumer:
    def __init__(self) -> None:
        self.sequences: list[int] = []

    def consume(self, frame: SoundProtocolFrame) -> None:
        self.sequences.append(frame.sequence)


def test_simulation_record_replay_preserves_identical_frames(tmp_path) -> None:
    config_path = _config(tmp_path)
    output = tmp_path / "nested" / "recording.jsonl"

    recorded = record_protocol(config_path, output, steps=12)
    replayed = replay_protocol(output)

    assert recorded.frames == replayed.frames
    assert len(recorded.frames) == 13
    assert recorded.frames[0].sequence == 0
    assert recorded.frames[-1].step == 12
    assert recorded.summary()["version"] == "0.1"


def test_replay_uses_pipeline_without_loading_simulation_config(
    tmp_path,
    monkeypatch,
) -> None:
    output = tmp_path / "recording.jsonl"
    recorded = record_protocol(_config(tmp_path), output, steps=3)
    consumer = SequenceConsumer()
    assert isinstance(consumer, ProtocolConsumer)

    def fail(*args, **kwargs):
        msg = "replay attempted to load a simulation"
        raise AssertionError(msg)

    monkeypatch.setattr(SimulationConfig, "from_file", fail)

    replayed = replay_protocol(output, [consumer])

    assert replayed.frames == recorded.frames
    assert consumer.sequences == [0, 1, 2, 3]


def test_protocol_cli_records_and_replays_wav(tmp_path) -> None:
    recording = tmp_path / "recording.jsonl"
    record_summary = tmp_path / "record-summary.json"
    replay_summary = tmp_path / "replay-summary.json"
    replay_wav = tmp_path / "replay.wav"

    record_exit = main(
        [
            "record",
            "--config",
            str(_config(tmp_path)),
            "--steps",
            "4",
            "--output",
            str(recording),
            "--summary",
            str(record_summary),
        ]
    )
    replay_exit = main(
        [
            "replay",
            "--input",
            str(recording),
            "--summary",
            str(replay_summary),
            "--output-wav",
            str(replay_wav),
            "--sample-rate",
            "8000",
            "--frame-size",
            "64",
        ]
    )

    recorded = json.loads(record_summary.read_text(encoding="utf-8"))
    replayed = json.loads(replay_summary.read_text(encoding="utf-8"))
    with wave.open(str(replay_wav), "rb") as stream:
        wav_frames = stream.getnframes()

    assert record_exit == replay_exit == 0
    assert recorded == replayed
    assert recorded["frames"] == 5
    assert wav_frames == 5 * 64


def test_record_protocol_rejects_negative_steps(tmp_path) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        record_protocol(_config(tmp_path), tmp_path / "recording.jsonl", steps=-1)
