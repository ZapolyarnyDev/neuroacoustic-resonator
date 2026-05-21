from __future__ import annotations

import json
import numpy as np
from scipy.io import wavfile  # type: ignore[import-untyped]

from neuroacoustic_resonator.audio.turn_detection import (
    TurnDetectionConfig,
    detect_and_write_turns,
    detect_voice_turns,
    main,
)


def test_detect_voice_turns_splits_audio_on_silence(tmp_path) -> None:
    sample_rate = 1_000
    audio = np.concatenate(
        [
            np.zeros(200, dtype=np.float32),
            np.ones(300, dtype=np.float32) * 0.8,
            np.zeros(400, dtype=np.float32),
            np.ones(250, dtype=np.float32) * 0.6,
            np.zeros(200, dtype=np.float32),
        ]
    )
    config = TurnDetectionConfig(
        input_wav=tmp_path / "unused.wav",
        frame_ms=40.0,
        hop_ms=20.0,
        threshold_ratio=0.25,
        min_voice_ms=80.0,
        min_silence_ms=160.0,
        padding_ms=0.0,
    )

    turns = detect_voice_turns(audio, sample_rate=sample_rate, config=config)

    assert len(turns) == 2
    assert turns[0][0] < 260
    assert turns[0][1] > 450
    assert turns[1][0] < 950
    assert turns[1][1] > 1100


def test_detect_and_write_turns_writes_wavs_and_summary(tmp_path) -> None:
    wav_path = tmp_path / "session.wav"
    output_dir = tmp_path / "turns"
    sample_rate = 1_000
    audio = np.concatenate(
        [
            np.zeros(200, dtype=np.float32),
            np.ones(300, dtype=np.float32) * 0.8,
            np.zeros(400, dtype=np.float32),
            np.ones(250, dtype=np.float32) * 0.6,
        ]
    )
    wavfile.write(wav_path, sample_rate, audio)

    rows = detect_and_write_turns(
        TurnDetectionConfig(
            input_wav=wav_path,
            output_dir=output_dir,
            frame_ms=40.0,
            hop_ms=20.0,
            threshold_ratio=0.25,
            min_voice_ms=80.0,
            min_silence_ms=160.0,
            padding_ms=0.0,
        )
    )
    summary = json.loads((output_dir / "turns.json").read_text(encoding="utf-8"))

    assert len(rows) == 2
    assert summary["turn_count"] == 2
    assert (output_dir / "turn_001.wav").exists()
    written_rate, written_samples = wavfile.read(output_dir / "turn_001.wav")
    assert written_rate == sample_rate
    assert written_samples.size > 0


def test_turn_detection_main_writes_outputs(tmp_path) -> None:
    wav_path = tmp_path / "session.wav"
    output_dir = tmp_path / "turns"
    wavfile.write(
        wav_path, 1_000, np.r_[np.zeros(100), np.ones(300)].astype(np.float32)
    )

    exit_code = main(
        [
            "--input",
            str(wav_path),
            "--output-dir",
            str(output_dir),
            "--threshold-ratio",
            "0.2",
            "--min-voice-ms",
            "40",
            "--min-silence-ms",
            "80",
        ]
    )

    assert exit_code == 0
    assert (output_dir / "turns.json").exists()
