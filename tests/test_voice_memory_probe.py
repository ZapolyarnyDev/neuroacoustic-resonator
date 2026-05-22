from __future__ import annotations

import csv
import json

import numpy as np
from scipy.io import wavfile  # type: ignore[import-untyped]

from neuroacoustic_resonator.analysis.voice_memory_probe import (
    VoiceMemoryProbeConfig,
    main,
    run_voice_memory_probe,
)


def test_run_voice_memory_probe_writes_repeated_response_comparison(tmp_path) -> None:
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
    csv_path = tmp_path / "voice-memory.csv"
    summary_path = tmp_path / "voice-memory.json"

    summary = run_voice_memory_probe(
        VoiceMemoryProbeConfig(
            config_path=config_path,
            input_wav=wav_path,
            output_csv=csv_path,
            output_summary=summary_path,
            frame_size=128,
            hop_size=64,
            input_assoc_gain=0.5,
            pause_steps=3,
            warmup_steps=2,
            max_steps=5,
        )
    )

    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    loaded = json.loads(summary_path.read_text(encoding="utf-8"))

    assert len(rows) == 10
    assert {row["repeat_index"] for row in rows} == {"1", "2"}
    assert summary["first"]["rows"] == 5
    assert summary["second"]["rows"] == 5
    assert summary["parameters"]["input_assoc_gain"] == 0.5
    assert "output_fast_response_score_corr" in summary["comparison"]
    assert "output_event_score_second_to_first_peak" in loaded["comparison"]


def test_voice_memory_probe_compares_memory_drive_strength(tmp_path) -> None:
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
    csv_path = tmp_path / "voice-memory.csv"
    summary_path = tmp_path / "voice-memory.json"

    summary = run_voice_memory_probe(
        VoiceMemoryProbeConfig(
            config_path=config_path,
            input_wav=wav_path,
            output_csv=csv_path,
            output_summary=summary_path,
            frame_size=128,
            hop_size=64,
            input_assoc_gain=0.5,
            pause_steps=3,
            warmup_steps=2,
            max_steps=5,
            compare_memory_drive_strength=0.25,
        )
    )

    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    loaded = json.loads(summary_path.read_text(encoding="utf-8"))

    assert len(rows) == 20
    assert {row["probe_label"] for row in rows} == {"baseline", "memory_drive"}
    assert "baseline" in summary
    assert "memory_drive" in summary
    assert "memory_drive_comparison" in summary
    assert loaded["memory_drive"]["parameters"]["compare_memory_drive_strength"] == 0.25
    assert (
        "output_fast_response_score_mean_abs_delta_memory_drive_to_baseline_ratio"
        in summary["memory_drive_comparison"]
    )


def test_voice_memory_probe_compares_silence_control(tmp_path) -> None:
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
    csv_path = tmp_path / "voice-memory.csv"
    summary_path = tmp_path / "voice-memory.json"

    summary = run_voice_memory_probe(
        VoiceMemoryProbeConfig(
            config_path=config_path,
            input_wav=wav_path,
            output_csv=csv_path,
            output_summary=summary_path,
            frame_size=128,
            hop_size=64,
            input_assoc_gain=0.5,
            pause_steps=3,
            warmup_steps=2,
            max_steps=5,
            compare_silence_control=True,
        )
    )

    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    loaded = json.loads(summary_path.read_text(encoding="utf-8"))

    assert len(rows) == 20
    assert {row["probe_label"] for row in rows} == {"baseline", "silence_control"}
    assert "silence_control" in summary
    assert "voice_vs_silence_control" in summary
    assert loaded["silence_control"]["parameters"]["compare_silence_control"] is True
    assert (
        "output_fast_response_score_mean_abs_delta_silence_control_to_baseline_ratio"
        in summary["voice_vs_silence_control"]
    )


def test_voice_memory_probe_main_writes_outputs(tmp_path) -> None:
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
    csv_path = tmp_path / "voice-memory.csv"
    summary_path = tmp_path / "voice-memory.json"

    exit_code = main(
        [
            "--config",
            str(config_path),
            "--input",
            str(wav_path),
            "--output-csv",
            str(csv_path),
            "--output-summary",
            str(summary_path),
            "--frame-size",
            "128",
            "--hop-size",
            "64",
            "--pause-steps",
            "2",
            "--max-steps",
            "3",
        ]
    )

    assert exit_code == 0
    assert csv_path.exists()
    assert summary_path.exists()
