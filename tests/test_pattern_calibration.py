from __future__ import annotations

import csv
import json

import numpy as np
import pytest
from scipy.io import wavfile  # type: ignore[import-untyped]

from neuroacoustic_resonator.analysis.pattern_calibration import (
    CalibrationStimulus,
    PatternCalibrationConfig,
    SyntheticStimulusSpec,
    main,
    run_pattern_calibration,
    synthetic_stimulus_audio,
)


def write_config(path) -> None:
    path.write_text(
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


def test_synthetic_stimulus_audio_generates_bounded_signal() -> None:
    audio = synthetic_stimulus_audio(
        SyntheticStimulusSpec(
            label="chirp",
            kind="chirp",
            duration_seconds=0.1,
            sample_rate=1_000,
        )
    )

    assert audio.shape == (100,)
    assert np.max(np.abs(audio)) <= 1.0


def test_run_pattern_calibration_writes_rows_summary_and_reinforcement(
    tmp_path,
) -> None:
    config_path = tmp_path / "config.yaml"
    write_config(config_path)
    wav_path = tmp_path / "voice.wav"
    samples = np.zeros(512, dtype=np.float32)
    samples[128:256] = 0.8
    wavfile.write(wav_path, 8_000, samples)
    csv_path = tmp_path / "calibration.csv"
    summary_path = tmp_path / "calibration.json"
    output_dir = tmp_path / "calibration"

    summary = run_pattern_calibration(
        PatternCalibrationConfig(
            config_path=config_path,
            stimuli=(CalibrationStimulus("voice", wav_path),),
            synthetic_stimuli=(
                SyntheticStimulusSpec(
                    label="tone",
                    kind="tone",
                    duration_seconds=0.1,
                    sample_rate=8_000,
                ),
            ),
            output_dir=output_dir,
            output_csv=csv_path,
            output_summary=summary_path,
            repeats=2,
            sample_rate=8_000,
            output_frame_size=80,
            input_frame_size=128,
            input_hop_size=64,
            response_seconds=0.1,
            warmup_steps=2,
        )
    )

    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    loaded = json.loads(summary_path.read_text(encoding="utf-8"))

    assert len(rows) == 4
    assert summary["rows"] == 4
    assert set(summary["stimuli"]) == {"voice", "tone"}
    assert "reinforcement" in summary
    assert loaded["reinforcement"]["sample_count"] == 4
    assert "response_audio_spectral_centroid_hz" in rows[0]
    assert (output_dir / "generated_inputs" / "tone.wav").exists()


def test_pattern_calibration_main_writes_outputs(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    write_config(config_path)
    csv_path = tmp_path / "calibration.csv"
    summary_path = tmp_path / "calibration.json"
    output_dir = tmp_path / "calibration"

    exit_code = main(
        [
            "--config",
            str(config_path),
            "--synthetic",
            "tone:tone:220:0.1",
            "--output-dir",
            str(output_dir),
            "--output-csv",
            str(csv_path),
            "--output-summary",
            str(summary_path),
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
            "--warmup-steps",
            "2",
        ]
    )

    assert exit_code == 0
    assert csv_path.exists()
    assert summary_path.exists()


def test_pattern_calibration_requires_stimuli() -> None:
    with pytest.raises(ValueError, match="stimulus"):
        PatternCalibrationConfig(stimuli=(), synthetic_stimuli=())
