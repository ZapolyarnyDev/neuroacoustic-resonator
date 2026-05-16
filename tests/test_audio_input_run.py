from __future__ import annotations

import csv
import json

import numpy as np
from scipy.io import wavfile  # type: ignore[import-untyped]

from neuroacoustic_resonator.analysis.audio_input_run import (
    AudioInputRunConfig,
    main,
    run_audio_input_simulation,
)


def test_run_audio_input_simulation_writes_rows_and_summary(tmp_path) -> None:
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
    csv_path = tmp_path / "audio-input.csv"
    summary_path = tmp_path / "audio-input.json"
    samples = np.zeros(512, dtype=np.float32)
    samples[128:256] = 1.0
    wavfile.write(wav_path, 8_000, samples)

    summary = run_audio_input_simulation(
        AudioInputRunConfig(
            config_path=config_path,
            input_wav=wav_path,
            output_csv=csv_path,
            output_summary=summary_path,
            frame_size=128,
            hop_size=64,
            warmup_steps=2,
            max_steps=5,
        )
    )

    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    loaded = json.loads(summary_path.read_text(encoding="utf-8"))

    assert len(rows) == 5
    assert rows[0]["audio_step"] == "0"
    assert "output_response_activity" in rows[0]
    assert "output_fast_response_score" in rows[0]
    assert summary["rows"] == 5
    assert summary["peak_input_value"] > 0.0
    assert "peak_output_response_activity" in summary
    assert "input_output_lag_steps" in loaded


def test_audio_input_run_main_writes_outputs(tmp_path) -> None:
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
    csv_path = tmp_path / "audio-input.csv"
    summary_path = tmp_path / "audio-input.json"
    wavfile.write(wav_path, 8_000, np.ones(256, dtype=np.float32))

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
            "--max-steps",
            "3",
        ]
    )

    assert exit_code == 0
    assert csv_path.exists()
    assert summary_path.exists()
