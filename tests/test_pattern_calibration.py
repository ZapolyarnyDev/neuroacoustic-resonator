from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile  # type: ignore[import-untyped]

from neuroacoustic_resonator.analysis.pattern_calibration import (
    CalibrationSeedSplit,
    CalibrationStimulus,
    PatternCalibrationConfig,
    SyntheticStimulusSpec,
    main,
    run_pattern_calibration,
    synthetic_stimulus_audio,
)
from neuroacoustic_resonator.protocol import (
    read_protocol_jsonl,
    write_protocol_jsonl,
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
    manifest_path = tmp_path / "manifest.json"
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
            output_manifest=manifest_path,
            seed_roots=(11, 29),
            seed_splits=(
                CalibrationSeedSplit(11, "train"),
                CalibrationSeedSplit(29, "test"),
            ),
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
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert len(rows) == 8
    assert {row["protocol_version"] for row in rows} == {"0.1"}
    assert summary["rows"] == 8
    assert set(summary["stimuli"]) == {"voice", "tone"}
    assert "reinforcement" in summary
    assert loaded["reinforcement"]["sample_count"] == 8
    assert "response_audio_spectral_centroid_hz" in rows[0]
    assert (output_dir / "generated_inputs" / "tone.wav").exists()
    assert len(manifest) == 8
    assert len({row["field_seed"] for row in manifest}) == 4
    assert {row["seed_root"] for row in manifest} == {11, 29}
    assert {row["split"] for row in manifest} == {"train", "test"}
    assert len({row["trial_id"] for row in manifest}) == 8
    protocol_path = Path(manifest[0]["protocol_jsonl"])
    metadata_path = Path(manifest[0]["metadata_json"])
    replay_path = tmp_path / "strict-replay.jsonl"
    frames = read_protocol_jsonl(protocol_path)
    write_protocol_jsonl(replay_path, frames)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert replay_path.read_bytes() == protocol_path.read_bytes()
    assert metadata["stimulus_label"] == manifest[0]["stimulus_label"]
    assert metadata["field_seed"] == manifest[0]["field_seed"]
    assert metadata["split"] == manifest[0]["split"]
    assert metadata["protocol_frames"] == len(frames)
    assert metadata["segments"]["response"]["frames"] > 0
    assert metadata["segments"]["response"]["sequence_end"] == frames[-1].sequence
    assert [frame.sequence for frame in frames] == sorted(
        frame.sequence for frame in frames
    )


def test_pattern_calibration_main_writes_outputs(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    write_config(config_path)
    csv_path = tmp_path / "calibration.csv"
    summary_path = tmp_path / "calibration.json"
    manifest_path = tmp_path / "manifest.json"
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
            "--output-manifest",
            str(manifest_path),
            "--seed",
            "11",
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
    assert manifest_path.exists()


def test_pattern_calibration_requires_stimuli() -> None:
    with pytest.raises(ValueError, match="stimulus"):
        PatternCalibrationConfig(stimuli=(), synthetic_stimuli=())
