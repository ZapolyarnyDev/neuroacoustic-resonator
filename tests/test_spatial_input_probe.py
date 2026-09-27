from __future__ import annotations

import json

import numpy as np
import pytest
from scipy.io import wavfile  # type: ignore[import-untyped]

from neuroacoustic_resonator.analysis.spatial_input_probe import (
    run_spatial_input_probe,
)


def test_spatial_probe_keeps_silence_and_uncoupled_controls(tmp_path) -> None:
    rate = 8_000
    time = np.arange(rate // 2) / rate
    tone = (0.6 * np.sin(2 * np.pi * 220 * time)).astype(np.float32)
    wav_path = tmp_path / "tone.wav"
    report_path = tmp_path / "report.json"
    wavfile.write(wav_path, rate, tone)

    report = run_spatial_input_probe(
        wav_path,
        "configs/field_only.yaml",
        report_path,
        seed=23,
        response_seconds=0.5,
    )
    assert report["status"] == "development_probe_not_stage_one_evidence"
    assert report["source_positions"] == [0.5]
    assert report["steps"] == 50
    assert max(report["transport_output_mass"]) > 0.0
    assert max(report["uncoupled_output_phase_rms_delta"]) == pytest.approx(0.0)
    assert json.loads(report_path.read_text(encoding="utf-8"))["field_seed"] == 23


def test_spatial_probe_uses_two_fixed_inlets_for_stereo(tmp_path) -> None:
    rate = 8_000
    time = np.arange(rate // 8) / rate
    tone = np.sin(2 * np.pi * 440 * time)
    stereo = np.column_stack((tone, tone * 0.25)).astype(np.float32)
    wav_path = tmp_path / "stereo.wav"
    wavfile.write(wav_path, rate, stereo)

    report = run_spatial_input_probe(
        wav_path,
        "configs/field_only.yaml",
        tmp_path / "stereo-report.json",
        seed=23,
        response_seconds=0.0,
    )
    assert report["source_positions"] == [0.25, 0.75]
    assert report["transport_config"]["source_positions"] == [0.25, 0.75]
