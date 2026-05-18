from __future__ import annotations

import numpy as np
import pytest
from scipy.io import wavfile  # type: ignore[import-untyped]

from neuroacoustic_resonator import FieldConfig, OscillatorField, RegionMasks
from neuroacoustic_resonator.audio.input import (
    AudioInputFeatures,
    WavInputDrive,
    extract_audio_input_features,
    main,
    write_audio_input_features_csv,
)
from neuroacoustic_resonator.core.field import TAU


def test_extract_audio_input_features_from_wav(tmp_path) -> None:
    wav_path = tmp_path / "voice.wav"
    sample_rate = 8_000
    time = np.arange(sample_rate // 2) / sample_rate
    audio = np.sin(2.0 * np.pi * 440.0 * time) * np.linspace(0.0, 1.0, time.size)
    wavfile.write(wav_path, sample_rate, audio.astype(np.float32))

    features = extract_audio_input_features(
        wav_path,
        frame_size=256,
        hop_size=128,
        drive_strength=0.5,
    )

    assert features.sample_rate == sample_rate
    assert features.frame_count > 1
    assert features.rms.shape == features.drive.shape
    assert np.max(features.drive) <= 0.5
    assert np.max(features.rms) > 0.0
    assert np.max(features.spectral_centroid) > 0.0


def test_wav_input_drive_applies_features_to_input_region(tmp_path) -> None:
    wav_path = tmp_path / "impulse.wav"
    samples = np.zeros(512, dtype=np.float32)
    samples[128:256] = 1.0
    wavfile.write(wav_path, 8_000, samples)
    features = extract_audio_input_features(
        wav_path,
        frame_size=128,
        hop_size=64,
        drive_strength=0.4,
    )
    field = OscillatorField(FieldConfig(size=6, seed=1))
    regions = RegionMasks.from_size(6)
    drive = WavInputDrive(features, regions)
    before = field.state.phase.copy()

    applied = drive.apply(field, int(np.argmax(features.drive)))

    assert applied > 0.0
    assert np.allclose(
        field.state.phase[regions.input],
        np.mod(before[regions.input] + applied, TAU),
    )
    assert np.allclose(field.state.phase[~regions.input], before[~regions.input])


def test_wav_input_drive_can_boost_assoc_and_output_regions(tmp_path) -> None:
    wav_path = tmp_path / "input.wav"
    wavfile.write(wav_path, 8_000, np.ones(128, dtype=np.float32))
    features = extract_audio_input_features(
        wav_path,
        frame_size=64,
        hop_size=64,
        drive_strength=0.2,
    )
    field = OscillatorField(FieldConfig(size=6, seed=1))
    regions = RegionMasks.from_size(6)
    drive = WavInputDrive(features, regions, assoc_gain=0.5, output_gain=0.25)
    before = field.state.phase.copy()

    applied = drive.apply(field, 0)

    assert applied > 0.0
    assert np.allclose(
        field.state.phase[regions.input],
        np.mod(before[regions.input] + applied, TAU),
    )
    assert np.allclose(
        field.state.phase[regions.assoc],
        np.mod(before[regions.assoc] + applied * 0.5, TAU),
    )
    assert np.allclose(
        field.state.phase[regions.output],
        np.mod(before[regions.output] + applied * 0.25, TAU),
    )


def test_wav_input_drive_rejects_negative_region_gains() -> None:
    features = AudioInputFeatures(
        sample_rate=8_000,
        frame_size=64,
        hop_size=64,
        rms=np.asarray([1.0]),
        onset=np.asarray([0.0]),
        spectral_centroid=np.asarray([0.0]),
        drive=np.asarray([0.2]),
    )
    regions = RegionMasks.from_size(6)

    with pytest.raises(ValueError, match="assoc_gain"):
        WavInputDrive(features, regions, assoc_gain=-0.1)
    with pytest.raises(ValueError, match="output_gain"):
        WavInputDrive(features, regions, output_gain=-0.1)


def test_audio_input_features_value_after_end_is_zero(tmp_path) -> None:
    wav_path = tmp_path / "short.wav"
    wavfile.write(wav_path, 8_000, np.ones(128, dtype=np.float32))
    features = extract_audio_input_features(wav_path, frame_size=64, hop_size=64)

    assert features.value_at_step(features.frame_count + 5) == 0.0
    with pytest.raises(ValueError, match="step"):
        features.value_at_step(-1)


def test_write_audio_input_features_csv(tmp_path) -> None:
    wav_path = tmp_path / "voice.wav"
    output_path = tmp_path / "features.csv"
    wavfile.write(wav_path, 8_000, np.ones(256, dtype=np.float32))
    features = extract_audio_input_features(wav_path, frame_size=128, hop_size=64)

    written = write_audio_input_features_csv(output_path, features)

    text = written.read_text(encoding="utf-8")
    assert text.startswith("step,time_seconds,rms,onset,spectral_centroid,drive")
    assert len(text.splitlines()) == features.frame_count + 1


def test_audio_input_main_writes_csv(tmp_path) -> None:
    wav_path = tmp_path / "voice.wav"
    output_path = tmp_path / "features.csv"
    wavfile.write(wav_path, 8_000, np.ones(256, dtype=np.float32))

    exit_code = main(
        [
            "--input",
            str(wav_path),
            "--output",
            str(output_path),
            "--frame-size",
            "128",
            "--hop-size",
            "64",
        ]
    )

    assert exit_code == 0
    assert output_path.exists()
