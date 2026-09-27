from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import wavfile  # type: ignore[import-untyped]

from neuroacoustic_resonator.core.field import OscillatorField
from neuroacoustic_resonator.core.regions import RegionMasks


@dataclass(frozen=True)
class AudioInputFeatures:
    sample_rate: int
    frame_size: int
    hop_size: int
    rms: np.ndarray
    onset: np.ndarray
    spectral_centroid: np.ndarray
    drive: np.ndarray

    @property
    def frame_count(self) -> int:
        return int(self.drive.shape[0])

    @property
    def duration_seconds(self) -> float:
        if self.sample_rate <= 0:
            return 0.0
        return self.frame_count * self.hop_size / self.sample_rate

    def value_at_step(self, step: int) -> float:
        if step < 0:
            msg = "step must be non-negative"
            raise ValueError(msg)
        if step >= self.frame_count:
            return 0.0
        return float(self.drive[step])

    def rows(self) -> list[dict[str, float | int]]:
        return [
            {
                "step": step,
                "time_seconds": step * self.hop_size / self.sample_rate,
                "rms": float(self.rms[step]),
                "onset": float(self.onset[step]),
                "spectral_centroid": float(self.spectral_centroid[step]),
                "drive": float(self.drive[step]),
            }
            for step in range(self.frame_count)
        ]


@dataclass(frozen=True)
class ChannelAudioFeatures:
    sample_rate: int
    hop_size: int
    sample_count: int
    band_energy: np.ndarray

    @property
    def channel_count(self) -> int:
        return int(self.band_energy.shape[1])

    @property
    def frame_count(self) -> int:
        return int(self.band_energy.shape[0])

    @property
    def duration_seconds(self) -> float:
        return self.sample_count / self.sample_rate

    def energy_at_time(self, time_seconds: float) -> np.ndarray:
        if time_seconds < 0.0:
            msg = "time_seconds must be non-negative"
            raise ValueError(msg)
        if time_seconds >= self.duration_seconds:
            return np.zeros(self.band_energy.shape[1:], dtype=np.float64)
        index = min(
            int(time_seconds * self.sample_rate / self.hop_size), self.frame_count - 1
        )
        return self.band_energy[index].copy()


def extract_channel_audio_features(
    wav_path: str | Path,
    *,
    frame_size: int = 1024,
    hop_size: int = 512,
    bands: int = 8,
) -> ChannelAudioFeatures:
    sample_rate, samples = wavfile.read(wav_path)
    return extract_channel_audio_array_features(
        samples,
        sample_rate=int(sample_rate),
        frame_size=frame_size,
        hop_size=hop_size,
        bands=bands,
    )


def extract_channel_audio_array_features(
    samples: np.ndarray,
    *,
    sample_rate: int,
    frame_size: int = 1024,
    hop_size: int = 512,
    bands: int = 8,
) -> ChannelAudioFeatures:
    if sample_rate < 1 or frame_size < 2 or hop_size < 1 or bands < 1:
        msg = "sample_rate, hop_size, bands must be positive; frame_size must exceed 1"
        raise ValueError(msg)
    audio = _to_channels_float(samples)
    frames = np.stack(
        [
            _frame_audio(audio[:, channel], frame_size=frame_size, hop_size=hop_size)
            for channel in range(audio.shape[1])
        ],
        axis=1,
    )
    spectrum = np.abs(np.fft.rfft(frames, axis=2))
    frequencies = np.fft.rfftfreq(frame_size, d=1.0 / sample_rate)
    edges = np.linspace(0.0, sample_rate / 2.0, bands + 1)
    energies = np.zeros((*spectrum.shape[:2], bands), dtype=np.float64)
    for band in range(bands):
        selected = (frequencies >= edges[band]) & (
            frequencies < edges[band + 1]
            if band < bands - 1
            else frequencies <= edges[band + 1]
        )
        energies[:, :, band] = (
            np.sqrt(np.sum(spectrum[:, :, selected] ** 2, axis=2)) / frame_size
        )
    return ChannelAudioFeatures(
        sample_rate=sample_rate,
        hop_size=hop_size,
        sample_count=audio.shape[0],
        band_energy=energies,
    )


class WavInputDrive:
    def __init__(
        self,
        features: AudioInputFeatures,
        regions: RegionMasks,
        *,
        assoc_gain: float = 0.0,
        output_gain: float = 0.0,
    ) -> None:
        if assoc_gain < 0.0:
            msg = "assoc_gain must be non-negative"
            raise ValueError(msg)
        if output_gain < 0.0:
            msg = "output_gain must be non-negative"
            raise ValueError(msg)
        self.features = features
        self.regions = regions
        self.assoc_gain = assoc_gain
        self.output_gain = output_gain

    def value(self, step: int) -> float:
        return self.features.value_at_step(step)

    def apply(self, field: OscillatorField, step: int) -> float:
        value = self.value(step)
        if value != 0.0:
            field.apply_phase_impulse(self.regions.input, value)
            if self.assoc_gain != 0.0:
                field.apply_phase_impulse(self.regions.assoc, value * self.assoc_gain)
            if self.output_gain != 0.0:
                field.apply_phase_impulse(
                    self.regions.output,
                    value * self.output_gain,
                )
        return value


def extract_audio_input_features(
    wav_path: str | Path,
    *,
    frame_size: int = 1024,
    hop_size: int = 512,
    drive_strength: float = 0.45,
    rms_weight: float = 0.45,
    onset_weight: float = 0.4,
    centroid_weight: float = 0.15,
) -> AudioInputFeatures:
    sample_rate, samples = wavfile.read(wav_path)
    return extract_audio_array_features(
        samples,
        sample_rate=int(sample_rate),
        frame_size=frame_size,
        hop_size=hop_size,
        drive_strength=drive_strength,
        rms_weight=rms_weight,
        onset_weight=onset_weight,
        centroid_weight=centroid_weight,
    )


def extract_audio_array_features(
    samples: np.ndarray,
    *,
    sample_rate: int,
    frame_size: int = 1024,
    hop_size: int = 512,
    drive_strength: float = 0.45,
    rms_weight: float = 0.45,
    onset_weight: float = 0.4,
    centroid_weight: float = 0.15,
) -> AudioInputFeatures:
    if sample_rate < 1:
        msg = "sample_rate must be positive"
        raise ValueError(msg)
    if frame_size < 1:
        msg = "frame_size must be positive"
        raise ValueError(msg)
    if hop_size < 1:
        msg = "hop_size must be positive"
        raise ValueError(msg)
    if drive_strength < 0.0:
        msg = "drive_strength must be non-negative"
        raise ValueError(msg)

    audio = _to_mono_float(samples)
    frames = _frame_audio(audio, frame_size=frame_size, hop_size=hop_size)
    rms = _frame_rms(frames)
    spectrum = np.abs(np.fft.rfft(frames, axis=1))
    onset = _spectral_onset(spectrum)
    spectral_centroid = _spectral_centroid(spectrum, sample_rate=sample_rate)
    drive = _combine_features(
        rms,
        onset,
        spectral_centroid,
        drive_strength=drive_strength,
        rms_weight=rms_weight,
        onset_weight=onset_weight,
        centroid_weight=centroid_weight,
    )
    return AudioInputFeatures(
        sample_rate=sample_rate,
        frame_size=frame_size,
        hop_size=hop_size,
        rms=rms,
        onset=onset,
        spectral_centroid=spectral_centroid,
        drive=drive,
    )


def write_audio_input_features_csv(
    path: str | Path,
    features: AudioInputFeatures,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "step",
                "time_seconds",
                "rms",
                "onset",
                "spectral_centroid",
                "drive",
            ),
        )
        writer.writeheader()
        writer.writerows(features.rows())
    return output


def _to_mono_float(samples: np.ndarray) -> np.ndarray:
    return np.mean(_to_channels_float(samples), axis=1)


def _to_channels_float(samples: np.ndarray) -> np.ndarray:
    audio = np.asarray(samples)
    if audio.ndim == 1:
        audio = audio[:, np.newaxis]
    if audio.ndim != 2 or audio.shape[1] < 1:
        msg = "WAV input must have at least one channel"
        raise ValueError(msg)

    if np.issubdtype(audio.dtype, np.integer):
        info = np.iinfo(audio.dtype)
        integer_peak = max(abs(info.min), abs(info.max))
        return audio.astype(np.float64) / integer_peak

    floating = audio.astype(np.float64)
    floating_peak = float(np.max(np.abs(floating))) if floating.size else 0.0
    if floating_peak > 1.0:
        floating = floating / floating_peak
    return floating


def _frame_audio(audio: np.ndarray, *, frame_size: int, hop_size: int) -> np.ndarray:
    if audio.size == 0:
        return np.zeros((1, frame_size), dtype=np.float64)

    frame_count = max(1, int(np.ceil(max(1, audio.size - frame_size) / hop_size)) + 1)
    padded_length = (frame_count - 1) * hop_size + frame_size
    padded = np.pad(audio, (0, max(0, padded_length - audio.size)))
    frames = np.empty((frame_count, frame_size), dtype=np.float64)
    window = np.hanning(frame_size)
    for index in range(frame_count):
        start = index * hop_size
        frames[index] = padded[start : start + frame_size] * window
    return frames


def _frame_rms(frames: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(np.square(frames), axis=1))


def _spectral_onset(spectrum: np.ndarray) -> np.ndarray:
    normalized = _normalize_rows(spectrum)
    flux = np.zeros(normalized.shape[0], dtype=np.float64)
    if normalized.shape[0] > 1:
        diff = normalized[1:] - normalized[:-1]
        flux[1:] = np.sum(np.maximum(diff, 0.0), axis=1)
    return flux


def _spectral_centroid(spectrum: np.ndarray, *, sample_rate: int) -> np.ndarray:
    frequencies = np.fft.rfftfreq((spectrum.shape[1] - 1) * 2, d=1.0 / sample_rate)
    energy = np.sum(spectrum, axis=1)
    weighted = np.sum(spectrum * frequencies, axis=1)
    centroid_hz = np.divide(
        weighted,
        energy,
        out=np.zeros_like(weighted),
        where=energy > 1e-12,
    )
    nyquist = sample_rate / 2.0
    return centroid_hz / nyquist if nyquist > 0.0 else centroid_hz


def _combine_features(
    rms: np.ndarray,
    onset: np.ndarray,
    spectral_centroid: np.ndarray,
    *,
    drive_strength: float,
    rms_weight: float,
    onset_weight: float,
    centroid_weight: float,
) -> np.ndarray:
    weights_total = rms_weight + onset_weight + centroid_weight
    if weights_total <= 0.0:
        msg = "at least one feature weight must be positive"
        raise ValueError(msg)

    combined = (
        rms_weight * _normalize_feature(rms)
        + onset_weight * _normalize_feature(onset)
        + centroid_weight * np.clip(spectral_centroid, 0.0, 1.0)
    ) / weights_total
    return np.clip(combined * drive_strength, 0.0, drive_strength)


def _normalize_feature(values: np.ndarray) -> np.ndarray:
    peak = float(np.max(values)) if values.size else 0.0
    if peak <= 1e-12:
        return np.zeros_like(values, dtype=np.float64)
    return values / peak


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    energy = np.sum(values, axis=1, keepdims=True)
    return np.divide(
        values,
        energy,
        out=np.zeros_like(values, dtype=np.float64),
        where=energy > 1e-12,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract offline WAV input features for the input field region.",
    )
    parser.add_argument("--input", type=Path, required=True, help="Input WAV path.")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV path.",
    )
    parser.add_argument("--frame-size", type=int, default=1024)
    parser.add_argument("--hop-size", type=int, default=512)
    parser.add_argument("--drive-strength", type=float, default=0.45)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    features = extract_audio_input_features(
        args.input,
        frame_size=args.frame_size,
        hop_size=args.hop_size,
        drive_strength=args.drive_strength,
    )
    output = write_audio_input_features_csv(args.output, features)
    print(
        f"Extracted audio input features: {output} "
        f"frames={features.frame_count} "
        f"duration={features.duration_seconds:.3f}s"
    )
    return 0
