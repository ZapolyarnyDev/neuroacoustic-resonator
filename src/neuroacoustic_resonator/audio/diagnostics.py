from __future__ import annotations

from typing import Any

import numpy as np

from neuroacoustic_resonator.analysis.output_patterns import OutputPatternHistory


def summarize_pattern_audio(
    audio: np.ndarray,
    pattern_history: OutputPatternHistory,
    *,
    sample_rate: int,
    frame_size: int,
) -> dict[str, Any]:
    if sample_rate < 1:
        msg = "sample_rate must be positive"
        raise ValueError(msg)
    if frame_size < 1:
        msg = "frame_size must be positive"
        raise ValueError(msg)
    samples = np.asarray(audio, dtype=np.float64).reshape(-1)
    frame_count = min(samples.size // frame_size, len(pattern_history.signatures))
    if frame_count == 0:
        return {
            "frames": 0,
            "duration_seconds": 0.0,
            "overall": audio_frame_metrics(
                np.zeros(0, dtype=np.float64),
                sample_rate=sample_rate,
            ),
            "by_pattern": {},
        }

    by_pattern_frames: dict[str, list[np.ndarray]] = {}
    for frame_index in range(frame_count):
        label = pattern_history.signatures[frame_index].label
        start = frame_index * frame_size
        frame = samples[start : start + frame_size]
        by_pattern_frames.setdefault(label, []).append(frame)

    return {
        "frames": frame_count,
        "duration_seconds": float(frame_count * frame_size / sample_rate),
        "overall": audio_frame_metrics(
            samples[: frame_count * frame_size],
            sample_rate=sample_rate,
        ),
        "by_pattern": {
            label: {
                "frames": len(frames),
                "duration_seconds": float(len(frames) * frame_size / sample_rate),
                **audio_frame_metrics(
                    np.concatenate(frames),
                    sample_rate=sample_rate,
                ),
            }
            for label, frames in sorted(by_pattern_frames.items())
        },
    }


def audio_frame_metrics(audio: np.ndarray, *, sample_rate: int) -> dict[str, float]:
    samples = np.asarray(audio, dtype=np.float64).reshape(-1)
    if samples.size == 0:
        return {
            "rms": 0.0,
            "peak": 0.0,
            "zero_crossing_rate": 0.0,
            "spectral_centroid_hz": 0.0,
        }
    rms = float(np.sqrt(np.mean(samples * samples)))
    peak = float(np.max(np.abs(samples)))
    crossings = np.count_nonzero(np.diff(np.signbit(samples)))
    zero_crossing_rate = float(crossings / max(samples.size - 1, 1))
    spectrum = np.abs(np.fft.rfft(samples))
    spectrum_sum = float(np.sum(spectrum))
    if spectrum_sum <= 1e-12:
        centroid = 0.0
    else:
        frequencies = np.fft.rfftfreq(samples.size, d=1.0 / sample_rate)
        centroid = float(np.sum(frequencies * spectrum) / spectrum_sum)
    return {
        "rms": rms,
        "peak": peak,
        "zero_crossing_rate": zero_crossing_rate,
        "spectral_centroid_hz": centroid,
    }
