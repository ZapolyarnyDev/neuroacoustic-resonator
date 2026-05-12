from __future__ import annotations

from pathlib import Path
import wave

import numpy as np
from numpy.typing import NDArray

from neuroacoustic_resonator.field import FieldState
from neuroacoustic_resonator.regions import RegionMasks

AudioArray = NDArray[np.float64]

TAU = 2.0 * np.pi


def render_output_frame(
    state: FieldState,
    regions: RegionMasks,
    *,
    sample_rate: int = 48_000,
    frame_size: int = 512,
    gain: float = 0.2,
) -> AudioArray:
    if sample_rate < 1:
        msg = "sample_rate must be positive"
        raise ValueError(msg)
    if frame_size < 1:
        msg = "frame_size must be positive"
        raise ValueError(msg)
    if gain < 0.0:
        msg = "gain must be non-negative"
        raise ValueError(msg)
    if state.phase.shape != regions.shape:
        msg = "state and regions must have matching shapes"
        raise ValueError(msg)

    mask = regions.output
    phase = state.phase[mask]
    frequency = np.clip(state.frequency[mask], 0.0, None)
    metabolite = state.metabolite[mask]
    coupling = state.coupling[mask]

    if phase.size == 0:
        return np.zeros(frame_size, dtype=np.float64)

    weights = np.clip(metabolite * (1.0 + coupling), 0.0, None)
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0.0:
        weights = np.ones_like(weights) / float(weights.size)
    else:
        weights = weights / weight_sum

    time = np.arange(frame_size, dtype=np.float64) / float(sample_rate)
    angular_frequency = TAU * frequency[:, np.newaxis]
    oscillator_bank = np.sin(phase[:, np.newaxis] + angular_frequency * time)
    frame = gain * np.sum(weights[:, np.newaxis] * oscillator_bank, axis=0)

    return np.clip(frame, -1.0, 1.0).astype(np.float64, copy=False)


def write_wav(path: str | Path, audio: AudioArray, *, sample_rate: int) -> Path:
    if sample_rate < 1:
        msg = "sample_rate must be positive"
        raise ValueError(msg)

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = np.asarray(np.round(clipped * 32767.0), dtype=np.int16)
    with wave.open(str(output_path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(pcm.tobytes())
    return output_path
