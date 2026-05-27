from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import wavfile  # type: ignore[import-untyped]

from neuroacoustic_resonator.audio.conversation import to_mono_float

TurnRows = list[dict[str, Any]]


@dataclass(frozen=True)
class TurnDetectionConfig:
    input_wav: Path
    output_dir: Path = Path("experiments") / "audio" / "turns"
    frame_ms: float = 25.0
    hop_ms: float = 10.0
    threshold_ratio: float = 0.18
    min_voice_ms: float = 120.0
    min_silence_ms: float = 250.0
    padding_ms: float = 80.0

    def __post_init__(self) -> None:
        if self.frame_ms <= 0.0:
            msg = "frame_ms must be positive"
            raise ValueError(msg)
        if self.hop_ms <= 0.0:
            msg = "hop_ms must be positive"
            raise ValueError(msg)
        if not 0.0 < self.threshold_ratio <= 1.0:
            msg = "threshold_ratio must be in (0, 1]"
            raise ValueError(msg)
        if self.min_voice_ms < 0.0:
            msg = "min_voice_ms must be non-negative"
            raise ValueError(msg)
        if self.min_silence_ms < 0.0:
            msg = "min_silence_ms must be non-negative"
            raise ValueError(msg)
        if self.padding_ms < 0.0:
            msg = "padding_ms must be non-negative"
            raise ValueError(msg)


def detect_and_write_turns(config: TurnDetectionConfig) -> TurnRows:
    sample_rate, samples = wavfile.read(config.input_wav)
    audio = to_mono_float(samples)
    turns = detect_voice_turns(audio, sample_rate=sample_rate, config=config)
    return write_voice_turns(
        samples,
        sample_rate=sample_rate,
        turns=turns,
        output_dir=config.output_dir,
    )


def detect_voice_turns(
    audio: np.ndarray,
    *,
    sample_rate: int,
    config: TurnDetectionConfig,
) -> list[tuple[int, int]]:
    if sample_rate < 1:
        msg = "sample_rate must be positive"
        raise ValueError(msg)
    if audio.size == 0:
        return []

    frame_size = max(1, int(round(config.frame_ms * sample_rate / 1000.0)))
    hop_size = max(1, int(round(config.hop_ms * sample_rate / 1000.0)))
    rms = frame_rms(audio, frame_size=frame_size, hop_size=hop_size)
    if rms.size == 0:
        return []

    threshold = float(np.max(rms)) * config.threshold_ratio
    active = rms >= threshold
    min_voice_frames = max(1, int(round(config.min_voice_ms / config.hop_ms)))
    min_silence_frames = max(1, int(round(config.min_silence_ms / config.hop_ms)))
    padding_samples = int(round(config.padding_ms * sample_rate / 1000.0))

    turns: list[tuple[int, int]] = []
    start_frame: int | None = None
    silence_run = 0
    for frame_index, is_active in enumerate(active):
        if is_active:
            if start_frame is None:
                start_frame = frame_index
            silence_run = 0
            continue
        if start_frame is None:
            continue
        silence_run += 1
        if silence_run < min_silence_frames:
            continue
        end_frame = frame_index - silence_run + 1
        if end_frame - start_frame >= min_voice_frames:
            turns.append(
                frame_range_to_samples(
                    start_frame,
                    end_frame,
                    hop_size=hop_size,
                    frame_size=frame_size,
                    padding_samples=padding_samples,
                    sample_count=audio.size,
                )
            )
        start_frame = None
        silence_run = 0

    if start_frame is not None and active.size - start_frame >= min_voice_frames:
        turns.append(
            frame_range_to_samples(
                start_frame,
                active.size,
                hop_size=hop_size,
                frame_size=frame_size,
                padding_samples=padding_samples,
                sample_count=audio.size,
            )
        )
    return merge_close_turns(turns, max_gap_samples=padding_samples)


def frame_rms(audio: np.ndarray, *, frame_size: int, hop_size: int) -> np.ndarray:
    values: list[float] = []
    for start in range(0, max(1, audio.size - frame_size + 1), hop_size):
        frame = audio[start : start + frame_size]
        if frame.size < frame_size:
            frame = np.pad(frame, (0, frame_size - frame.size))
        values.append(float(np.sqrt(np.mean(np.square(frame)))))
    if not values:
        values.append(float(np.sqrt(np.mean(np.square(audio)))))
    return np.asarray(values, dtype=np.float64)


def frame_range_to_samples(
    start_frame: int,
    end_frame: int,
    *,
    hop_size: int,
    frame_size: int,
    padding_samples: int,
    sample_count: int,
) -> tuple[int, int]:
    start = max(0, start_frame * hop_size - padding_samples)
    end = min(sample_count, end_frame * hop_size + frame_size + padding_samples)
    return start, end


def merge_close_turns(
    turns: list[tuple[int, int]],
    *,
    max_gap_samples: int,
) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in turns:
        if not merged or start - merged[-1][1] > max_gap_samples:
            merged.append((start, end))
            continue
        previous_start, previous_end = merged[-1]
        merged[-1] = previous_start, max(previous_end, end)
    return merged


def write_voice_turns(
    samples: np.ndarray,
    *,
    sample_rate: int,
    turns: list[tuple[int, int]],
    output_dir: Path,
) -> TurnRows:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: TurnRows = []
    for index, (start, end) in enumerate(turns, start=1):
        output = output_dir / f"turn_{index:03d}.wav"
        wavfile.write(output, sample_rate, samples[start:end])
        rows.append(
            {
                "index": index,
                "path": str(output),
                "start_sample": start,
                "end_sample": end,
                "start_seconds": start / sample_rate,
                "end_seconds": end / sample_rate,
                "duration_seconds": (end - start) / sample_rate,
            }
        )
    summary = {
        "sample_rate": sample_rate,
        "turn_count": len(rows),
        "turns": rows,
    }
    (output_dir / "turns.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split a voice WAV into utterance turns."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=TurnDetectionConfig.output_dir
    )
    parser.add_argument("--frame-ms", type=float, default=25.0)
    parser.add_argument("--hop-ms", type=float, default=10.0)
    parser.add_argument("--threshold-ratio", type=float, default=0.18)
    parser.add_argument("--min-voice-ms", type=float, default=120.0)
    parser.add_argument("--min-silence-ms", type=float, default=250.0)
    parser.add_argument("--padding-ms", type=float, default=80.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = TurnDetectionConfig(
        input_wav=args.input,
        output_dir=args.output_dir,
        frame_ms=args.frame_ms,
        hop_ms=args.hop_ms,
        threshold_ratio=args.threshold_ratio,
        min_voice_ms=args.min_voice_ms,
        min_silence_ms=args.min_silence_ms,
        padding_ms=args.padding_ms,
    )
    rows = detect_and_write_turns(config)
    print(f"Detected voice turns: count={len(rows)} output_dir={config.output_dir}")
    print(json.dumps({"config": asdict(config), "turns": rows}, indent=2, default=str))
    return 0
