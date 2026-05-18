from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import wavfile  # type: ignore[import-untyped]

from neuroacoustic_resonator.analysis.audio_input_run import (
    AudioInputRunConfig,
    AudioInputRows,
    run_audio_input_simulation,
)

VoiceProbeSummary = dict[str, Any]


@dataclass(frozen=True)
class VoiceVsSilenceProbeConfig:
    config_path: Path = Path("configs") / "field_only.yaml"
    input_wav: Path = Path("experiments") / "audio" / "my_voice.wav"
    output_dir: Path = Path("experiments") / "logs"
    prefix: str = "voice_vs_silence"
    frame_size: int = 1024
    hop_size: int = 512
    drive_strength: float = 0.45
    warmup_steps: int = 100
    max_steps: int | None = None

    def __post_init__(self) -> None:
        if self.frame_size < 1:
            msg = "frame_size must be positive"
            raise ValueError(msg)
        if self.hop_size < 1:
            msg = "hop_size must be positive"
            raise ValueError(msg)
        if self.drive_strength < 0.0:
            msg = "drive_strength must be non-negative"
            raise ValueError(msg)
        if self.warmup_steps < 0:
            msg = "warmup_steps must be non-negative"
            raise ValueError(msg)
        if self.max_steps is not None and self.max_steps < 1:
            msg = "max_steps must be positive"
            raise ValueError(msg)


def run_voice_vs_silence_probe(
    config: VoiceVsSilenceProbeConfig,
) -> VoiceProbeSummary:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    silence_path = config.output_dir / f"{config.prefix}_silence.wav"
    write_matching_silence(config.input_wav, silence_path)

    voice_csv = config.output_dir / f"{config.prefix}_voice.csv"
    voice_summary = config.output_dir / f"{config.prefix}_voice_summary.json"
    silence_csv = config.output_dir / f"{config.prefix}_silence.csv"
    silence_summary = config.output_dir / f"{config.prefix}_silence_summary.json"

    voice = run_audio_input_simulation(
        AudioInputRunConfig(
            config_path=config.config_path,
            input_wav=config.input_wav,
            output_csv=voice_csv,
            output_summary=voice_summary,
            frame_size=config.frame_size,
            hop_size=config.hop_size,
            drive_strength=config.drive_strength,
            warmup_steps=config.warmup_steps,
            max_steps=config.max_steps,
        )
    )
    silence = run_audio_input_simulation(
        AudioInputRunConfig(
            config_path=config.config_path,
            input_wav=silence_path,
            output_csv=silence_csv,
            output_summary=silence_summary,
            frame_size=config.frame_size,
            hop_size=config.hop_size,
            drive_strength=config.drive_strength,
            warmup_steps=config.warmup_steps,
            max_steps=config.max_steps,
        )
    )

    voice_rows = read_rows(voice_csv)
    silence_rows = read_rows(silence_csv)
    summary = summarize_voice_probe(
        voice_rows,
        silence_rows,
        voice_summary=voice,
        silence_summary=silence,
        config=config,
        silence_path=silence_path,
    )
    output_summary = config.output_dir / f"{config.prefix}_summary.json"
    output_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def write_matching_silence(input_wav: str | Path, output_wav: str | Path) -> Path:
    sample_rate, samples = wavfile.read(input_wav)
    silence = np.zeros_like(samples)
    output = Path(output_wav)
    output.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(output, sample_rate, silence)
    return output


def summarize_voice_probe(
    voice_rows: AudioInputRows,
    silence_rows: AudioInputRows,
    *,
    voice_summary: dict[str, Any],
    silence_summary: dict[str, Any],
    config: VoiceVsSilenceProbeConfig,
    silence_path: Path,
) -> VoiceProbeSummary:
    voice_stats = summarize_rows(voice_rows)
    silence_stats = summarize_rows(silence_rows)
    return {
        "config": str(config.config_path),
        "input_wav": str(config.input_wav),
        "silence_wav": str(silence_path),
        "parameters": {
            "frame_size": config.frame_size,
            "hop_size": config.hop_size,
            "drive_strength": config.drive_strength,
            "warmup_steps": config.warmup_steps,
            "max_steps": config.max_steps,
        },
        "voice": voice_stats,
        "silence": silence_stats,
        "voice_summary": voice_summary,
        "silence_summary": silence_summary,
        "ratios": {
            "peak_fast_response": safe_ratio(
                voice_stats["peak_output_fast_response_score"],
                silence_stats["peak_output_fast_response_score"],
            ),
            "peak_event_score": safe_ratio(
                voice_stats["peak_output_event_score"],
                silence_stats["peak_output_event_score"],
            ),
            "peak_response_activity": safe_ratio(
                voice_stats["peak_output_response_activity"],
                silence_stats["peak_output_response_activity"],
            ),
            "absolute_output_delta": safe_ratio(
                abs(voice_stats["output_activity_delta"]),
                abs(silence_stats["output_activity_delta"]),
            ),
        },
    }


def summarize_rows(rows: AudioInputRows) -> dict[str, float | int]:
    if not rows:
        msg = "rows must not be empty"
        raise ValueError(msg)
    output_activity = column(rows, "output_activity")
    output_response = column(rows, "output_response_activity")
    fast = column(rows, "output_fast_response_score")
    event = column(rows, "output_event_score")
    input_value = column(rows, "input_value")
    return {
        "rows": len(rows),
        "duration_seconds": float(rows[-1]["time_seconds"]),
        "peak_input_value": float(np.max(input_value)),
        "active_input_frames_005": int(np.sum(input_value >= 0.05)),
        "active_input_frames_010": int(np.sum(input_value >= 0.10)),
        "output_activity_start": float(output_activity[0]),
        "output_activity_end": float(output_activity[-1]),
        "output_activity_delta": float(output_activity[-1] - output_activity[0]),
        "peak_output_activity": float(np.max(output_activity)),
        "peak_output_response_activity": float(np.max(output_response)),
        "mean_output_response_activity": float(np.mean(output_response)),
        "peak_output_fast_response_score": float(np.max(fast)),
        "mean_output_fast_response_score": float(np.mean(fast)),
        "peak_output_event_score": float(np.max(event)),
        "mean_output_event_score": float(np.mean(event)),
    }


def read_rows(path: str | Path) -> AudioInputRows:
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(stream)
        ]


def column(rows: AudioInputRows, key: str) -> np.ndarray:
    return np.asarray([row[key] for row in rows], dtype=np.float64)


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / max(denominator, 1e-12))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare voice-driven field response against matching silence.",
    )
    parser.add_argument(
        "--config", type=Path, default=VoiceVsSilenceProbeConfig.config_path
    )
    parser.add_argument(
        "--input", type=Path, required=True, help="Input voice WAV path."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=VoiceVsSilenceProbeConfig.output_dir
    )
    parser.add_argument("--prefix", type=str, default=VoiceVsSilenceProbeConfig.prefix)
    parser.add_argument("--frame-size", type=int, default=1024)
    parser.add_argument("--hop-size", type=int, default=512)
    parser.add_argument("--drive-strength", type=float, default=0.45)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = VoiceVsSilenceProbeConfig(
        config_path=args.config,
        input_wav=args.input,
        output_dir=args.output_dir,
        prefix=args.prefix,
        frame_size=args.frame_size,
        hop_size=args.hop_size,
        drive_strength=args.drive_strength,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
    )
    summary = run_voice_vs_silence_probe(config)
    ratios = summary["ratios"]
    print(
        "Voice probe: "
        f"fast_ratio={ratios['peak_fast_response']:.3f} "
        f"event_ratio={ratios['peak_event_score']:.3f} "
        f"response_ratio={ratios['peak_response_activity']:.3f} "
        f"drift_ratio={ratios['absolute_output_delta']:.3f}"
    )
    print(f"Wrote summary: {config.output_dir / f'{config.prefix}_summary.json'}")
    return 0
