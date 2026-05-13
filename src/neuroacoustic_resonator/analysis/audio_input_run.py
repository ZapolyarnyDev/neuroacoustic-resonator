from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from neuroacoustic_resonator.analysis.metrics import (
    RegionalActivityMetrics,
    RegionalActivityTracker,
)
from neuroacoustic_resonator.audio.input import (
    AudioInputFeatures,
    WavInputDrive,
    extract_audio_input_features,
)
from neuroacoustic_resonator.core.config import SimulationConfig
from neuroacoustic_resonator.core.regions import RegionMasks
from neuroacoustic_resonator.core.simulation import Simulation, SimulationFrame

AudioInputRows = list[dict[str, Any]]
AudioInputSummary = dict[str, Any]


@dataclass(frozen=True)
class AudioInputRunConfig:
    config_path: Path = Path("configs") / "field_only.yaml"
    input_wav: Path = Path("experiments") / "audio" / "input.wav"
    output_csv: Path = Path("experiments") / "logs" / "audio_input_run.csv"
    output_summary: Path = Path("experiments") / "logs" / "audio_input_run_summary.json"
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


def run_audio_input_simulation(config: AudioInputRunConfig) -> AudioInputSummary:
    sim_config = SimulationConfig.from_file(config.config_path)
    simulation = Simulation.from_config(sim_config)
    regions = RegionMasks.from_size(sim_config.field.size)
    tracker = RegionalActivityTracker()
    features = extract_audio_input_features(
        config.input_wav,
        frame_size=config.frame_size,
        hop_size=config.hop_size,
        drive_strength=config.drive_strength,
    )
    drive = WavInputDrive(features, regions)

    for _ in range(config.warmup_steps):
        simulation.step()

    steps = features.frame_count
    if config.max_steps is not None:
        steps = min(steps, config.max_steps)

    rows: AudioInputRows = []
    for step in range(steps):
        input_value = drive.apply(simulation.field, step)
        simulation.step_index += 1
        state = simulation.field.step()
        simulation.last_input_value = input_value
        frame = SimulationFrame(
            state=state,
            metrics=simulation.field.metrics(step=simulation.step_index),
            local_synchrony=simulation.field.local_synchrony(),
        )
        metrics = tracker.update(frame, regions, input_value=input_value)
        rows.append(audio_input_row(features, metrics, step=step))

    summary = summarize_audio_input_rows(rows, config=config)
    write_audio_input_rows(config.output_csv, rows)
    write_audio_input_summary(config.output_summary, summary)
    return summary


def audio_input_row(
    features: AudioInputFeatures,
    metrics: RegionalActivityMetrics,
    *,
    step: int,
) -> dict[str, Any]:
    return {
        "step": metrics.step,
        "audio_step": step,
        "time_seconds": step * features.hop_size / features.sample_rate,
        "rms": float(features.rms[step]),
        "onset": float(features.onset[step]),
        "spectral_centroid": float(features.spectral_centroid[step]),
        "input_value": metrics.input_value,
        "input_activity": metrics.input_activity,
        "input_fast_activity": metrics.input_fast_activity,
        "input_slow_activity": metrics.input_slow_activity,
        "assoc_activity": metrics.assoc_activity,
        "assoc_fast_activity": metrics.assoc_fast_activity,
        "assoc_slow_activity": metrics.assoc_slow_activity,
        "output_activity": metrics.output_activity,
        "output_fast_activity": metrics.output_fast_activity,
        "output_slow_activity": metrics.output_slow_activity,
        "left_to_right_ratio": metrics.left_to_right_ratio,
        "output_event_score": metrics.output_event_score,
        "output_fast_response_score": metrics.output_fast_response_score,
        "output_slow_drift_score": metrics.output_slow_drift_score,
    }


def summarize_audio_input_rows(
    rows: AudioInputRows,
    *,
    config: AudioInputRunConfig,
) -> AudioInputSummary:
    if not rows:
        msg = "audio input rows must not be empty"
        raise ValueError(msg)

    input_value = column(rows, "input_value")
    output_activity = column(rows, "output_activity")
    output_fast_score = column(rows, "output_fast_response_score")
    output_slow_score = column(rows, "output_slow_drift_score")
    peak_drive_index = int(np.argmax(input_value))
    peak_fast_index = int(np.argmax(output_fast_score))
    return {
        "config": str(config.config_path),
        "input_wav": str(config.input_wav),
        "parameters": {
            "frame_size": config.frame_size,
            "hop_size": config.hop_size,
            "drive_strength": config.drive_strength,
            "warmup_steps": config.warmup_steps,
            "max_steps": config.max_steps,
        },
        "rows": len(rows),
        "duration_seconds": float(rows[-1]["time_seconds"]),
        "peak_input_value": float(input_value[peak_drive_index]),
        "peak_input_step": int(rows[peak_drive_index]["audio_step"]),
        "peak_output_activity": float(np.max(output_activity)),
        "peak_output_fast_response_score": float(output_fast_score[peak_fast_index]),
        "peak_output_fast_response_step": int(rows[peak_fast_index]["audio_step"]),
        "peak_output_slow_drift_score": float(np.max(output_slow_score)),
        "mean_left_to_right_ratio": float(np.mean(column(rows, "left_to_right_ratio"))),
        "input_output_lag_steps": int(
            rows[peak_fast_index]["audio_step"] - rows[peak_drive_index]["audio_step"]
        ),
    }


def write_audio_input_rows(path: str | Path, rows: AudioInputRows) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output


def write_audio_input_summary(path: str | Path, summary: AudioInputSummary) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return output


def column(rows: AudioInputRows, key: str) -> np.ndarray:
    return np.asarray([row[key] for row in rows], dtype=np.float64)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the field with an offline WAV-derived input drive.",
    )
    parser.add_argument("--config", type=Path, default=AudioInputRunConfig.config_path)
    parser.add_argument("--input", type=Path, required=True, help="Input WAV path.")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=AudioInputRunConfig.output_csv,
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=AudioInputRunConfig.output_summary,
    )
    parser.add_argument("--frame-size", type=int, default=1024)
    parser.add_argument("--hop-size", type=int, default=512)
    parser.add_argument("--drive-strength", type=float, default=0.45)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = AudioInputRunConfig(
        config_path=args.config,
        input_wav=args.input,
        output_csv=args.output_csv,
        output_summary=args.output_summary,
        frame_size=args.frame_size,
        hop_size=args.hop_size,
        drive_strength=args.drive_strength,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
    )
    summary = run_audio_input_simulation(config)
    print(
        "Audio input run: "
        f"rows={summary['rows']} "
        f"peak_input={summary['peak_input_value']:.6f} "
        f"peak_fast_response={summary['peak_output_fast_response_score']:.6f} "
        f"lag={summary['input_output_lag_steps']}"
    )
    print(f"Wrote rows: {config.output_csv}")
    print(f"Wrote summary: {config.output_summary}")
    return 0
