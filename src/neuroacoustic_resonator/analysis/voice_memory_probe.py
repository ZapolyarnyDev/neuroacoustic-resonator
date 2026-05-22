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

VoiceMemoryRows = list[dict[str, Any]]
VoiceMemorySummary = dict[str, Any]


@dataclass(frozen=True)
class VoiceMemoryProbeConfig:
    config_path: Path = Path("configs") / "field_only.yaml"
    input_wav: Path = Path("experiments") / "audio" / "my_voice.wav"
    output_csv: Path = Path("experiments") / "logs" / "voice_memory_probe.csv"
    output_summary: Path = (
        Path("experiments") / "logs" / "voice_memory_probe_summary.json"
    )
    frame_size: int = 1024
    hop_size: int = 512
    drive_strength: float = 0.45
    input_assoc_gain: float = 0.8
    input_output_gain: float = 0.0
    warmup_steps: int = 100
    pause_steps: int = 128
    max_steps: int | None = None
    compare_memory_drive_strength: float | None = None
    compare_silence_control: bool = False

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
        if self.input_assoc_gain < 0.0:
            msg = "input_assoc_gain must be non-negative"
            raise ValueError(msg)
        if self.input_output_gain < 0.0:
            msg = "input_output_gain must be non-negative"
            raise ValueError(msg)
        if self.warmup_steps < 0:
            msg = "warmup_steps must be non-negative"
            raise ValueError(msg)
        if self.pause_steps < 0:
            msg = "pause_steps must be non-negative"
            raise ValueError(msg)
        if self.max_steps is not None and self.max_steps < 1:
            msg = "max_steps must be positive"
            raise ValueError(msg)
        if (
            self.compare_memory_drive_strength is not None
            and self.compare_memory_drive_strength < 0.0
        ):
            msg = "compare_memory_drive_strength must be non-negative"
            raise ValueError(msg)


def run_voice_memory_probe(config: VoiceMemoryProbeConfig) -> VoiceMemorySummary:
    sim_config = SimulationConfig.from_file(config.config_path)
    features = extract_audio_input_features(
        config.input_wav,
        frame_size=config.frame_size,
        hop_size=config.hop_size,
        drive_strength=config.drive_strength,
    )
    rows = collect_voice_memory_rows(sim_config, config, features=features)
    summary = summarize_voice_memory_rows(rows, config=config)
    if (
        config.compare_memory_drive_strength is None
        and not config.compare_silence_control
    ):
        write_voice_memory_rows(config.output_csv, rows)
        write_voice_memory_summary(config.output_summary, summary)
        return summary

    labeled_rows = label_rows(rows, "baseline")
    comparison_summary: VoiceMemorySummary = {
        "config": str(config.config_path),
        "input_wav": str(config.input_wav),
        "parameters": summary["parameters"],
        "baseline": summary,
    }

    if config.compare_silence_control:
        silence_rows = collect_voice_memory_rows(
            sim_config,
            config,
            features=silence_features_like(features),
        )
        silence_summary = summarize_voice_memory_rows(silence_rows, config=config)
        labeled_rows += label_rows(silence_rows, "silence_control")
        comparison_summary["silence_control"] = silence_summary
        comparison_summary["voice_vs_silence_control"] = compare_probe_summaries(
            summary,
            silence_summary,
            right_label="silence_control",
        )

    if config.compare_memory_drive_strength is None:
        write_voice_memory_rows(config.output_csv, labeled_rows)
        write_voice_memory_summary(config.output_summary, comparison_summary)
        return comparison_summary

    memory_sim_config = sim_config.model_copy(
        update={
            "field": sim_config.field.model_copy(
                update={
                    "memory_drive_strength": config.compare_memory_drive_strength,
                }
            )
        }
    )
    memory_rows = collect_voice_memory_rows(
        memory_sim_config,
        config,
        features=features,
    )
    memory_summary = summarize_voice_memory_rows(memory_rows, config=config)
    labeled_rows += label_rows(memory_rows, "memory_drive")
    comparison_summary["memory_drive"] = memory_summary
    comparison_summary["memory_drive_comparison"] = compare_probe_summaries(
        summary,
        memory_summary,
        right_label="memory_drive",
    )
    write_voice_memory_rows(config.output_csv, labeled_rows)
    write_voice_memory_summary(config.output_summary, comparison_summary)
    return comparison_summary


def collect_voice_memory_rows(
    sim_config: SimulationConfig,
    config: VoiceMemoryProbeConfig,
    *,
    features: AudioInputFeatures,
) -> VoiceMemoryRows:
    simulation = Simulation.from_config(sim_config)
    regions = RegionMasks.from_size(sim_config.field.size)
    tracker = RegionalActivityTracker()
    drive = WavInputDrive(
        features,
        regions,
        assoc_gain=config.input_assoc_gain,
        output_gain=config.input_output_gain,
    )

    for _ in range(config.warmup_steps):
        frame = simulation.step()
        tracker.update(frame, regions, input_value=simulation.last_input_value)

    rows: VoiceMemoryRows = []
    rows.extend(
        run_voice_repeat(
            simulation,
            tracker,
            regions,
            features,
            drive,
            repeat_index=1,
            max_steps=config.max_steps,
        )
    )
    for _ in range(config.pause_steps):
        frame = simulation.step()
        tracker.update(frame, regions, input_value=simulation.last_input_value)
    rows.extend(
        run_voice_repeat(
            simulation,
            tracker,
            regions,
            features,
            drive,
            repeat_index=2,
            max_steps=config.max_steps,
        )
    )

    return rows


def label_rows(rows: VoiceMemoryRows, label: str) -> VoiceMemoryRows:
    return [{"probe_label": label, **row} for row in rows]


def silence_features_like(features: AudioInputFeatures) -> AudioInputFeatures:
    zeros = np.zeros(features.frame_count, dtype=np.float64)
    return AudioInputFeatures(
        sample_rate=features.sample_rate,
        frame_size=features.frame_size,
        hop_size=features.hop_size,
        rms=zeros.copy(),
        onset=zeros.copy(),
        spectral_centroid=zeros.copy(),
        drive=zeros.copy(),
    )


def run_voice_repeat(
    simulation: Simulation,
    tracker: RegionalActivityTracker,
    regions: RegionMasks,
    features: AudioInputFeatures,
    drive: WavInputDrive,
    *,
    repeat_index: int,
    max_steps: int | None,
) -> VoiceMemoryRows:
    steps = features.frame_count
    if max_steps is not None:
        steps = min(steps, max_steps)

    rows: VoiceMemoryRows = []
    for audio_step in range(steps):
        input_value = drive.apply(simulation.field, audio_step)
        simulation.step_index += 1
        state = simulation.field.step()
        simulation.last_input_value = input_value
        frame = SimulationFrame(
            state=state,
            metrics=simulation.field.metrics(step=simulation.step_index),
            local_synchrony=simulation.field.local_synchrony(),
        )
        metrics = tracker.update(frame, regions, input_value=input_value)
        rows.append(
            voice_memory_row(
                features,
                metrics,
                repeat_index=repeat_index,
                audio_step=audio_step,
            )
        )
    return rows


def voice_memory_row(
    features: AudioInputFeatures,
    metrics: RegionalActivityMetrics,
    *,
    repeat_index: int,
    audio_step: int,
) -> dict[str, Any]:
    return {
        "repeat_index": repeat_index,
        "global_step": metrics.step,
        "audio_step": audio_step,
        "time_seconds": audio_step * features.hop_size / features.sample_rate,
        "rms": float(features.rms[audio_step]),
        "onset": float(features.onset[audio_step]),
        "spectral_centroid": float(features.spectral_centroid[audio_step]),
        "input_value": metrics.input_value,
        "input_activity": metrics.input_activity,
        "assoc_activity": metrics.assoc_activity,
        "output_activity": metrics.output_activity,
        "output_activity_baseline": metrics.output_activity_baseline,
        "output_response_activity": metrics.output_response_activity,
        "output_fast_activity": metrics.output_fast_activity,
        "output_slow_activity": metrics.output_slow_activity,
        "left_to_right_ratio": metrics.left_to_right_ratio,
        "output_event_score": metrics.output_event_score,
        "output_fast_response_score": metrics.output_fast_response_score,
        "output_slow_drift_score": metrics.output_slow_drift_score,
    }


def summarize_voice_memory_rows(
    rows: VoiceMemoryRows,
    *,
    config: VoiceMemoryProbeConfig,
) -> VoiceMemorySummary:
    first = [row for row in rows if row["repeat_index"] == 1]
    second = [row for row in rows if row["repeat_index"] == 2]
    if not first or not second:
        msg = "voice memory rows must contain both repeats"
        raise ValueError(msg)

    return {
        "config": str(config.config_path),
        "input_wav": str(config.input_wav),
        "parameters": {
            "frame_size": config.frame_size,
            "hop_size": config.hop_size,
            "drive_strength": config.drive_strength,
            "input_assoc_gain": config.input_assoc_gain,
            "input_output_gain": config.input_output_gain,
            "warmup_steps": config.warmup_steps,
            "pause_steps": config.pause_steps,
            "max_steps": config.max_steps,
            "compare_memory_drive_strength": config.compare_memory_drive_strength,
            "compare_silence_control": config.compare_silence_control,
        },
        "rows": len(rows),
        "first": summarize_repeat(first),
        "second": summarize_repeat(second),
        "comparison": compare_repeats(first, second),
    }


def summarize_repeat(rows: VoiceMemoryRows) -> dict[str, float | int]:
    output_response = column(rows, "output_response_activity")
    fast = column(rows, "output_fast_response_score")
    event = column(rows, "output_event_score")
    drift = column(rows, "output_slow_drift_score")
    return {
        "rows": len(rows),
        "start_global_step": int(rows[0]["global_step"]),
        "end_global_step": int(rows[-1]["global_step"]),
        "peak_output_response_activity": float(np.max(output_response)),
        "mean_output_response_activity": float(np.mean(output_response)),
        "peak_output_fast_response_score": float(np.max(fast)),
        "mean_output_fast_response_score": float(np.mean(fast)),
        "peak_output_event_score": float(np.max(event)),
        "mean_output_event_score": float(np.mean(event)),
        "peak_output_slow_drift_score": float(np.max(drift)),
        "mean_output_slow_drift_score": float(np.mean(drift)),
    }


def compare_repeats(
    first: VoiceMemoryRows,
    second: VoiceMemoryRows,
) -> dict[str, float]:
    keys = (
        "output_response_activity",
        "output_fast_response_score",
        "output_event_score",
        "output_slow_drift_score",
    )
    comparison: dict[str, float] = {}
    length = min(len(first), len(second))
    for key in keys:
        first_values = column(first[:length], key)
        second_values = column(second[:length], key)
        comparison[f"{key}_corr"] = safe_correlation(first_values, second_values)
        comparison[f"{key}_mean_abs_delta"] = float(
            np.mean(np.abs(second_values - first_values))
        )
        comparison[f"{key}_second_to_first_peak"] = safe_ratio(
            float(np.max(second_values)),
            float(np.max(first_values)),
        )
    return comparison


def compare_probe_summaries(
    baseline: VoiceMemorySummary,
    compared: VoiceMemorySummary,
    *,
    right_label: str,
) -> dict[str, float]:
    baseline_comparison = baseline["comparison"]
    compared_comparison = compared["comparison"]
    keys = (
        "output_response_activity_mean_abs_delta",
        "output_fast_response_score_mean_abs_delta",
        "output_event_score_mean_abs_delta",
        "output_slow_drift_score_mean_abs_delta",
    )
    comparison: dict[str, float] = {}
    for key in keys:
        baseline_value = float(baseline_comparison[key])
        compared_value = float(compared_comparison[key])
        comparison[f"{key}_baseline"] = baseline_value
        comparison[f"{key}_{right_label}"] = compared_value
        comparison[f"{key}_{right_label}_to_baseline_ratio"] = safe_ratio(
            compared_value,
            baseline_value,
        )
    return comparison


def column(rows: VoiceMemoryRows, key: str) -> np.ndarray:
    return np.asarray([row[key] for row in rows], dtype=np.float64)


def safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.size != right.size:
        msg = "correlation arrays must have matching sizes"
        raise ValueError(msg)
    if left.size < 2:
        return 0.0
    left_std = float(np.std(left))
    right_std = float(np.std(right))
    if left_std == 0.0 or right_std == 0.0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / max(denominator, 1e-12))


def write_voice_memory_rows(path: str | Path, rows: VoiceMemoryRows) -> Path:
    if not rows:
        msg = "voice memory rows must not be empty"
        raise ValueError(msg)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output


def write_voice_memory_summary(
    path: str | Path,
    summary: VoiceMemorySummary,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Play the same WAV twice through one field and compare responses.",
    )
    parser.add_argument(
        "--config", type=Path, default=VoiceMemoryProbeConfig.config_path
    )
    parser.add_argument(
        "--input", type=Path, required=True, help="Input voice WAV path."
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=VoiceMemoryProbeConfig.output_csv,
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=VoiceMemoryProbeConfig.output_summary,
    )
    parser.add_argument("--frame-size", type=int, default=1024)
    parser.add_argument("--hop-size", type=int, default=512)
    parser.add_argument("--drive-strength", type=float, default=0.45)
    parser.add_argument("--input-assoc-gain", type=float, default=0.8)
    parser.add_argument("--input-output-gain", type=float, default=0.0)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--pause-steps", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--compare-memory-drive-strength", type=float, default=None)
    parser.add_argument("--compare-silence-control", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = VoiceMemoryProbeConfig(
        config_path=args.config,
        input_wav=args.input,
        output_csv=args.output_csv,
        output_summary=args.output_summary,
        frame_size=args.frame_size,
        hop_size=args.hop_size,
        drive_strength=args.drive_strength,
        input_assoc_gain=args.input_assoc_gain,
        input_output_gain=args.input_output_gain,
        warmup_steps=args.warmup_steps,
        pause_steps=args.pause_steps,
        max_steps=args.max_steps,
        compare_memory_drive_strength=args.compare_memory_drive_strength,
        compare_silence_control=args.compare_silence_control,
    )
    summary = run_voice_memory_probe(config)
    if "memory_drive_comparison" in summary:
        comparison = summary["memory_drive"]["comparison"]
        memory_comparison = summary["memory_drive_comparison"]
        print(
            "Voice memory probe: "
            f"memory_drive_strength={config.compare_memory_drive_strength:.3f} "
            "fast_delta_ratio="
            f"{memory_comparison['output_fast_response_score_mean_abs_delta_memory_drive_to_baseline_ratio']:.3f} "
            "event_delta_ratio="
            f"{memory_comparison['output_event_score_mean_abs_delta_memory_drive_to_baseline_ratio']:.3f}"
        )
    elif "voice_vs_silence_control" in summary:
        control = summary["voice_vs_silence_control"]
        print(
            "Voice memory probe: "
            "silence_control=true "
            "fast_delta_ratio="
            f"{control['output_fast_response_score_mean_abs_delta_silence_control_to_baseline_ratio']:.3f} "
            "event_delta_ratio="
            f"{control['output_event_score_mean_abs_delta_silence_control_to_baseline_ratio']:.3f}"
        )
    else:
        comparison = summary["comparison"]
        print(
            "Voice memory probe: "
            f"fast_corr={comparison['output_fast_response_score_corr']:.3f} "
            f"event_corr={comparison['output_event_score_corr']:.3f} "
            "fast_peak_ratio="
            f"{comparison['output_fast_response_score_second_to_first_peak']:.3f} "
            "response_peak_ratio="
            f"{comparison['output_response_activity_second_to_first_peak']:.3f}"
        )
    print(f"Wrote rows: {config.output_csv}")
    print(f"Wrote summary: {config.output_summary}")
    return 0
