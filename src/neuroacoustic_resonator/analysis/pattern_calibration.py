from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from scipy.io import wavfile  # type: ignore[import-untyped]

from neuroacoustic_resonator.analysis.reinforcement import (
    compute_pattern_reinforcement_signals,
)
from neuroacoustic_resonator.audio.conversation import (
    VoiceConversationConfig,
    render_voice_conversation,
)

SyntheticKind = Literal["tone", "pulse", "chirp", "noise", "silence"]
CalibrationRows = list[dict[str, Any]]
CalibrationSummary = dict[str, Any]


@dataclass(frozen=True)
class SyntheticStimulusSpec:
    label: str
    kind: SyntheticKind
    frequency_hz: float = 220.0
    end_frequency_hz: float = 660.0
    duration_seconds: float = 0.5
    amplitude: float = 0.65
    sample_rate: int = 8_000

    def __post_init__(self) -> None:
        if not self.label:
            msg = "label must not be empty"
            raise ValueError(msg)
        if self.frequency_hz <= 0.0:
            msg = "frequency_hz must be positive"
            raise ValueError(msg)
        if self.end_frequency_hz <= 0.0:
            msg = "end_frequency_hz must be positive"
            raise ValueError(msg)
        if self.duration_seconds <= 0.0:
            msg = "duration_seconds must be positive"
            raise ValueError(msg)
        if self.amplitude < 0.0:
            msg = "amplitude must be non-negative"
            raise ValueError(msg)
        if self.sample_rate < 1:
            msg = "sample_rate must be positive"
            raise ValueError(msg)


@dataclass(frozen=True)
class CalibrationStimulus:
    label: str
    wav_path: Path
    source_type: str = "wav"

    def __post_init__(self) -> None:
        if not self.label:
            msg = "label must not be empty"
            raise ValueError(msg)


@dataclass(frozen=True)
class PatternCalibrationConfig:
    config_path: Path = Path("configs") / "field_only.yaml"
    stimuli: tuple[CalibrationStimulus, ...] = ()
    synthetic_stimuli: tuple[SyntheticStimulusSpec, ...] = ()
    output_dir: Path = Path("experiments") / "pattern_calibration"
    output_csv: Path = Path("experiments") / "logs" / "pattern_calibration.csv"
    output_summary: Path = (
        Path("experiments") / "logs" / "pattern_calibration_summary.json"
    )
    repeats: int = 1
    sample_rate: int = 8_000
    output_frame_size: int = 256
    input_frame_size: int = 256
    input_hop_size: int = 128
    drive_strength: float = 0.45
    input_assoc_gain: float = 0.8
    input_output_gain: float = 0.0
    response_seconds: float = 0.35
    warmup_steps: int = 16
    gain: float = 0.35

    def __post_init__(self) -> None:
        if not self.stimuli and not self.synthetic_stimuli:
            msg = "at least one stimulus is required"
            raise ValueError(msg)
        if self.repeats < 1:
            msg = "repeats must be positive"
            raise ValueError(msg)
        if self.sample_rate < 1:
            msg = "sample_rate must be positive"
            raise ValueError(msg)
        if self.output_frame_size < 1:
            msg = "output_frame_size must be positive"
            raise ValueError(msg)
        if self.input_frame_size < 1:
            msg = "input_frame_size must be positive"
            raise ValueError(msg)
        if self.input_hop_size < 1:
            msg = "input_hop_size must be positive"
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
        if self.response_seconds <= 0.0:
            msg = "response_seconds must be positive"
            raise ValueError(msg)
        if self.warmup_steps < 0:
            msg = "warmup_steps must be non-negative"
            raise ValueError(msg)
        if self.gain < 0.0:
            msg = "gain must be non-negative"
            raise ValueError(msg)


def run_pattern_calibration(config: PatternCalibrationConfig) -> CalibrationSummary:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    stimuli = list(config.stimuli) + materialize_synthetic_stimuli(config)
    rows: CalibrationRows = []
    trials: list[dict[str, Any]] = []
    for stimulus in stimuli:
        for repeat_index in range(1, config.repeats + 1):
            trial_summary = run_calibration_trial(config, stimulus, repeat_index)
            row = calibration_row(stimulus, repeat_index, trial_summary)
            rows.append(row)
            trials.append(
                {
                    "stimulus_label": stimulus.label,
                    "repeat_index": repeat_index,
                    "summary": trial_summary,
                }
            )

    reinforcement = compute_pattern_reinforcement_signals(rows)
    summary: CalibrationSummary = {
        "config": str(config.config_path),
        "parameters": calibration_parameters(config),
        "rows": len(rows),
        "stimuli": summarize_stimuli(rows),
        "reinforcement": reinforcement.to_dict(),
        "trials": trials,
    }
    write_calibration_rows(config.output_csv, rows)
    write_calibration_summary(config.output_summary, summary)
    return summary


def run_calibration_trial(
    config: PatternCalibrationConfig,
    stimulus: CalibrationStimulus,
    repeat_index: int,
) -> dict[str, Any]:
    trial_stem = f"{safe_name(stimulus.label)}-r{repeat_index:02d}"
    output_wav = config.output_dir / f"{trial_stem}-response.wav"
    output_summary = config.output_dir / f"{trial_stem}-summary.json"
    return render_voice_conversation(
        VoiceConversationConfig(
            config_path=config.config_path,
            input_wavs=(stimulus.wav_path,),
            output_wav=output_wav,
            output_summary=output_summary,
            sample_rate=config.sample_rate,
            output_frame_size=config.output_frame_size,
            input_frame_size=config.input_frame_size,
            input_hop_size=config.input_hop_size,
            drive_strength=config.drive_strength,
            input_assoc_gain=config.input_assoc_gain,
            input_output_gain=config.input_output_gain,
            response_seconds=config.response_seconds,
            pause_seconds=0.0,
            warmup_steps=config.warmup_steps,
            gain=config.gain,
            include_input_audio=False,
            use_response_policy=False,
        )
    )


def calibration_row(
    stimulus: CalibrationStimulus,
    repeat_index: int,
    trial_summary: dict[str, Any],
) -> dict[str, Any]:
    utterance = trial_summary["utterances"][0]
    input_history = utterance["input_output_pattern_history"]
    response_history = utterance["response_output_pattern_history"]
    audio = utterance["response_pattern_audio_diagnostics"]["overall"]
    return {
        "stimulus_label": stimulus.label,
        "source_type": stimulus.source_type,
        "input_wav": str(stimulus.wav_path),
        "repeat_index": repeat_index,
        "input_frames": utterance["input_frames"],
        "input_duration_seconds": utterance["input_duration_seconds"],
        "peak_input_value": utterance["peak_input_value"],
        "mean_input_value": utterance["mean_input_value"],
        "peak_input_fast_response_score": utterance["peak_input_fast_response_score"],
        "peak_input_event_score": utterance["peak_input_event_score"],
        "peak_response_score": utterance["peak_response_score"],
        "mean_response_score": utterance["mean_response_score"],
        "input_active_pattern_label": input_history["active_dominant_label"],
        "input_pattern_confidence": input_history["mean_confidence"],
        "response_active_pattern_label": response_history["active_dominant_label"],
        "response_pattern_confidence": response_history["mean_confidence"],
        "response_pattern_frames": response_history["frames"],
        "response_active_pattern_frames": response_history["active_frames"],
        "response_peak_activation": response_history["peak_activation"],
        "response_peak_activation_label": response_history["peak_activation_label"],
        "response_audio_rms": audio["rms"],
        "response_audio_peak": audio["peak"],
        "response_audio_zero_crossing_rate": audio["zero_crossing_rate"],
        "response_audio_spectral_centroid_hz": audio["spectral_centroid_hz"],
        "response_wav": trial_summary["output_wav"],
    }


def materialize_synthetic_stimuli(
    config: PatternCalibrationConfig,
) -> list[CalibrationStimulus]:
    generated_dir = config.output_dir / "generated_inputs"
    generated_dir.mkdir(parents=True, exist_ok=True)
    stimuli: list[CalibrationStimulus] = []
    for spec in config.synthetic_stimuli:
        path = generated_dir / f"{safe_name(spec.label)}.wav"
        write_synthetic_stimulus(path, spec)
        stimuli.append(CalibrationStimulus(spec.label, path, source_type=spec.kind))
    return stimuli


def write_synthetic_stimulus(path: Path, spec: SyntheticStimulusSpec) -> Path:
    audio = synthetic_stimulus_audio(spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(path, spec.sample_rate, audio.astype(np.float32))
    return path


def synthetic_stimulus_audio(spec: SyntheticStimulusSpec) -> np.ndarray:
    sample_count = max(1, int(round(spec.duration_seconds * spec.sample_rate)))
    time = np.arange(sample_count, dtype=np.float64) / spec.sample_rate
    envelope = raised_cosine_envelope(sample_count)
    if spec.kind == "tone":
        audio = np.sin(2.0 * np.pi * spec.frequency_hz * time)
    elif spec.kind == "pulse":
        carrier = np.sin(2.0 * np.pi * spec.frequency_hz * time)
        gate = (np.sin(2.0 * np.pi * 7.0 * time) > 0.35).astype(np.float64)
        audio = carrier * gate
    elif spec.kind == "chirp":
        sweep = np.linspace(spec.frequency_hz, spec.end_frequency_hz, sample_count)
        phase = 2.0 * np.pi * np.cumsum(sweep) / spec.sample_rate
        audio = np.sin(phase)
    elif spec.kind == "noise":
        rng = np.random.default_rng(seed=stable_seed(spec.label))
        audio = rng.normal(0.0, 1.0, sample_count)
    elif spec.kind == "silence":
        audio = np.zeros(sample_count, dtype=np.float64)
    else:
        msg = f"unsupported synthetic stimulus kind: {spec.kind}"
        raise ValueError(msg)
    return np.clip(spec.amplitude * envelope * audio, -1.0, 1.0)


def raised_cosine_envelope(sample_count: int) -> np.ndarray:
    if sample_count < 4:
        return np.ones(sample_count, dtype=np.float64)
    edge = max(1, sample_count // 20)
    envelope = np.ones(sample_count, dtype=np.float64)
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, edge))
    envelope[:edge] = ramp
    envelope[-edge:] = ramp[::-1]
    return envelope


def summarize_stimuli(rows: CalibrationRows) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["stimulus_label"]), []).append(row)
    return {
        label: {
            "trials": len(group),
            "mean_peak_response_score": float(
                np.mean([trial["peak_response_score"] for trial in group])
            ),
            "mean_response_audio_rms": float(
                np.mean([trial["response_audio_rms"] for trial in group])
            ),
            "response_pattern_labels": sorted(
                {str(trial["response_active_pattern_label"]) for trial in group}
            ),
        }
        for label, group in sorted(grouped.items())
    }


def calibration_parameters(config: PatternCalibrationConfig) -> dict[str, Any]:
    return {
        "repeats": config.repeats,
        "sample_rate": config.sample_rate,
        "output_frame_size": config.output_frame_size,
        "input_frame_size": config.input_frame_size,
        "input_hop_size": config.input_hop_size,
        "drive_strength": config.drive_strength,
        "input_assoc_gain": config.input_assoc_gain,
        "input_output_gain": config.input_output_gain,
        "response_seconds": config.response_seconds,
        "warmup_steps": config.warmup_steps,
        "gain": config.gain,
    }


def write_calibration_rows(path: str | Path, rows: CalibrationRows) -> Path:
    if not rows:
        msg = "calibration rows must not be empty"
        raise ValueError(msg)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output


def write_calibration_summary(path: str | Path, summary: CalibrationSummary) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return output


def stable_seed(value: str) -> int:
    return sum((index + 1) * ord(char) for index, char in enumerate(value)) % (2**32)


def safe_name(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "-_" else "-" for char in value)
    return cleaned.strip("-") or "stimulus"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run pattern calibration across WAV and synthetic stimuli.",
    )
    parser.add_argument(
        "--config", type=Path, default=PatternCalibrationConfig.config_path
    )
    parser.add_argument("--input", type=Path, action="append", default=[])
    parser.add_argument("--synthetic", action="append", default=[])
    parser.add_argument(
        "--output-dir", type=Path, default=PatternCalibrationConfig.output_dir
    )
    parser.add_argument(
        "--output-csv", type=Path, default=PatternCalibrationConfig.output_csv
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=PatternCalibrationConfig.output_summary,
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--sample-rate", type=int, default=8_000)
    parser.add_argument("--output-frame-size", type=int, default=256)
    parser.add_argument("--input-frame-size", type=int, default=256)
    parser.add_argument("--input-hop-size", type=int, default=128)
    parser.add_argument("--drive-strength", type=float, default=0.45)
    parser.add_argument("--input-assoc-gain", type=float, default=0.8)
    parser.add_argument("--input-output-gain", type=float, default=0.0)
    parser.add_argument("--response-seconds", type=float, default=0.35)
    parser.add_argument("--warmup-steps", type=int, default=16)
    return parser


def parse_synthetic(value: str, *, sample_rate: int) -> SyntheticStimulusSpec:
    parts = value.split(":")
    if len(parts) < 2:
        msg = "synthetic stimulus must be label:kind[:frequency][:duration]"
        raise ValueError(msg)
    label = parts[0]
    kind = parts[1]
    frequency = float(parts[2]) if len(parts) >= 3 else 220.0
    duration = float(parts[3]) if len(parts) >= 4 else 0.5
    return SyntheticStimulusSpec(
        label=label,
        kind=kind,  # type: ignore[arg-type]
        frequency_hz=frequency,
        duration_seconds=duration,
        sample_rate=sample_rate,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    stimuli = tuple(
        CalibrationStimulus(path.stem, path, source_type="wav") for path in args.input
    )
    synthetic = tuple(
        parse_synthetic(value, sample_rate=args.sample_rate) for value in args.synthetic
    )
    summary = run_pattern_calibration(
        PatternCalibrationConfig(
            config_path=args.config,
            stimuli=stimuli,
            synthetic_stimuli=synthetic,
            output_dir=args.output_dir,
            output_csv=args.output_csv,
            output_summary=args.output_summary,
            repeats=args.repeats,
            sample_rate=args.sample_rate,
            output_frame_size=args.output_frame_size,
            input_frame_size=args.input_frame_size,
            input_hop_size=args.input_hop_size,
            drive_strength=args.drive_strength,
            input_assoc_gain=args.input_assoc_gain,
            input_output_gain=args.input_output_gain,
            response_seconds=args.response_seconds,
            warmup_steps=args.warmup_steps,
        )
    )
    reinforcement = summary["reinforcement"]
    print(
        "Pattern calibration: "
        f"rows={summary['rows']} "
        f"reward={reinforcement['reward']:.3f} "
        f"stability={reinforcement['stability']:.3f} "
        f"diversity={reinforcement['diversity']:.3f}"
    )
    print(f"Wrote rows: {args.output_csv}")
    print(f"Wrote summary: {args.output_summary}")
    return 0
