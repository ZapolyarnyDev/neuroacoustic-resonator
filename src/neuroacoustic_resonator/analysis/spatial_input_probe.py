from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from math import ceil
from pathlib import Path
from typing import Any

import numpy as np

from neuroacoustic_resonator.audio.input import (
    ChannelAudioFeatures,
    extract_channel_audio_features,
)
from neuroacoustic_resonator.audio.spatial_input import (
    SpatialAudioDrive,
    SpatialInputConfig,
)
from neuroacoustic_resonator.configuration import SimulationConfig
from neuroacoustic_resonator.core.field import FieldConfig
from neuroacoustic_resonator.core.regions import RegionMasks
from neuroacoustic_resonator.core.simulation import Simulation


def run_spatial_input_probe(
    wav_path: str | Path,
    config_path: str | Path,
    output_path: str | Path,
    *,
    seed: int,
    response_seconds: float = 0.35,
    frame_size: int = 256,
    hop_size: int = 128,
    bands: int = 8,
    source_positions: tuple[float, ...] | None = None,
) -> dict[str, Any]:
    if response_seconds < 0.0:
        msg = "response_seconds must be non-negative"
        raise ValueError(msg)
    if seed < 0:
        msg = "seed must be non-negative"
        raise ValueError(msg)
    features = extract_channel_audio_features(
        wav_path, frame_size=frame_size, hop_size=hop_size, bands=bands
    )
    if features.sample_count == 0:
        msg = "spatial probe requires non-empty audio"
        raise ValueError(msg)
    positions = (
        source_positions
        if source_positions is not None
        else SpatialInputConfig.canonical(features.channel_count).source_positions
    )
    input_config = SpatialInputConfig(source_positions=positions)
    field_config = replace(
        SimulationConfig.from_file(config_path).field.to_runtime(), seed=seed
    )
    silent_features = replace(features, band_energy=np.zeros_like(features.band_energy))
    steps = ceil((features.duration_seconds + response_seconds) / field_config.dt)
    active = _branch(features, field_config, input_config, steps)
    silent = _branch(silent_features, field_config, input_config, steps)
    uncoupled_field = replace(
        field_config, coupling_strength=0.0, coupling_homeostasis_rate=0.0
    )
    uncoupled_active = _branch(features, uncoupled_field, input_config, steps)
    uncoupled_silent = _branch(silent_features, uncoupled_field, input_config, steps)
    coupled_delta = _phase_rms(active["output_phase"], silent["output_phase"])
    uncoupled_delta = _phase_rms(
        uncoupled_active["output_phase"], uncoupled_silent["output_phase"]
    )
    transport_config = asdict(
        SpatialAudioDrive(
            features, Simulation(field_config), input_config
        ).transport.config
    )
    transport_config["source_positions"] = list(positions)
    report: dict[str, Any] = {
        "status": "development_probe_not_stage_one_evidence",
        "input_wav": str(wav_path),
        "simulation_config": str(config_path),
        "field_seed": seed,
        "field_dt": field_config.dt,
        "input_frame_size": frame_size,
        "input_hop_size": hop_size,
        "bands": bands,
        "input_duration_seconds": features.duration_seconds,
        "response_seconds": response_seconds,
        "steps": steps,
        "source_positions": list(positions),
        "transport_config": transport_config,
        "coupled_output_phase_rms_delta": coupled_delta,
        "uncoupled_output_phase_rms_delta": uncoupled_delta,
        "transport_output_mass": active["transport_output_mass"],
        "coupled_output_local_synchrony": active["output_local_synchrony"],
        "silence_output_local_synchrony": silent["output_local_synchrony"],
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _branch(
    features: ChannelAudioFeatures,
    field_config: FieldConfig,
    input_config: SpatialInputConfig,
    steps: int,
) -> dict[str, Any]:
    simulation = Simulation(field_config)
    drive = SpatialAudioDrive(features, simulation, input_config)
    regions = RegionMasks.from_size(field_config.size)
    output_phase: list[np.ndarray] = []
    output_local_synchrony: list[float] = []
    transport_output_mass: list[float] = []
    for index in range(steps):
        frame = drive.step(simulation, index * field_config.dt)
        output_phase.append(frame.state.phase[regions.output].copy())
        output_local_synchrony.append(
            float(np.mean(frame.local_synchrony[regions.output]))
        )
        transport_output_mass.append(
            float(np.sum(drive.transport.state[:, regions.output]))
        )
    return {
        "output_phase": np.stack(output_phase),
        "output_local_synchrony": output_local_synchrony,
        "transport_output_mass": transport_output_mass,
    }


def _phase_rms(active: np.ndarray, silent: np.ndarray) -> list[float]:
    circular_delta = np.angle(np.exp(1j * (active - silent)))
    return [float(value) for value in np.sqrt(np.mean(circular_delta**2, axis=1))]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the spatial input development probe."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/field_only.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=67)
    parser.add_argument("--response-seconds", type=float, default=0.35)
    parser.add_argument("--source-positions", type=float, nargs="+")
    args = parser.parse_args(argv)
    run_spatial_input_probe(
        args.input,
        args.config,
        args.output,
        seed=args.seed,
        response_seconds=args.response_seconds,
        source_positions=(
            tuple(args.source_positions) if args.source_positions is not None else None
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
