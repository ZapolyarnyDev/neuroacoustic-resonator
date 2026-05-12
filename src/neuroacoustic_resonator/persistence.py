from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from neuroacoustic_resonator.field import FieldConfig, FieldState, OscillatorField
from neuroacoustic_resonator.input_drive import SyntheticInputConfig
from neuroacoustic_resonator.simulation import Simulation

CHECKPOINT_VERSION = 1


@dataclass(frozen=True)
class CheckpointPaths:
    arrays_path: Path
    metadata_path: Path


def checkpoint_paths(path: str | Path) -> CheckpointPaths:
    arrays_path = Path(path)
    if arrays_path.suffix != ".npz":
        arrays_path = arrays_path.with_suffix(".npz")
    return CheckpointPaths(
        arrays_path=arrays_path,
        metadata_path=arrays_path.with_suffix(".yaml"),
    )


def save_field_state(
    path: str | Path,
    state: FieldState,
    *,
    metadata: dict[str, Any] | None = None,
) -> CheckpointPaths:
    paths = checkpoint_paths(path)
    paths.arrays_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        paths.arrays_path,
        phase=state.phase,
        frequency=state.frequency,
        metabolite=state.metabolite,
        coupling=state.coupling,
        trace=state.trace,
    )
    with paths.metadata_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(metadata or {}, stream, sort_keys=True)
    return paths


def load_field_state(path: str | Path) -> FieldState:
    paths = checkpoint_paths(path)
    with np.load(paths.arrays_path) as data:
        required = ("phase", "frequency", "metabolite", "coupling", "trace")
        missing = [name for name in required if name not in data]
        if missing:
            msg = f"checkpoint is missing arrays: {', '.join(missing)}"
            raise ValueError(msg)
        state = FieldState(
            phase=np.asarray(data["phase"], dtype=np.float64),
            frequency=np.asarray(data["frequency"], dtype=np.float64),
            metabolite=np.asarray(data["metabolite"], dtype=np.float64),
            coupling=np.asarray(data["coupling"], dtype=np.float64),
            trace=np.asarray(data["trace"], dtype=np.float64),
        )
    _validate_state_shapes(state)
    return state


def load_checkpoint_metadata(path: str | Path) -> dict[str, Any]:
    paths = checkpoint_paths(path)
    with paths.metadata_path.open("r", encoding="utf-8") as stream:
        metadata = yaml.safe_load(stream) or {}
    if not isinstance(metadata, dict):
        msg = "checkpoint metadata must be a mapping"
        raise ValueError(msg)
    return metadata


def save_simulation_checkpoint(
    path: str | Path,
    simulation: Simulation,
    *,
    metadata: dict[str, Any] | None = None,
) -> CheckpointPaths:
    checkpoint_metadata: dict[str, Any] = {
        "version": CHECKPOINT_VERSION,
        "step_index": simulation.step_index,
        "field_config": asdict(simulation.field.config),
        "synthetic_input": asdict(simulation.input_drive.config),
    }
    if metadata:
        checkpoint_metadata["metadata"] = metadata
    return save_field_state(path, simulation.field.state, metadata=checkpoint_metadata)


def load_simulation_checkpoint(path: str | Path) -> Simulation:
    metadata = load_checkpoint_metadata(path)
    version = metadata.get("version")
    if version != CHECKPOINT_VERSION:
        msg = f"unsupported checkpoint version: {version}"
        raise ValueError(msg)

    state = load_field_state(path)
    field_config_data = metadata.get("field_config")
    if not isinstance(field_config_data, dict):
        msg = "checkpoint metadata must contain field_config"
        raise ValueError(msg)

    synthetic_input_data = metadata.get("synthetic_input", {})
    if not isinstance(synthetic_input_data, dict):
        msg = "checkpoint metadata synthetic_input must be a mapping"
        raise ValueError(msg)

    field = OscillatorField.from_state(FieldConfig(**field_config_data), state)
    simulation = Simulation(
        field=field,
        synthetic_input=SyntheticInputConfig(**synthetic_input_data),
    )
    simulation.step_index = int(metadata.get("step_index", 0))
    return simulation


def _validate_state_shapes(state: FieldState) -> None:
    shape = state.phase.shape
    for name, value in (
        ("frequency", state.frequency),
        ("metabolite", state.metabolite),
        ("coupling", state.coupling),
        ("trace", state.trace),
    ):
        if value.shape != shape:
            msg = f"{name} shape must match phase shape"
            raise ValueError(msg)
