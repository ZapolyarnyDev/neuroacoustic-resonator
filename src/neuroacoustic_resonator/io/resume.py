from __future__ import annotations

from pathlib import Path

from neuroacoustic_resonator.io.persistence import (
    CheckpointPaths,
    load_simulation_checkpoint,
    save_simulation_checkpoint,
)


def resume_simulation_checkpoint(
    checkpoint: str | Path,
    *,
    steps: int,
    output: str | Path | None = None,
) -> CheckpointPaths:
    if steps < 1:
        msg = "steps must be positive"
        raise ValueError(msg)

    checkpoint_path = Path(checkpoint)
    simulation = load_simulation_checkpoint(checkpoint_path)
    start_step = simulation.step_index
    simulation.run(steps)
    output_path = (
        Path(output)
        if output is not None
        else checkpoint_path.with_name(
            f"{checkpoint_path.stem}-resumed-{simulation.step_index}.npz"
        )
    )
    return save_simulation_checkpoint(
        output_path,
        simulation,
        metadata={
            "resumed_from": str(checkpoint_path),
            "start_step": start_step,
            "resumed_steps": steps,
        },
    )
