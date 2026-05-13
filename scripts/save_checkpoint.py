from __future__ import annotations

import os
import sys
from pathlib import Path


def _bootstrap() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    if os.environ.get("NEUROACOUSTIC_BOOTSTRAPPED") != "1":
        candidates = (
            repo_root / ".venv" / "Scripts" / "python.exe",
            repo_root / ".venv" / "bin" / "python",
        )
        current_python = Path(sys.executable).resolve()
        for candidate in candidates:
            if candidate.exists() and candidate.resolve() != current_python:
                os.environ["NEUROACOUSTIC_BOOTSTRAPPED"] = "1"
                os.execv(str(candidate), [str(candidate), *sys.argv])

    src_path = str(repo_root / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


def _main() -> int:
    _bootstrap()

    import argparse

    from neuroacoustic_resonator.core.config import SimulationConfig
    from neuroacoustic_resonator.io.persistence import save_simulation_checkpoint
    from neuroacoustic_resonator.core.simulation import Simulation

    parser = argparse.ArgumentParser(
        description="Run a simulation and save a checkpoint as NPZ + YAML.",
    )
    parser.add_argument("--config", type=Path, default=Path("configs") / "default.yaml")
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments") / "states" / "checkpoint.npz",
    )
    args = parser.parse_args()

    if args.steps < 0:
        msg = "steps must be non-negative"
        raise ValueError(msg)

    config = SimulationConfig.from_file(args.config)
    simulation = Simulation.from_config(config)
    simulation.run(args.steps)
    paths = save_simulation_checkpoint(
        args.output,
        simulation,
        metadata={"config_path": str(args.config)},
    )
    print(
        f"Saved checkpoint: {paths.arrays_path} "
        f"metadata={paths.metadata_path} step={simulation.step_index}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
