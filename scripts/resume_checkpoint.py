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

    from neuroacoustic_resonator.io.persistence import (
        load_simulation_checkpoint,
    )
    from neuroacoustic_resonator.io.resume import resume_simulation_checkpoint

    parser = argparse.ArgumentParser(
        description="Resume a saved simulation checkpoint and write a new checkpoint.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output checkpoint path. Defaults to <checkpoint-stem>-resumed-<step>.npz.",
    )
    args = parser.parse_args()

    simulation = load_simulation_checkpoint(args.checkpoint)
    start_step = simulation.step_index
    paths = resume_simulation_checkpoint(
        args.checkpoint,
        steps=args.steps,
        output=args.output,
    )
    final_step = start_step + args.steps
    print(
        f"Resumed checkpoint: {args.checkpoint} "
        f"start_step={start_step} final_step={final_step}"
    )
    print(f"Saved checkpoint: {paths.arrays_path} metadata={paths.metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
