from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from importlib import import_module
from pathlib import Path

DELEGATED: dict[str, tuple[str, str]] = {
    "benchmark": (
        "neuroacoustic_resonator.analysis.benchmark",
        "Benchmark field stepping.",
    ),
    "protocol": (
        "neuroacoustic_resonator.cli.protocol",
        "Record or replay a Sound Protocol stream.",
    ),
    "live": (
        "neuroacoustic_resonator.audio.live_conversation",
        "Turn-based microphone interaction.",
    ),
    "conversation": (
        "neuroacoustic_resonator.audio.conversation",
        "Run recorded WAV files through the field.",
    ),
    "causal": (
        "neuroacoustic_resonator.analysis.controlled_equilibration",
        "Paired stimulus/control corpus from equilibrated checkpoints.",
    ),
    "ablations": (
        "neuroacoustic_resonator.analysis.equilibration_ablations",
        "Equilibration variants of the paired corpus.",
    ),
    "parts": (
        "neuroacoustic_resonator.analysis.field_parts",
        "Paired corpus with each field component disabled.",
    ),
    "report": (
        "neuroacoustic_resonator.analysis.research_report",
        "Write the key report for the current commit.",
    ),
}


def preview(args: argparse.Namespace) -> int:
    from neuroacoustic_resonator.configuration import SimulationConfig
    from neuroacoustic_resonator.viz import save_field_preview

    config = SimulationConfig.from_file(args.config)
    simulation = config.create_simulation()
    frame = simulation.run(config.steps)[-1]
    preview_path = save_field_preview(frame, config.preview_path)
    print(
        f"Field preview: size={frame.state.phase.shape[0]}x"
        f"{frame.state.phase.shape[1]}, "
        f"mean_metabolite={frame.metrics.mean_metabolite:.3f}, "
        f"global_synchrony={frame.metrics.global_synchrony:.3f}, "
        f"preview={preview_path}"
    )
    return 0


def checkpoint(args: argparse.Namespace) -> int:
    from neuroacoustic_resonator.configuration import SimulationConfig
    from neuroacoustic_resonator.io.persistence import save_simulation_checkpoint

    if args.steps < 0:
        msg = "steps must be non-negative"
        raise ValueError(msg)
    config = SimulationConfig.from_file(args.config)
    simulation = config.create_simulation()
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


def resume(args: argparse.Namespace) -> int:
    from neuroacoustic_resonator.io.persistence import load_simulation_checkpoint
    from neuroacoustic_resonator.io.resume import resume_simulation_checkpoint

    start_step = load_simulation_checkpoint(args.checkpoint).step_index
    paths = resume_simulation_checkpoint(
        args.checkpoint,
        steps=args.steps,
        output=args.output,
    )
    print(
        f"Resumed checkpoint: {args.checkpoint} "
        f"start_step={start_step} final_step={start_step + args.steps}"
    )
    print(f"Saved checkpoint: {paths.arrays_path} metadata={paths.metadata_path}")
    return 0


LOCAL: dict[str, Callable[[argparse.Namespace], int]] = {
    "preview": preview,
    "checkpoint": checkpoint,
    "resume": resume,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neuroacoustic",
        description="Neuroacoustic resonator commands.",
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")

    preview_parser = commands.add_parser(
        "preview", help="Run a short simulation and save a field preview."
    )
    preview_parser.add_argument(
        "--config", type=Path, default=Path("configs") / "default.yaml"
    )

    checkpoint_parser = commands.add_parser(
        "checkpoint", help="Run a simulation and save a checkpoint."
    )
    checkpoint_parser.add_argument(
        "--config", type=Path, default=Path("configs") / "default.yaml"
    )
    checkpoint_parser.add_argument("--steps", type=int, default=1_000)
    checkpoint_parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments") / "states" / "checkpoint.npz",
    )

    resume_parser = commands.add_parser(
        "resume", help="Continue a checkpoint into a new checkpoint."
    )
    resume_parser.add_argument("--checkpoint", type=Path, required=True)
    resume_parser.add_argument("--steps", type=int, default=1_000)
    resume_parser.add_argument("--output", type=Path, default=None)

    for name, (_, summary) in DELEGATED.items():
        commands.add_parser(name, help=summary, add_help=False)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] in DELEGATED:
        module = import_module(DELEGATED[arguments[0]][0])
        return int(module.main(arguments[1:]))
    args = build_parser().parse_args(arguments)
    return LOCAL[args.command](args)
