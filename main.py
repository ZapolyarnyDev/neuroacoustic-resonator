from pathlib import Path

from neuroacoustic_resonator import Simulation, SimulationConfig, save_field_preview


def main():
    config = SimulationConfig.from_file(Path("configs") / "default.yaml")
    simulation = Simulation.from_config(config)
    frame = simulation.run(config.steps)[-1]
    preview_path = save_field_preview(frame, config.preview_path)
    print(
        "Neuroacoustic resonator field initialized: "
        f"size={frame.state.phase.shape[0]}x{frame.state.phase.shape[1]}, "
        f"mean_metabolite={frame.metrics.mean_metabolite:.3f}, "
        f"global_synchrony={frame.metrics.global_synchrony:.3f}, "
        f"preview={preview_path}"
    )


if __name__ == "__main__":
    main()
