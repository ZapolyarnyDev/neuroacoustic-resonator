from pathlib import Path

from neuroacoustic_resonator import FieldConfig, Simulation, save_field_preview


def main():
    simulation = Simulation(FieldConfig(size=16, seed=67))
    frame = simulation.run(32)[-1]
    preview_path = save_field_preview(frame, Path("outputs") / "field-preview.png")
    print(
        "Neuroacoustic resonator field initialized: "
        f"size={frame.state.phase.shape[0]}x{frame.state.phase.shape[1]}, "
        f"mean_metabolite={frame.metrics.mean_metabolite:.3f}, "
        f"global_synchrony={frame.metrics.global_synchrony:.3f}, "
        f"preview={preview_path}"
    )


if __name__ == "__main__":
    main()
