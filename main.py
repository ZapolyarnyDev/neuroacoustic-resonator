from neuroacoustic_resonator import FieldConfig, Simulation


def main():
    simulation = Simulation(FieldConfig(size=16, seed=67))
    frame = simulation.step()
    print(
        "Neuroacoustic resonator field initialized: "
        f"size={frame.state.phase.shape[0]}x{frame.state.phase.shape[1]}, "
        f"mean_metabolite={frame.metrics.mean_metabolite:.3f}, "
        f"global_synchrony={frame.metrics.global_synchrony:.3f}"
    )


if __name__ == "__main__":
    main()
