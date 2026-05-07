from neuroacoustic_resonator import FieldConfig, OscillatorField


def main():
    field = OscillatorField(FieldConfig(size=16, seed=67))
    state = field.step()
    print(
        "Neuroacoustic resonator field initialized: "
        f"size={state.phase.shape[0]}x{state.phase.shape[1]}, "
        f"mean_metabolite={state.metabolite.mean():.3f}"
    )


if __name__ == "__main__":
    main()
