from neuroacoustic_resonator import FieldConfig, Simulation, save_phase_preview


def test_save_phase_preview_creates_png(tmp_path) -> None:
    simulation = Simulation(FieldConfig(size=4, seed=1))
    frame = simulation.step()

    output_path = save_phase_preview(frame, tmp_path / "phase.png")

    assert output_path.exists()
    assert output_path.suffix == ".png"
    assert output_path.stat().st_size > 0
