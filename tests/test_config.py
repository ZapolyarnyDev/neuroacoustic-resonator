import pytest
from pydantic import ValidationError

from neuroacoustic_resonator import Simulation, SimulationConfig


def test_simulation_config_loads_from_yaml(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
field:
  size: 6
  seed: 12
steps: 5
preview_path: outputs/test-preview.png
""",
        encoding="utf-8",
    )

    config = SimulationConfig.from_file(config_path)

    assert config.field.size == 6
    assert config.field.seed == 12
    assert config.field.metabolite_diffusion == 0.0
    assert not config.synthetic_input.enabled
    assert config.steps == 5
    assert config.preview_path.parts == ("outputs", "test-preview.png")


def test_simulation_config_rejects_unknown_keys(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("unknown: true\n", encoding="utf-8")

    with pytest.raises(ValidationError):
        SimulationConfig.from_file(config_path)


def test_simulation_can_start_from_config_file(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
field:
  size: 5
steps: 2
""",
        encoding="utf-8",
    )

    simulation = Simulation.from_config_file(config_path)
    frame = simulation.run(2)[-1]

    assert frame.metrics.step == 2
    assert frame.state.phase.shape == (5, 5)


@pytest.mark.parametrize(
    "config_name",
    [
        "default.yaml",
        "field_only.yaml",
        "synthetic_input.yaml",
        "audio_demo.yaml",
        "long_run.yaml",
    ],
)
def test_project_configs_load(config_name) -> None:
    config = SimulationConfig.from_file(f"configs/{config_name}")

    assert config.field.size > 1
    assert config.steps >= 1
