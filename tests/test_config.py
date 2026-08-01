import pytest
from pydantic import ValidationError

from neuroacoustic_resonator import SimulationConfig


def test_simulation_config_loads_from_yaml(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
field:
  size: 6
  seed: 12
  memory_drive_strength: 0.15
  memory_drive_assoc_gain: 1.5
steps: 5
preview_path: outputs/test-preview.png
""",
        encoding="utf-8",
    )

    config = SimulationConfig.from_file(config_path)

    assert config.field.size == 6
    assert config.field.seed == 12
    assert config.field.memory_drive_strength == 0.15
    assert config.field.memory_drive_assoc_gain == 1.5
    assert config.field.metabolite_diffusion == 0.0
    assert config.field.boundary_x == "periodic"
    assert config.field.boundary_y == "periodic"
    assert not config.synthetic_input.enabled
    assert config.protocol.frame_interval_steps == 1
    assert config.protocol.confirmation_frames == 3
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

    simulation = SimulationConfig.from_file(config_path).create_simulation()
    frame = simulation.run(2)[-1]

    assert frame.metrics.step == 2
    assert frame.state.phase.shape == (5, 5)


@pytest.mark.parametrize(
    "config_name",
    [
        "default.yaml",
        "field_only.yaml",
        "synthetic_input.yaml",
        "long_run.yaml",
    ],
)
def test_project_configs_load(config_name) -> None:
    config = SimulationConfig.from_file(f"configs/{config_name}")

    assert config.field.size > 1
    assert config.protocol.frame_interval_steps >= 1
    assert config.steps >= 1


def test_protocol_config_creates_runtime_configs(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
protocol:
  frame_interval_steps: 4
  activity_threshold: 0.1
  confidence_threshold: 0.5
  confirmation_frames: 5
  minimum_active_frames: 6
  hysteresis_margin: 0.2
  novelty_threshold: 0.3
""",
        encoding="utf-8",
    )

    protocol = SimulationConfig.from_file(config_path).protocol
    encoder = protocol.to_encoder_config()
    detector = protocol.to_detector_config()

    assert encoder.frame_interval_steps == 4
    assert detector.activity_threshold == 0.1
    assert detector.confidence_threshold == 0.5
    assert detector.confirmation_frames == 5
    assert detector.minimum_active_frames == 6
    assert detector.hysteresis_margin == 0.2
    assert detector.novelty_threshold == 0.3


def test_protocol_config_rejects_invalid_values(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
protocol:
  frame_interval_steps: 0
""",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        SimulationConfig.from_file(config_path)
