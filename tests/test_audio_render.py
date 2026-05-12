import wave

import pytest

from neuroacoustic_resonator.audio_render import (
    main,
    render_audio_demo,
    steps_for_duration,
)


def test_steps_for_duration_rounds_up_to_full_frames() -> None:
    assert steps_for_duration(1.0, sample_rate=10, frame_size=4) == 3


def test_render_audio_demo_writes_wav(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    output_path = tmp_path / "demo.wav"
    config_path.write_text(
        """
field:
  size: 4
  seed: 1
steps: 2
""",
        encoding="utf-8",
    )

    written_path = render_audio_demo(
        config_path,
        sample_rate=8_000,
        frame_size=16,
        output_path=output_path,
    )

    with wave.open(str(written_path), "rb") as stream:
        assert stream.getframerate() == 8_000
        assert stream.getnframes() == 32


def test_render_audio_demo_rejects_invalid_steps(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("field:\n  size: 4\nsteps: 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="steps"):
        render_audio_demo(config_path, steps=0)


def test_render_audio_demo_rejects_steps_and_duration_together(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("field:\n  size: 4\nsteps: 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="either steps or duration_seconds"):
        render_audio_demo(config_path, steps=1, duration_seconds=1.0)


def test_audio_render_main_defaults_to_audible_duration(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    output_path = tmp_path / "demo.wav"
    config_path.write_text(
        """
field:
  size: 4
  seed: 1
steps: 1
""",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--config",
            str(config_path),
            "--sample-rate",
            "100",
            "--frame-size",
            "10",
            "--output",
            str(output_path),
        ]
    )

    with wave.open(str(output_path), "rb") as stream:
        assert stream.getnframes() == 500
    assert exit_code == 0


def test_audio_render_main_writes_wav(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    output_path = tmp_path / "demo.wav"
    config_path.write_text(
        """
field:
  size: 4
  seed: 1
steps: 1
""",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--config",
            str(config_path),
            "--steps",
            "2",
            "--sample-rate",
            "8000",
            "--frame-size",
            "8",
            "--carrier-frequency",
            "110",
            "--output",
            str(output_path),
        ]
    )

    with wave.open(str(output_path), "rb") as stream:
        assert stream.getnframes() == 16
    assert exit_code == 0
