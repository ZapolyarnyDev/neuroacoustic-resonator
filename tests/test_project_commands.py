from __future__ import annotations

import re
from pathlib import Path


def test_readme_documents_every_public_just_recipe() -> None:
    justfile = Path("Justfile").read_text(encoding="utf-8")
    readme = Path("README.md").read_text(encoding="utf-8")
    recipes = {
        match.group(1)
        for match in re.finditer(r"^([a-z][a-z0-9-]*)(?:\s+[^:]*)?:$", justfile, re.M)
    }
    recipes.remove("default")

    missing = sorted(recipe for recipe in recipes if f"just {recipe}" not in readme)

    assert not missing, f"README.md does not document Just recipes: {missing}"


def test_current_command_surface_excludes_archived_workflows() -> None:
    justfile = Path("Justfile").read_text(encoding="utf-8")
    readme = Path("README.md").read_text(encoding="utf-8")
    archived = (
        "stage-one",
        "pattern-calibration",
        "propagation-probe",
        "voice-memory-probe",
    )

    assert all(f"{name}:" not in justfile for name in archived)
    assert all(f"just {name}" not in readme for name in archived)


def _write_small_config(path: Path) -> Path:
    path.write_text(
        "field:\n  size: 4\n  seed: 1\nsteps: 3\n"
        f"preview_path: {(path.parent / 'preview.png').as_posix()}\n",
        encoding="utf-8",
    )
    return path


def test_cli_preview_writes_image(tmp_path) -> None:
    from neuroacoustic_resonator.cli.main import main

    config = _write_small_config(tmp_path / "config.yaml")

    assert main(["preview", "--config", str(config)]) == 0
    assert (tmp_path / "preview.png").exists()


def test_cli_checkpoint_and_resume_advance_steps(tmp_path) -> None:
    from neuroacoustic_resonator.cli.main import main
    from neuroacoustic_resonator.io.persistence import load_checkpoint_metadata

    config = _write_small_config(tmp_path / "config.yaml")
    saved = tmp_path / "saved.npz"
    resumed = tmp_path / "resumed.npz"

    save_args = ["--config", str(config), "--steps", "2", "--output", str(saved)]
    resume_args = ["--checkpoint", str(saved), "--steps", "3", "--output", str(resumed)]

    assert main(["checkpoint", *save_args]) == 0
    assert main(["resume", *resume_args]) == 0
    assert load_checkpoint_metadata(resumed)["step_index"] == 5


def test_cli_delegates_arguments_to_module_main(monkeypatch) -> None:
    from neuroacoustic_resonator.analysis import research_report
    from neuroacoustic_resonator.cli.main import main

    received: list[list[str]] = []

    def fake_main(argv: list[str] | None = None) -> int:
        received.append(list(argv or []))
        return 0

    monkeypatch.setattr(research_report, "main", fake_main)

    assert main(["report", "--example", "value"]) == 0
    assert received == [["--example", "value"]]


def test_cli_delegated_modules_expose_main() -> None:
    from importlib import import_module

    from neuroacoustic_resonator.cli.main import DELEGATED

    for module_name, _ in DELEGATED.values():
        assert callable(import_module(module_name).main)
