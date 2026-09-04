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
