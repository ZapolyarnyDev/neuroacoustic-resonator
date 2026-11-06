from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from neuroacoustic_resonator.analysis.baselines import variant_corpus_config
from neuroacoustic_resonator.analysis.controlled_equilibration import (
    ControlledEquilibrationCorpusConfig,
)
from neuroacoustic_resonator.analysis.field_parts import (
    FIELD_PARTS,
    NO_EQUILIBRATION,
    cut_output_region,
    format_row,
    signal_strength,
)
from neuroacoustic_resonator.analysis.protocol_embeddings import (
    EmbeddingRow,
    write_embedding_rows,
)
from neuroacoustic_resonator.core.field import FieldConfig
from neuroacoustic_resonator.core.regions import RegionMasks
from neuroacoustic_resonator.core.simulation import Simulation

CONFIG = Path("configs/controlled_equilibration.yaml")


def test_field_parts_have_unique_slugs_and_known_parameters() -> None:
    known = set(FieldConfig.__dataclass_fields__)

    assert len({part.slug for part in FIELD_PARTS}) == len(FIELD_PARTS)
    for part in FIELD_PARTS:
        assert set(part.field_updates) <= known


def test_variant_config_applies_field_updates_and_equilibration(tmp_path) -> None:
    config = ControlledEquilibrationCorpusConfig.from_file(CONFIG)

    variant = ControlledEquilibrationCorpusConfig.from_file(
        variant_corpus_config(
            config,
            tmp_path / "variant",
            field_updates={"metabolite_cost": 0.0},
            equilibration=NO_EQUILIBRATION,
        )
    )

    field = yaml.safe_load(variant.simulation_config.read_text(encoding="utf-8"))
    assert field["field"]["metabolite_cost"] == 0.0
    assert field["field"]["coupling_strength"] == pytest.approx(0.16)
    assert variant.equilibration.neutral_steps == 0
    assert variant.equilibration.metabolite_baseline is None
    assert variant.output_dir == tmp_path / "variant"


def test_cut_output_region_zeroes_only_output_coupling() -> None:
    simulation = Simulation(FieldConfig(size=10, seed=3))
    output = RegionMasks.from_size(10).output

    cut_output_region(simulation)

    coupling = simulation.field.state.coupling
    assert np.all(coupling[output] == 0.0)
    assert np.all(coupling[~output] == pytest.approx(0.16))


def test_cut_output_region_stops_neighbor_drive_into_output() -> None:
    simulation = Simulation(
        FieldConfig(
            size=10,
            seed=3,
            frequency_spread=0.0,
            frequency_plasticity_rate=0.0,
            frequency_homeostasis_rate=0.0,
            coupling_homeostasis_rate=0.0,
        )
    )
    output = RegionMasks.from_size(10).output
    cut_output_region(simulation)
    before = simulation.field.state.phase[output]

    simulation.field.step()

    advance = np.mod(simulation.field.state.phase[output] - before, 2.0 * np.pi)
    assert advance == pytest.approx(np.full(advance.shape, 0.02))


def write_rows(path: Path, values: dict[str, list[float]]) -> Path:
    labels = ("tone", "chirp", "pulse", "noise")
    rows: list[EmbeddingRow] = []
    for index in range(8):
        row: EmbeddingRow = {
            "trial_id": f"t{index}",
            "stimulus_label": labels[index % 4],
            "source_type": labels[index % 4],
            "seed_root": 101,
            "field_seed": index,
            "repeat_index": 1,
            "split": "train",
            "protocol_version": "0.1",
            "response_frames": 8,
        }
        row.update({name: column[index] for name, column in values.items()})
        rows.append(row)
    write_embedding_rows(path, rows, feature_columns=tuple(values))
    return path


def test_signal_strength_measures_stimulus_share(tmp_path) -> None:
    separated = write_rows(
        tmp_path / "separated.csv",
        {"a": [0.0, 1.0, 2.0, 3.0] * 2, "zero": [0.0] * 8},
    )
    silent = write_rows(tmp_path / "silent.csv", {"zero": [0.0] * 8})

    assert signal_strength(separated) == {"stimulus_eta": 1.0, "changing_features": 1}
    assert signal_strength(silent) == {"stimulus_eta": 0.0, "changing_features": 0}


def test_format_row_marks_missing_signal() -> None:
    row = format_row("full field", {"no_signal": True, "changing_features": 0})

    assert row == "| full field | no signal | — | — | 0.000 | 0 |"
