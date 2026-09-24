from __future__ import annotations

import json
from pathlib import Path

import pytest

from neuroacoustic_resonator.analysis.equilibration_ablations import (
    SCENARIOS,
    EquilibrationAblationConfig,
    scenario_corpus_config,
    summarize_ablation_study,
)


def test_ablation_changes_one_equilibration_component_at_a_time() -> None:
    study = EquilibrationAblationConfig.from_file(
        "configs/controlled_equilibration_ablations.yaml"
    )
    configurations = {name: scenario_corpus_config(study, name) for name in SCENARIOS}
    full = configurations["full"].equilibration.model_dump()

    assert study.seed_roots == (101, 211, 307, 401, 601)
    assert study.repeats == 2
    assert full == {
        "neutral_steps": 512,
        "phase_damping": 0.2,
        "metabolite_baseline": 1.0,
    }
    assert configurations["no_warmup"].equilibration.neutral_steps == 0
    assert configurations["no_phase_damping"].equilibration.phase_damping == 0.0
    assert (
        configurations["no_metabolite_baseline"].equilibration.metabolite_baseline
        is None
    )
    for name, config in configurations.items():
        changed = {
            key
            for key, value in config.equilibration.model_dump().items()
            if value != full[key]
        }
        assert len(changed) == (0 if name == "full" else 1)
        assert config.seed_roots == study.seed_roots
        assert config.repeats == study.repeats
        assert config.output_dir == study.output_dir / name
        assert len(config.stimuli) == 5


def test_ablation_summary_selects_from_complete_comparable_scenarios(
    tmp_path: Path,
) -> None:
    study = EquilibrationAblationConfig.from_file(
        "configs/controlled_equilibration_ablations.yaml"
    ).model_copy(update={"output_dir": tmp_path})
    accuracies = {
        "full": 0.5,
        "no_warmup": 0.4,
        "no_phase_damping": 0.6,
        "no_metabolite_baseline": 0.3,
    }
    for name in SCENARIOS:
        output = tmp_path / name
        output.mkdir()
        (output / "summary.json").write_text(
            json.dumps(
                {
                    "branch_trials": 50,
                    "causal_pairs": 40,
                    "checkpoints": 10,
                    "fingerprints_verified": True,
                }
            ),
            encoding="utf-8",
        )
        (output / "regional_causal_report.json").write_text(
            json.dumps(
                {
                    "representations": {
                        "output": {
                            "balanced_accuracy": accuracies[name],
                            "roots_above_chance": 4,
                            "bootstrap_ci": [0.3, 0.7],
                            "permutation_p_value": 0.01,
                            "stimulus_fraction": 0.2,
                            "field_seed_fraction": 0.4,
                            "stimulus_to_seed_ratio": 0.5,
                            "separation_margin": 1.0,
                            "cliffs_delta": 0.1,
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        (output / "causal_output_evidence.json").write_text(
            json.dumps(
                {
                    "design": {"seed_roots": sorted(study.seed_roots)},
                    "classification": {
                        "by_checkpoint": [
                            {"field_seed": root * 10 + repeat}
                            for root in study.seed_roots
                            for repeat in (1, 2)
                        ],
                        "by_seed_root": [
                            {"seed_root": root, "balanced_accuracy": accuracies[name]}
                            for root in study.seed_roots
                        ],
                    },
                }
            ),
            encoding="utf-8",
        )

    report = summarize_ablation_study(study)

    assert report["selected_scenario"] == "no_phase_damping"
    assert report["scenarios"]["no_phase_damping"][
        "accuracy_difference_from_full"
    ] == pytest.approx(0.1)
    assert report["confirmatory"] is False
    assert (tmp_path / "ablation_report.json").exists()
