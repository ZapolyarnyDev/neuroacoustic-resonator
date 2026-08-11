from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from neuroacoustic_resonator.analysis.distinguishability_corpus import (
    DistinguishabilityCorpusConfig,
)
from neuroacoustic_resonator.analysis.stage_one_evidence import (
    EvidenceGateThresholds,
    RobustnessScenarioConfig,
    ScenarioCorpusOverrides,
    ScenarioFieldOverrides,
    StageOneEvidenceConfig,
    evaluate_scenario_gate,
    evaluate_stage_one_gate,
    prepare_stage_one,
)
from neuroacoustic_resonator.configuration import SimulationConfig


def stage_config(tmp_path: Path) -> StageOneEvidenceConfig:
    return StageOneEvidenceConfig(
        corpus_config=Path("configs/distinguishability_corpus.yaml"),
        output_root=tmp_path / "matrix",
        output_report=tmp_path / "stage-one.json",
        scenarios=(
            RobustnessScenarioConfig(name="baseline"),
            RobustnessScenarioConfig(
                name="coupling-low",
                field=ScenarioFieldOverrides(coupling_strength=0.128),
            ),
            RobustnessScenarioConfig(
                name="drive-high",
                corpus=ScenarioCorpusOverrides(drive_strength=0.54),
            ),
        ),
        thresholds=EvidenceGateThresholds(min_scenario_pass_rate=1.0),
        bootstrap_samples=10,
        permutation_samples=10,
    )


def passing_reports() -> tuple[dict[str, Any], dict[str, Any]]:
    separation: dict[str, Any] = {
        "overall": {
            "available": True,
            "seed_roots": [101, 211, 307, 401, 503, 601, 701],
            "separation_margin": 1.5,
            "separation_margin_ci": [0.5, 2.0],
            "cliffs_delta": 0.8,
            "cliffs_delta_ci": [0.4, 1.0],
        }
    }
    classification: dict[str, Any] = {
        "test": {
            "seed_roots": [601, 701],
            "balanced_accuracy": 0.8,
            "balanced_accuracy_ci": [0.6, 0.9],
            "per_class_recall": {"silence": 0.75},
        },
        "permutation_baseline": {
            "balanced_accuracy_ci": [0.1, 0.3],
            "p_value": 0.01,
        },
    }
    return separation, classification


def test_prepare_stage_one_materializes_isolated_scenario_configs(
    tmp_path: Path,
) -> None:
    config = stage_config(tmp_path)

    scenarios = prepare_stage_one(config)

    baseline_simulation = SimulationConfig.from_file(scenarios[0].simulation_config)
    coupling_simulation = SimulationConfig.from_file(scenarios[1].simulation_config)
    drive_corpus = DistinguishabilityCorpusConfig.from_file(scenarios[2].corpus_config)
    assert [scenario.name for scenario in scenarios] == [
        "baseline",
        "coupling-low",
        "drive-high",
    ]
    assert baseline_simulation.field.coupling_strength == 0.16
    assert coupling_simulation.field.coupling_strength == 0.128
    assert drive_corpus.drive_strength == 0.54
    assert drive_corpus.output_manifest == scenarios[2].manifest
    assert drive_corpus.simulation_config == scenarios[2].simulation_config


def test_scenario_gate_requires_separation_classification_and_silence() -> None:
    separation, classification = passing_reports()
    thresholds = EvidenceGateThresholds()

    passing = evaluate_scenario_gate(separation, classification, thresholds)
    classification["permutation_baseline"]["p_value"] = 0.2
    failing = evaluate_scenario_gate(separation, classification, thresholds)

    assert passing["passed"]
    assert all(passing["checks"].values())
    assert not failing["passed"]
    assert not failing["checks"]["permutation_p_value"]


def test_stage_gate_requires_every_robustness_scenario(tmp_path: Path) -> None:
    config = stage_config(tmp_path)
    scenarios = prepare_stage_one(config)
    for scenario in scenarios:
        scenario.result.write_text(
            json.dumps(
                {
                    "scenario": scenario.name,
                    "gate": {"passed": True},
                }
            ),
            encoding="utf-8",
        )

    report = evaluate_stage_one_gate(config, scenarios)

    assert report["status"] == "passed"
    assert report["passed"]
    assert report["scenario_pass_rate"] == 1.0


def test_default_stage_matrix_covers_moderate_parameter_changes() -> None:
    config = StageOneEvidenceConfig.from_file(
        "configs/distinguishability_stage_one.yaml"
    )

    assert {scenario.name for scenario in config.scenarios} == {
        "baseline",
        "coupling-low",
        "coupling-high",
        "drive-low",
        "drive-high",
        "frequency-spread-low",
        "frequency-spread-high",
    }
