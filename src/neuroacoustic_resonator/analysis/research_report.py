from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from neuroacoustic_resonator.analysis.controlled_equilibration import (
    ControlledEquilibrationCorpusConfig,
)
from neuroacoustic_resonator.analysis.provenance import git_state

REGIONS = ("input", "assoc", "output", "combined")


def write_research_report(
    config_path: str | Path,
    reports_dir: str | Path,
    *,
    repository: str | Path = ".",
) -> Path:
    config = ControlledEquilibrationCorpusConfig.from_file(config_path)
    summary = json.loads(config.output_summary.read_text(encoding="utf-8"))
    provenance = summary.get("provenance")
    if not isinstance(provenance, dict):
        msg = f"summary has no provenance: {config.output_summary}"
        raise ValueError(msg)
    require_current_commit(provenance, git_state(repository))
    created = datetime.fromisoformat(provenance["created_at"])
    commit = provenance["git"]["commit"]
    output = Path(reports_dir) / f"{created.date().isoformat()}-{commit[:7]}.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_report(summary), encoding="utf-8")
    return output


def require_current_commit(
    provenance: Mapping[str, Any],
    current: Mapping[str, Any],
) -> None:
    recorded = provenance["git"]
    if recorded.get("commit") is None or current.get("commit") is None:
        msg = "report requires results and a working tree inside a git repository"
        raise ValueError(msg)
    if recorded.get("dirty"):
        msg = "results were produced from a dirty working tree"
        raise ValueError(msg)
    if current.get("dirty"):
        msg = "working tree has uncommitted changes"
        raise ValueError(msg)
    if recorded["commit"] != current["commit"]:
        msg = (
            f"results belong to {recorded['commit'][:7]}, "
            f"HEAD is {current['commit'][:7]}"
        )
        raise ValueError(msg)


def render_report(summary: Mapping[str, Any]) -> str:
    provenance = summary["provenance"]
    git = provenance["git"]
    environment = provenance["environment"]
    packages = environment["packages"]
    classification = summary["cross_seed_classification"]
    bootstrap = classification["checkpoint_clustered_bootstrap"]
    permutation = classification["checkpoint_label_permutation"]
    variance = summary["causal_difference"]
    created = datetime.fromisoformat(provenance["created_at"])
    configs = ", ".join(
        f"{Path(item['path']).name} `{item['sha256'][:7]}`"
        for item in provenance["configs"]
    )
    arguments = ", ".join(
        f"{name} {value}" for name, value in provenance["arguments"].items()
    )
    roots = " / ".join(
        f"{item['balanced_accuracy']:.3f}" for item in classification["by_seed_root"]
    )
    low, high = bootstrap["balanced_accuracy_ci"]
    lines = [
        f"# Controlled equilibration — {created.date().isoformat()} · {git['commit'][:7]}",
        "",
        f"Ветка `{git['branch']}` · чистое дерево · "
        f"Python {environment['python']}, numpy {packages['numpy']}",
        f"Конфиги: {configs}",
        f"Команда: `{provenance['command']}` ({arguments})",
        "",
        "| Метрика | Значение |",
        "|---|---|",
        "| Balanced accuracy (output pattern) | "
        f"{classification['balanced_accuracy']:.3f} "
        f"(chance {classification['chance_level']:.2f}) |",
        f"| 95% CI | [{low:.3f}, {high:.3f}] |",
        f"| p (перестановки) | {format_p(permutation['p_value'])} |",
        f"| По seed roots | {roots} |",
        "| Доля дисперсии: стимул / seed | "
        f"{variance['stimulus_fraction']:.3f} / {variance['field_seed_fraction']:.3f} |",
        "",
        "| Регион | Accuracy | p |",
        "|---|---|---|",
    ]
    representations = summary["regional_representations"]
    for region in REGIONS:
        item = representations.get(region)
        if item is None:
            continue
        lines.append(
            f"| {region} | {item['balanced_accuracy']:.3f} | "
            f"{format_p(item['permutation_p_value'])} |"
        )
    return "\n".join(lines) + "\n"


def format_p(value: float) -> str:
    return f"{value:.2g}" if value >= 0.01 else f"{value:.1g}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write the key research report for the current commit."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/controlled_equilibration.yaml"),
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("experiments/reports"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = write_research_report(args.config, args.reports_dir)
    print(output)
    return 0
